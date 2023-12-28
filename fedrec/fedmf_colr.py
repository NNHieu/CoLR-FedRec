from collections import OrderedDict
from typing import Any

import math
import copy

from rec.models import MF
from fedrec.core.models import FedRecModel
from .core.strategies import FedRecAvgParameters
import torch
from torch import nn
import torch.nn.functional as F

# Adapted from https://github.com/microsoft/LoRA
class CoLREmbedding(nn.Embedding):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        r: int = 0,
        **kwargs
    ):
        assert r > 0
        nn.Embedding.__init__(self, num_embeddings, embedding_dim, **kwargs)
        self._r = r
        self._A = nn.Parameter(self.weight.new_zeros((num_embeddings, r)))
        self._B = nn.Parameter(self.weight.new_zeros((r, embedding_dim)))
        self.register_buffer('_scaling', torch.tensor([1]))
        # Freeze weight and B matrices
        self.weight.requires_grad = False
        self._B.requires_grad = False
        self.merged = False
        self.reset_parameters()

    def reset_parameters(self):
        nn.Embedding.reset_parameters(self)
        if hasattr(self, '_r'):
            self.reset_lowrank_parameters(init_B_strategy='zeros', keep_B=False)
    
    @torch.no_grad()
    def reset_lowrank_parameters(self, init_B_strategy, keep_B=False, scale_norm=1.):
        # initialize A the same way as the default for nn.Linear and B to zero
        nn.init.zeros_(self._A)
        if not keep_B:
            if init_B_strategy == "random":
                nn.init.normal_(self._B, mean=0, std=scale_norm*math.sqrt(1/ self._r))
            elif init_B_strategy == "l2norm":
                nn.init.normal_(self._B)
                self._B /= torch.linalg.norm(self._B, dim=1, keepdim=True)
                self._B *= scale_norm
            elif init_B_strategy == "orthnorm":
                nn.init.normal_(self._B)
                U, S, Vh = torch.linalg.svd(self._B.data, full_matrices=False)
                self._B.data = (U @ Vh) * math.sqrt(self.embedding_dim / self._r)
                self._B *= scale_norm
            elif init_B_strategy == 'zeros':
                nn.init.zeros_(self._B)
            else:
                raise ValueError("Unknown init_B_strategy: %s" % init_B_strategy)
        self.merged = False
    
    def merge_lowrank_weights(self):
        self.weight.data += (self._A @ self._B) * self._scaling
        self.merged = True
    
    def forward(self, x: torch.Tensor):
        result = nn.Embedding.forward(self, x)
        if not self.merged:
            after_A = F.embedding(
                x, self._A, self.padding_idx, self.max_norm,
                self.norm_type, self.scale_grad_by_freq, self.sparse
            )
            result += (after_A @ self._B) * self._scaling
            return result
        else:
            return result

class CoLRMF(MF):
    def __init__(self, user_num, item_num, gmf_emb_size=16, rank=4):
        ItemEmbLayer = lambda num_emb, emb_dim: CoLREmbedding(num_emb, emb_dim, r=rank, scale_grad_by_freq=True)
        super().__init__(user_num, item_num, gmf_emb_size, ItemEmbedding=ItemEmbLayer)
        # We rescale grad of the A martix base on an analysis of estimation error
        scale_grad_A = math.sqrt(self.embed_item_GMF._r / self.embed_item_GMF.embedding_dim)
        self.embed_item_GMF._A.register_hook(lambda grad: grad * scale_grad_A)
    
    def _merge_all_lowrank_weights(self):
        self.embed_item_GMF.merge_lowrank_weights()
    
    def _reset_all_lowrank_weights(self,  init_B_strategy, keep_B=False):
        self.embed_item_GMF.reset_lowrank_parameters(init_B_strategy=init_B_strategy, keep_B=keep_B)

class FedMFCoLRAvgParameters(FedRecAvgParameters):
    '''
    Wrapper class for public parameters. This class implements basic operators required by FedAvg. 
    '''
    
    def __init__(self, shareable, interaction_mask) -> None:
        super().__init__()
        self._shareable_params_tree = shareable
        self._meta = {
            'interaction_mask': interaction_mask
        }
    
    @classmethod
    def zero_like(cls, sample):
        new_zeros_params = {}
        new_zeros_params['embed_item_GMF.weight'] = sample._shareable_params_tree['embed_item_GMF.weight']
        new_zeros_params['embed_item_GMF._A'] = torch.zeros_like(sample._shareable_params_tree['embed_item_GMF._A'])
        new_zeros_params['embed_item_GMF._B'] = sample._shareable_params_tree['embed_item_GMF._B']
        new_zeros_params['embed_item_GMF._scaling'] = sample._shareable_params_tree['embed_item_GMF._scaling']

        return FedMFCoLRAvgParameters(new_zeros_params, 
                                  torch.zeros_like(sample._meta['interaction_mask']))
    
    @property
    def meta(self):
        return self._meta

    def add_(self, other, weight=1.):
        self._shareable_params_tree['embed_item_GMF._A'] += other._shareable_params_tree['embed_item_GMF._A']
        self._meta['interaction_mask'] += other._meta['interaction_mask']
    
    def div_(self, weight):

        visited_item = self._meta['interaction_mask']
        visited_item[visited_item == 0] = 1
        self._shareable_params_tree['embed_item_GMF._A'].div_(visited_item.unsqueeze(-1))
    
    def sub_(self, other):
        self._shareable_params_tree['embed_item_GMF._A'].sub_(other._shareable_params_tree['embed_item_GMF._A'])
    
    def to_dict(self):
        return self._shareable_params_tree
    
    def numel(self):
        # Since we only need to communicate the A matrix betweem the central server and participants
        return self._shareable_params_tree['embed_item_GMF._A'].numel()

class FedMFCoLR(FedRecModel):
    def __init__(self, item_num, gmf_emb_size=16, user_num=1, rank=4):
        super(FedRecModel, self).__init__()
        self.net = CoLRMF(user_num, item_num, gmf_emb_size, rank=rank)

        # This buffer keep track of visited item during the local training phase 
        # and is used for aggregation step at the server
        # We encrypt this buffer with HE before sending it to the server
        self.register_buffer('inter_mask', torch.zeros(item_num, dtype=torch.long))

    def reinit_private_params(self):
        '''
        Reinitialize parameters at each client.
        '''
        self.net._init_weight_()
    
    def on_server_prepare(self, **kwargs) -> FedMFCoLRAvgParameters:
        '''
        Called from the central server at the begining of each training round.
        '''
        # Reinit the B matrix
        self.net._merge_all_lowrank_weights()
        self.net._reset_all_lowrank_weights(init_B_strategy='orthnorm', 
                                         keep_B=False)
        splitted_params = self.get_splited_params(dummy_private=True)
        return splitted_params['private'], splitted_params['public']

    def get_splited_params(self, dummy_private=False) -> dict[str, Any]:
        '''
        Split state dict into private and public param dict.
        '''
        public_params, private_params = {}, {}
        for key, val in self.net.state_dict().items():
            if 'user' in key:
                if dummy_private:
                    private_params[key] = torch.zeros_like(val)
                else:
                    private_params[key] = val.detach().clone()
            elif key == "inter_mask":
                continue
            else:
                public_params[key] = val.detach().clone()
        public_params = FedMFCoLRAvgParameters(public_params, self.inter_mask.data.clone())
        return {
            'private': private_params,
            'public': public_params
        }

    def set_state_from_splited_params(self, splited_params):
        '''
        Load state dict from private and public params
        '''
        private_params = splited_params[0]
        submit_params: FedMFCoLRAvgParameters = splited_params[1]
        params_dict = dict(private_params, **submit_params.to_dict())
        state_dict = OrderedDict(params_dict)
        self.net.load_state_dict(state_dict, strict=True)
        self.inter_mask.zero_()

    def forward(self, user, item, label=None, gamma=0.):
        self.inter_mask[item] = 1
        return self.net(user, item, label=label, gamma=gamma)

    def local_train(self, train_loader, optimizer, num_epochs, device, base_lr, wd, mask_zero_user_index=False):
        metrics = super().local_train(train_loader, optimizer, num_epochs, device, base_lr, wd, mask_zero_user_index)
        with torch.no_grad():
            user_emb_norms = self.net.embed_user_GMF.weight.norm(dim=-1).mean()
            item_emb_norms = self.net.embed_item_GMF.weight.norm(dim=-1).mean()
            metrics['user_emb_norms'] = user_emb_norms.item()
            metrics['item_emb_norms'] = item_emb_norms.item()
        return metrics
    
    @classmethod
    def merge_client_params(cls, clients, server_params, model, device):
        client_weights = [c._private_params.values() for c in clients]
        client_weights = [torch.cat(w, dim=0).to(device) for w in zip(*client_weights)]
        client_weights = {k: v for k,v in zip(clients[0]._private_params.keys(), client_weights)}
        eval_model = copy.deepcopy(model)
        eval_model.set_state_from_splited_params([clients[0]._private_params, server_params])
        eval_model.net.embed_user_GMF = torch.nn.Embedding.from_pretrained(client_weights['embed_user_GMF.weight'])
        return eval_model

    def reg_loss(self, item, user):
        '''
        Only apply weight decay on embeddings of selected items and users.
        '''
        gmf_item_emb = self.net.embed_item_GMF(item)
        gmf_user_emb = self.net.embed_user_GMF(user)
        
        item_emb_reg = (gmf_item_emb**2).sum()
        user_emb_reg = (gmf_user_emb**2).sum()

        reg_loss = item_emb_reg + user_emb_reg
        return reg_loss

