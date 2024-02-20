from collections import OrderedDict
import math
from typing import Any

from functools import partial
import copy

import numpy as np
from rec.models import MF
from fedrec.core.models import FedRecModel
from .core.strategies import FedRecAvgParameters
import torch
import torch.nn.functional as F

class FedMFAvgParameters(FedRecAvgParameters):
    '''
    Wrapper class for public parameters. This class implements basic operators required by FedAvg. 
    '''
    
    def __init__(self, shareable, interaction_mask) -> None:
        super().__init__()
        self._shareable_params_tree = shareable
        self._meta = {
            'interaction_mask': interaction_mask
        }
        self._numel = self._shareable_params_tree['embed_item_GMF.weight'].numel()
    
    @classmethod
    def zero_like(cls, sample):
        new_zeros_params = {}
        new_zeros_params['embed_item_GMF.weight'] = torch.zeros_like(sample._shareable_params_tree['embed_item_GMF.weight'])
        return FedMFAvgParameters(new_zeros_params, 
                                  torch.zeros_like(sample._meta['interaction_mask']))
    
    @property
    def meta(self):
        return self._meta

    def add_(self, other, weight=1.):     
        self._shareable_params_tree['embed_item_GMF.weight'] += other._shareable_params_tree['embed_item_GMF.weight']
        self._meta['interaction_mask'] += other._meta['interaction_mask']
    
    def div_(self, weight):
        visited_item = self._meta['interaction_mask']
        visited_item[visited_item == 0] = 1
        self._shareable_params_tree['embed_item_GMF.weight'].div_(visited_item.unsqueeze(-1))
    
    def sub_(self, other):
        self._shareable_params_tree['embed_item_GMF.weight'].sub_(other._shareable_params_tree['embed_item_GMF.weight'])
    
    def to_dict(self):
        return self._shareable_params_tree
    
    def numel(self):
        return self._numel
    
    def pre_transfer(self, config):
        compressor_name = config.net.compressor.name
        if compressor_name == 'svd':
            # SVD Compresor
            rank = config.net.compressor.rank
            matrix = self._shareable_params_tree['embed_item_GMF.weight']
            with torch.no_grad():
                U, S, V = torch.svd(matrix)
                U, S, V = U[:, :rank], S[:rank], V[:, :rank]
            self._shareable_params_tree['embed_item_GMF.weight'] = U, S, V
            self._numel = U.numel() + S.numel() + V.numel()
        elif compressor_name == 'topk':
            # TopL Compresor
            ratio = config.net.compressor.ratio
            matrix = self._shareable_params_tree['embed_item_GMF.weight']
            k = math.ceil(ratio * matrix.numel())
            matrix = self._shareable_params_tree['embed_item_GMF.weight']
            shape = matrix.shape
            flatten_matrix = matrix.view(-1)
            _, idx = flatten_matrix.abs().topk(k)
            values = flatten_matrix[idx]
            self._shareable_params_tree['embed_item_GMF.weight'] = shape, idx, values
            self._numel = values.numel() + idx.numel()
        else:
            pass
    
    def post_transfer(self, config):
        compressor_name = config.net.compressor.name
        if compressor_name == 'svd':
            # SVD Compresor
            # low-rank decomposition via matmul
            U, S, V = self._shareable_params_tree['embed_item_GMF.weight']
            self._shareable_params_tree['embed_item_GMF.weight'] = (torch.mm(
                torch.mm(U, torch.diag(S)), V.t()))
        elif compressor_name == 'topk':
            # TopL Compresor
            shape, tensor_indices, tensor_data = self._shareable_params_tree['embed_item_GMF.weight']
            tensor = tensor_data.new_zeros(int(np.prod(shape)))
            tensor.index_copy_(0, tensor_indices, tensor_data)
            tensor = tensor.reshape(list(shape))
            self._shareable_params_tree['embed_item_GMF.weight'] = tensor
        else:
            pass
        self._numel = self._shareable_params_tree['embed_item_GMF.weight'].numel()

class FedMF(FedRecModel):
    def __init__(self, item_num, gmf_emb_size=16, user_num=1):
        super(FedRecModel, self).__init__()
        ItemEmbedding = partial(torch.nn.Embedding, scale_grad_by_freq=True)
        self.net = MF(user_num, item_num, gmf_emb_size, ItemEmbedding)

        # This buffer keep track of visited item during the local training phase 
        # and is used for aggregation step at the server
        # We encrypt this buffer with HE before sending it to the server
        self.register_buffer('inter_mask', torch.zeros(item_num, dtype=torch.long))

    def reinit_private_params(self):
        '''
        Reinitialize parameters at each client.
        '''
        self.net._init_weight_()
    
    def on_server_prepare(self, **kwargs) -> FedMFAvgParameters:
        '''
        Called from the central server at the begining of each training round.
        '''
        splitted_params = self.get_splited_params(dummy_private=True)
        return splitted_params['private'], splitted_params['public']

    def get_splited_params(self, dummy_private=False):
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
        public_params = FedMFAvgParameters(public_params, self.inter_mask.data.clone())
        return {
            'private': private_params,
            'public': public_params
        }

    def set_state_from_splited_params(self, splited_params):
        '''
        Load state dict from private and public params
        '''
        private_params = splited_params[0]
        submit_params: FedMFAvgParameters = splited_params[1]
        params_dict = dict(private_params, **submit_params.to_dict())
        state_dict = OrderedDict(params_dict)
        self.net.load_state_dict(state_dict, strict=True)
        self.inter_mask.zero_()

    def forward(self, user, item, label=None, gamma=0.):
        self.inter_mask[item] = 1
        return self.net(user, item, label=label, gamma=gamma)

    # def get_optimizer(self, config, opt_params=None):
    #     opt_params = [
    #         {'params': self.net.embed_item_GMF.parameters(), 'lr': config.TRAIN.lr},
    #         {'params': self.net.embed_user_GMF.parameters(), 'lr': config.TRAIN.lr},
    #     ]
    #     return super().get_optimizer(config, opt_params)
    
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

