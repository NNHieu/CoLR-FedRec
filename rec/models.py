import torch

import torch.nn as nn
import torch.nn.functional as F 

class MF(nn.Module):
    def __init__(self, user_num, item_num, gmf_emb_size=16, ItemEmbedding=nn.Embedding):
        """
        user_num: number of users
        item_num: number of items
        """		
        super(MF, self).__init__()

        self.embed_user_GMF = nn.Embedding(user_num, gmf_emb_size, scale_grad_by_freq=True)
        self.embed_item_GMF = ItemEmbedding(item_num, gmf_emb_size)
        self._init_weight_()

    def _init_weight_(self):
        """ We leave the weights initialization here. """
        d = self.embed_user_GMF.weight.shape[1]
        std = torch.sqrt(torch.tensor(1 / d))
        nn.init.normal_(self.embed_user_GMF.weight, std=std)
        nn.init.normal_(self.embed_item_GMF.weight, std=std)


    def forward(self, user, item, label=None, gamma=0., **kwargs):
        batch_size = user.shape[0]
        embed_user_GMF = self.embed_user_GMF(user)
        embed_item_GMF = self.embed_item_GMF(item)
        output_GMF = embed_user_GMF * embed_item_GMF
        prediction = output_GMF.sum(dim=-1)
        if label is None:
            return prediction.view(-1)
        loss = F.binary_cross_entropy_with_logits(prediction, label)
        if gamma > 0:
            item_emb_reg = (embed_item_GMF**2).sum() 
            user_emb_reg = (embed_user_GMF**2).sum()
            reg_loss = item_emb_reg + user_emb_reg
            reg = gamma * reg_loss / batch_size
            loss += reg
        return prediction.view(-1), loss
