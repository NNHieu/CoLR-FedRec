import torch
import torch.utils.data as data
from rec.data import get_datamodule, RecDataModule
import pandas as pd

class FedDataModule(object):
    def __init__(self, cfg) -> None:
        self.cfg = cfg
        self.rec_datamodule: RecDataModule = get_datamodule(cfg)
    
    def setup(self):
        self.rec_datamodule.setup()
        self.num_users = self.rec_datamodule.num_users
        self.num_items = self.rec_datamodule.num_items

        pos_item_df = self.rec_datamodule.post_train_df.groupby('user').agg({'item': list}).reset_index()
        self.pos_item_dict = dict(zip(pos_item_df['user'], pos_item_df['item']))

    @staticmethod
    def _explode_sample(data_list, cid, client_sample, rating):
        for item in client_sample:
            data_list.append([int(cid), int(item), rating])

    def train_dataset(self, cid_list):
        '''
        Return user i's private dataset for each i in cid_list
        '''
        num_negatives = self.rec_datamodule.num_train_negatives
        data = []
        for cid in cid_list:
            self._explode_sample(data, cid, self.pos_item_dict[cid], 1.0)
        
        for cid in cid_list:
            neg_sample = self.rec_datamodule._sample_neg_per_user(cid, num_negatives=num_negatives)
            self._explode_sample(data, cid, neg_sample, 0.0)
        return data
    
    def test_dataloader(self):
        return data.DataLoader(self.rec_datamodule.test_dataset(), batch_size=1024, shuffle=False, num_workers=0)

    def train_dataloader(self, cid=None, for_eval=False):
        if cid is None:
            if for_eval:
                return data.DataLoader(self.rec_datamodule.train_dataset(), batch_size=1024, shuffle=False, num_workers=0)
            else:
                return data.DataLoader(self.rec_datamodule.train_dataset(), **self.cfg.DATALOADER)
        return data.DataLoader(self.train_dataset(cid), **self.cfg.DATALOADER)
    
    def val_dataloader(self):
        val_dataset = self.rec_datamodule.val_dataset()
        if val_dataset is None:
            return None
        return data.DataLoader(val_dataset, batch_size=1024, shuffle=False, num_workers=0)