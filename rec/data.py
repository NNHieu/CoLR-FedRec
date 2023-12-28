from pathlib import Path
import numpy as np 
import pandas as pd 
import torch

from torch.utils.data import TensorDataset

import random

def get_neg_items(rating_df: pd.DataFrame, test_inter_dict=None):
    """return all negative items & 100 sampled negative items"""
    item_pool = set(rating_df['item'].unique())
    interactions = rating_df.groupby('user')['item'].apply(set).reset_index().rename(
        columns={'item': 'pos_items'})
    user_interaction_count = interactions[['user',]]
    user_interaction_count['num_interaction'] = interactions['pos_items'].apply(len)
    user_interaction_count = dict(zip(user_interaction_count['user'].values, user_interaction_count['num_interaction'].values.tolist()))

    pos_item_dict = dict(zip(interactions['user'].values, interactions['pos_items'].values))

    if test_inter_dict is not None:
        for u, test_items in test_inter_dict.items():
            pos_item_dict[u].update(test_items)
    return item_pool, pos_item_dict, user_interaction_count

class RecDataModule():
    def __init__(self, root, num_train_negatives=4) -> None:
        self.root = Path(root)
        self.num_train_negatives = num_train_negatives
    
    def process_test_data(self, test_df):
        test_data = []
        for index, row in test_df.iterrows():
            u = row['user']
            for i in row['neg_sample']:
                test_data.append((int(u), int(i), 0.0))
            test_data.append((int(u), int(row['pos_item']), 1.0))
        # test_data = np.array(test_data)
        return test_data

    def setup(self):
        self.post_train_df = pd.read_csv(self.root / 'train.csv')
        self.test_df = pd.read_csv(self.root / 'test.csv')

        self.num_users = max(self.post_train_df['user'].max(), self.test_df['user'].max()) + 1
        self.num_items = max(self.post_train_df['item'].max(), self.test_df['pos_item'].max()) + 1

        self.test_df['neg_sample'] = self.test_df['neg_sample'].apply(lambda x: [int(s) for s in x[1:-1].split(',')])
        test_inter_dict = self.test_df.apply(lambda x: x['neg_sample'] + [x['pos_item']], axis=1)
        test_inter_dict = dict(zip(self.test_df['user'].values, test_inter_dict.values))

        if "v2" in str(self.root):
            self.val_df = pd.read_csv(self.root / 'val.csv')
            self.val_df['neg_sample'] = self.val_df['neg_sample'].apply(lambda x: [int(s) for s in x[1:-1].split(',')])
            
            self.num_users = max(self.num_users, self.val_df['user'].max() + 1)
            self.num_items = max(self.num_items, self.val_df['pos_item'].max() + 1)
            
            val_inter_dict = self.val_df.apply(lambda x: x['neg_sample'] + [x['pos_item']], axis=1)
            val_inter_dict = dict(zip(self.val_df['user'].values, val_inter_dict.values))
            for u, test_items in val_inter_dict.items():
                test_inter_dict[u] = set(test_inter_dict[u]).union(test_items)
            self.val_data = self.process_test_data(self.val_df)

        else:
            self.val_data = None

        self.item_pool, self.pos_item_dict, self.user_interaction_count = get_neg_items(self.post_train_df, test_inter_dict)

        self.test_data = self.process_test_data(self.test_df)

        self.item_pool = tuple(self.item_pool)

        self.post_train_df['user'] = self.post_train_df['user'].astype(int)
        self.post_train_df['item'] = self.post_train_df['item'].astype(int)
        self.post_train_df['rating'] = self.post_train_df['rating'].astype(float)

    def _get_neg_items_of_user(self, u):
        return self.item_pool - self.pos_item_dict[u]

    def _sample_neg_per_user(self, u, num_negatives):
        # Sampling negative examples 
        sample_size = num_negatives*self.user_interaction_count[u]
        neg_samples = []
        while len(neg_samples) < sample_size:
            k = num_negatives
            sample = random.sample(self.item_pool, k)
            sample = set(sample) - self.pos_item_dict[u]
            neg_samples.extend(sample)
        neg_samples = neg_samples[:sample_size]
        return neg_samples
    
    def _sample_neg_traindf(self, num_negatives):
        # Sampling negative examples 
        neg_samples = {}
        for u in range(len(self.pos_item_dict)):
            neg_samples[u] = self._sample_neg_per_user(u, num_negatives)
        neg_rating_df = pd.DataFrame.from_records(list(neg_samples.items()), columns=['user', 'neg_sample'])
        neg_rating_df = neg_rating_df.explode('neg_sample').reset_index(drop=True)
        neg_rating_df['rating'] = 0.0
        neg_rating_df.columns = ['user', 'item', 'rating']
        neg_rating_df['user'] = neg_rating_df['user'].astype(int)
        neg_rating_df['item'] = neg_rating_df['item'].astype(int)
        neg_rating_df['rating'] = neg_rating_df['rating'].astype(float)

        return neg_rating_df

    def train_dataset(self, num_negatives=None):
        if num_negatives is None:
            num_negatives = self.num_train_negatives
        neg_train_df = self._sample_neg_traindf(num_negatives)
        train_rating_df = pd.concat([self.post_train_df, neg_train_df], ignore_index=True)
        data = train_rating_df[['user', 'item', 'rating']].values.tolist()
        for i in range(len(data)):
            data[i] = (int(data[i][0]), int(data[i][1]), float(data[i][2]))
        return data

    def test_dataset(self):
        return self.test_data

    def val_dataset(self):
        if self.val_data is None:
            return None
        return self.val_data

def get_datamodule(cfg):
    if cfg.DATA.name == "movielens":
        root = cfg.DATA.root + "/ml-1m"
        dm = RecDataModule(root=root, num_train_negatives=cfg.DATA.num_negatives)
    elif cfg.DATA.name == "pinterest":
        root = cfg.DATA.root + "/pinterest"
        dm = RecDataModule(root=root, num_train_negatives=cfg.DATA.num_negatives)
    else:
        raise ValueError
    return dm


