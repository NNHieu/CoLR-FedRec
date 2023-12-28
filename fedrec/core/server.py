from typing import List, Tuple
from .client import ClientSampler
import torch.nn
import numpy as np
import random
import torch
import tqdm
import rec.evaluate as evaluate
from utils.stats import TimeStats
from .strategies import FedRecAvg

class SimpleServer:
    def __init__(self, cfg, model, client_sampler: ClientSampler):
        self.cfg = cfg
        self.client_sampler = client_sampler
        self.fedmodel = model
        self._timestats = TimeStats()

    def train_round(self, epoch_idx: int = 0):
        '''
        Flow:
        1. Sample clients & Prepare dataloader for each client
        2. Prepare parameter
        3. Train each client
        4. Aggregate updates
        5. Update server model
        '''

        # 1. Sample clients
        with self._timestats.timer("sampling_clients"):
            participants, all_data_size = self.client_sampler.next_round(self.cfg.FED.num_clients)

        # 2. Prepare parameter
        dummy_priv_params, self.server_params = self.fedmodel.on_server_prepare()
        strategy = FedRecAvg(self.server_params)
        
        # 3. Train each client
        total_loss, update_numel = 0., 0
        user_norm, item_norm = 0., 0.
        self._timestats.set_aggregation_epoch(epoch_idx)
        pbar = tqdm.tqdm(participants, desc='Training', disable=True)
        
        for client in pbar:
            with self._timestats.timer("client_time", max_agg=True):
                update, data_size, metrics = client.fit(self.fedmodel,
                                                        self.server_params, 
                                                        local_epochs=self.cfg.FED.local_epochs, 
                                                        config=self.cfg, 
                                                        device=self.cfg.TRAIN.device, 
                                                        stats_logger=self._timestats,
                                                        mask_zero_user_index=True)
                total_loss += metrics['loss'][-1] # loss at the last local training round
                update_numel += update.numel()
                user_norm += metrics['user_emb_norms']
                item_norm += metrics['item_emb_norms']
                update.post_transfer(self.cfg)

            with self._timestats.timer("server_time"):
                strategy.collect(update, weight=(data_size/all_data_size))

        with self._timestats.timer("server_time"):
            aggregated_update = strategy.finallize()
            # Simulate compression
            aggregated_update.pre_transfer(self.cfg)
            aggregated_update.post_transfer(self.cfg)
            
            strategy.step_server_optim(self.server_params, aggregated_update)
            self.fedmodel.set_state_from_splited_params([dummy_priv_params, self.server_params])

        return {"avg_participant_loss": total_loss / len(participants), 
                "update_numel": update_numel / len(participants), 
                "data_size": all_data_size,
                "item_norm": item_norm / len(participants), 
                "user_norm": user_norm / len(participants),
                }

    @torch.no_grad()
    def evaluate(self, val_loader, test_loader, train_loader=None):
        sorted_client_set = self.client_sampler.sorted_client_set
        metrics = {}
        with self._timestats.timer("evaluate"):
            eval_model = self.fedmodel.merge_client_params(sorted_client_set, self.server_params, self.fedmodel, self.cfg.TRAIN.device)
            if train_loader is not None:
                train_loss = evaluate.cal_loss(eval_model, train_loader,device=self.cfg.TRAIN.device)
                metrics['train/loss'] = train_loss
            eval_model.eval()
            if test_loader is not None:
                HR, NDCG = evaluate.metrics(eval_model, test_loader, self.cfg.EVAL.topk, device=self.cfg.TRAIN.device)
                metrics['test/HR'] = HR
                metrics['test/NDCG'] = NDCG
            if val_loader is not None:
                HR_val, NDCG_val = evaluate.metrics(eval_model, val_loader, self.cfg.EVAL.topk, device=self.cfg.TRAIN.device)
                metrics['val/HR'] = HR_val
                metrics['val/NDCG'] = NDCG_val
        return metrics