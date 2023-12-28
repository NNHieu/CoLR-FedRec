from collections import OrderedDict

import torch
import torch.nn as nn


class FedRecModel(nn.Module):
    def forward(self, *args, **kwargs):
        raise NotImplementedError
    
    def reinit_private_params(self):
        raise NotImplementedError
    
    def on_server_prepare(self, **kwargs):
        pass

    def set_state_from_splited_params(self, splited_params):
        private_params, submit_params = splited_params
        params_dict = dict(private_params, **submit_params)
        state_dict = OrderedDict(params_dict)
        self.load_state_dict(state_dict, strict=True)
    
    def get_splited_params(self):
        raise NotImplementedError

    def get_optimizer(self, config, opt_params = None):
        if opt_params is None:
            opt_params = self.parameters()
        if config.TRAIN.optimizer == 'sgd':
            optimizer = torch.optim.SGD(opt_params, lr=config.TRAIN.lr,) # weight_decay=config.TRAIN.weight_decay)
        elif config.TRAIN.optimizer == 'adam':
            optimizer = torch.optim.Adam(opt_params, lr=config.TRAIN.lr, weight_decay=config.TRAIN.weight_decay)
        return optimizer

    def _scale_lr_for_item_emb(self, base_lr, batch_size, optimizer):
        optimizer.param_groups[0]['lr'] = base_lr * batch_size

    def local_train(self, train_loader, optimizer, num_epochs, device, base_lr, wd, mask_zero_user_index=False):
        loss_hist = []

        self.train()
        for e in range(num_epochs):
            total_loss, count_example = 0, 0
            for user, item, label in train_loader:
                user, item, label = user.to(device), item.to(device), label.float().to(device)
                if mask_zero_user_index:
                    user *= 0
                # rescale lr for embedding layer since we set scale_grad_by_freq=True at embedding layer and set reduction='mean' at the loss function
                self._scale_lr_for_item_emb(base_lr, item.shape[0], optimizer)
                optimizer.zero_grad()
                prediction, loss = self.forward(user, item, label, gamma=wd)
                loss.backward()
                optimizer.step()

                count_example += 1
                total_loss += loss.item()
            total_loss /= count_example
            loss_hist.append(total_loss)
        return {"loss": loss_hist}