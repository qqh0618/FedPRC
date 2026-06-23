import time

import torch
import torch.nn as nn
import torch.optim as optim
from core.Client.ClientBase import Client
from typing import List, Optional, Tuple


def _get_trainable_params(model: nn.Module) -> List[nn.Parameter]:
    """Return trainable parameters in deterministic order (no side-effects)."""
    return [p for _, p in model.named_parameters() if p.requires_grad]


def _get_param_stat_indices(model: nn.Module) -> Tuple[List[int], List[int]]:
    """
    Classify trainable parameter indices into:
      - param_indices: weights/biases (everything except BN running stats)
      - stat_indices:  BatchNorm running_mean / running_var
    """
    param_idx, stat_idx = [], []
    for i, (name, p) in enumerate(model.named_parameters()):
        if not p.requires_grad:
            continue
        if 'running' in name:
            stat_idx.append(i)
        else:
            param_idx.append(i)
    return param_idx, stat_idx



class ClientFedATP(Client):
    """
    This class is for train the local model with input global model(copied) and output the updated weight
    args: argument 
    Loader_train,Loader_val,Loaders_test: input for training and inference
    user: the index of local model
    idxs: the index for data of this local model
    logger: log the loss and the process
    """
    def __init__(self, args, model, local_idx_dataidx_map, idx, logger, code_length, num_classes, device):
        super().__init__(args, model, local_idx_dataidx_map, idx, logger, code_length, num_classes, device)
        # ==============================
        # FedATP
        self.model = model.to(self.device)
        self.model.eval()

        self._params = _get_trainable_params(self.model)
        self._num_params = len(self._params)

        # Separate indices for weight/bias params vs BN running stats
        self._param_indices, self._stat_indices = _get_param_stat_indices(self.model)

        print(f'[ATP] Model has {self._num_params} trainable parameters')
        print(f'      Params (weights/biases): {len(self._param_indices)}')
        print(f'      BN running stats:        {len(self._stat_indices)}')
        # ==============================


    def update_weights(self, global_round):   # 训练模型
        self.model.to(self.device)
        self.model.train()

        self._params = _get_trainable_params(self.model)
        # Init adaptation rates to zero
        adapt_lrs = torch.zeros(self._num_params, device=self.device)

        epoch_loss = []
        optimizer = self.optim
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=self.args.lr_sh_rate, gamma=0.5)
        self.trainloader = self.get_trainloader()
        for iter in range(self.args.local_ep):
            batch_loss = []
            start_time = time.time()
            for batch_idx, (X, y) in enumerate(self.trainloader):
                X = X.to(self.device)
                y = y.to(self.device)
                # 转为半精度
                # X = X.half()
                # y = y.half()
                optimizer.zero_grad()
                _,p = self.model(X)
                loss = self.ce(p,y)               
                loss.backward()
                if self.args.clip_grad != None:
                    nn.utils.clip_grad_norm_(self.model.parameters(), max_norm = self.args.clip_grad)
                optimizer.step()
                batch_loss.append(loss.item())
            # print("一轮训练耗时:", time.time() - start_time, )
            total_loss = sum(batch_loss) / len(batch_loss)
            epoch_loss.append(total_loss)
            self.logger.info(f"Round {global_round} Client {self.idx} Epoch {iter} Loss {total_loss}")
        self.save_model(self.args.logdir + '/client_' + str(self.idx) + '.pth')
        return self.model.state_dict(), sum(epoch_loss) / len(epoch_loss)


