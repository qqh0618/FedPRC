import copy
import numpy as np

import time
import random
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from core.Client.ClientBase import Client
from torch.autograd import grad
import torch.nn.functional as F

class ClientFedAS(Client):
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
        # =====================
        self.yxy_hyp = 20
        self.wo_local = False
        # =====================

    
    def update_weights(self, global_round):   # 训练模型
        self.model.to(self.device)
        self.model.train()
        epoch_loss = []
        optimizer = self.optim
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=self.args.lr_sh_rate, gamma=0.5)
        self.trainloader = self.get_trainloader()

        # ---------------------------------------
        # fedas
        self.fim_trace_history = []
        # ---------------------------------------

        for iter in range(self.args.local_ep):
            batch_loss = []
            start_time = time.time()
            for batch_idx, (X, y) in enumerate(self.trainloader):
                X = X.to(self.device)
                y = y.to(self.device)
                optimizer.zero_grad()
                _, p = self.model(X)
                loss = self.ce(p,y)
                loss.backward()
                if self.args.clip_grad != None:
                    nn.utils.clip_grad_norm_(self.model.parameters(), max_norm = self.args.clip_grad)
                optimizer.step()
                batch_loss.append(loss.item())
            print("一轮训练耗时:", time.time() - start_time, )
            total_loss = sum(batch_loss) / len(batch_loss)
            epoch_loss.append(total_loss)
            self.logger.info(f"Round {global_round} Client {self.idx} Epoch {iter} Loss {total_loss}")
        self.save_model(self.args.logdir + '/client_' + str(self.idx) + '.pth')

        # =======================================================================
        # FedAS
        self.model.eval()
        fim_trace_sum = 0
        for i, (x, y) in enumerate(self.trainloader):
            # Forward pass
            x = x.to(self.device)
            y = y.to(self.device)
            x.requires_grad = False
            y.requires_grad = False
            y = y.long()
            outputs, _ = self.model(x)
            # Negative log likelihood as our loss
            nll = - F.log_softmax(outputs, dim=1)[range(len(y)), y].mean()
            # Compute gradient of the negative log likelihood w.r.t. model parameters

            grads = grad(nll, self.model.parameters(), allow_unused=True)

            # Compute and accumulate the trace of the Fisher Information Matrix
            for g in grads:
                try:
                    fim_trace_sum += torch.sum(g ** 2).detach()
                    # print(False)
                except:
                    # print(True)
                    fim_trace_sum += 0

        # add the fisher log
        self.fim_trace_history.append(fim_trace_sum.item())
        # =======================================================================

        return self.model.state_dict(), sum(epoch_loss) / len(epoch_loss)

    def get_fim_trace_history(self):
        return self.fim_trace_history
