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


def set_params(net, global_model, finetune=False):
    """
        Reference: Be careful with the state_dict[key].
        https://discuss.pytorch.org/t/how-to-copy-a-modified-state-dict-into-a-models-state-dict/64828/4.
    """
    with torch.no_grad():
        for key in global_model.state_dict().keys():
            if key not in ['prototype']:
                net.state_dict()[key].copy_(global_model.state_dict()[key])

        if finetune:  # 微调是全量梯度更新
            for name, p in net.named_parameters():
                try:
                    if name in ['prototype']:
                        p.requires_grad = True
                    else:
                        p.requires_grad = True
                except:
                    pass
        else:
            for name, p in net.named_parameters():
                try:
                    if name in ['prototype']:
                        p.requires_grad = False
                    else:
                        p.requires_grad = True
                except:
                    pass



class ClientFedBABU(Client):
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
        self.finetune = False
    def update_weights(self, global_round):   # 训练模型
        self.model.to(self.device)
        self.model.train()
        epoch_loss = []
        optimizer = self.optim
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=self.args.lr_sh_rate, gamma=0.5)
        self.trainloader = self.get_trainloader()

        set_params(self.model, copy.deepcopy(self.model), self.finetune)

        for iter in range(self.args.local_ep):
            batch_loss = []
            start_time = time.time()
            for batch_idx, (X, y) in enumerate(self.trainloader):
                X = X.to(self.device)
                y = y.to(self.device)
                optimizer.zero_grad()
                feats, _  = self.model(X)
                # ====================================
                # FedBABU
                feats_norm = torch.norm(feats, p=2, dim=1, keepdim=True).clamp(min=1e-12)  # 512*10 5120
                feats_embedding = torch.div(feats, feats_norm)
                norm_prototype = self.model.prototype
                logits = torch.matmul(feats_embedding, norm_prototype.T)
                p = self.model.scaling * logits
                # ====================================
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
        return self.model.state_dict(),sum(epoch_loss) / len(epoch_loss)
