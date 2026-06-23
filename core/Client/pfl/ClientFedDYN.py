import copy

import torch.nn as nn
import torch.optim as optim
import torch
import numpy as np
import torch.nn.functional as F

from core.Client.ClientBase import Client



class ClientFedDYN(Client):
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
        # =========================
        #     feddyn
        self.c_previous_gradient = None
        # =========================


    def update_weights(self, global_round):   # 训练模型
        self.model.to(self.device)
        self.model.train()
        epoch_loss = []
        optimizer = self.optim
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=self.args.lr_sh_rate, gamma=0.5)
        self.trainloader = self.get_trainloader()

        # =========================
        #     feddyn
        ALPHA = 1e-3
        self.par_flat = torch.cat([p.reshape(-1) for p in copy.deepcopy(self.model).parameters()])
        # =========================

        for iter in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (X, y) in enumerate(self.trainloader):
                X = X.to(self.device)
                y = y.to(self.device)
                optimizer.zero_grad()
                f, p = self.model(X)

                loss = self.ce(p, y)

                # =========================
                #     feddyn
                cur_flat = torch.cat([p.reshape(-1) for p in self.model.parameters()])

                # Compute the linear penalty: prev_grad_flat · cur_flat
                linear_penalty = torch.sum(self.c_previous_gradient.data * cur_flat)
                # Compute the quadratic penalty: (alpha / 2) * || cur_flat - par_flat || ^ 2
                norm_penalty = (ALPHA / 2) * torch.linalg.norm(cur_flat - self.par_flat, 2) ** 2
                # =========================

                loss = loss - linear_penalty + norm_penalty
                loss.backward()
                if self.args.clip_grad != None:
                    nn.utils.clip_grad_norm_(self.model.parameters(), max_norm = self.args.clip_grad)
                optimizer.step()
                batch_loss.append(loss.item())
            total_loss = sum(batch_loss) / len(batch_loss)
            epoch_loss.append(total_loss)
            self.logger.info(f"Round {global_round} Client {self.idx} Epoch {iter} Loss {total_loss}")

        self.cur_flat = torch.cat([p.detach().reshape(-1) for p in self.model.parameters()]).to(self.device)

        self.save_model(self.args.logdir + '/client_' + str(self.idx) + '.pth')
        return self.model.state_dict(),sum(epoch_loss) / len(epoch_loss)
