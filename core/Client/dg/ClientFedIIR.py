import copy

import torch
import torch.nn as nn
import torch.optim as optim
from torch import autograd

from core.Client.ClientBase import Client

class ClientFedIIR(Client):
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
        # ================
        # FedIIR
        self.penalty_weight = 1e-4  # 源码中： 1e-2, lambda r: r.choice([1e-2, 5e-3, 1e-3, 5e-4, 1e-4]), 原文中在1e-4最好
        self.grad_mean = None
        self.trainloader = self.get_trainloader()

        # ================
    def update_weights(self, global_round):   # 训练模型
        self.model.to(self.device)
        self.model.train()
        epoch_loss = []
        optimizer = self.optim
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=self.args.lr_sh_rate, gamma=0.5)
        part = 0.1
        self.trainloader = self.get_trainloader()
        for iter in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (X, y) in enumerate(self.trainloader):
                X = X.to(self.device)
                y = y.to(self.device)
                optimizer.zero_grad()
                _, p = self.model(X)

                # ======================
                # FedIIR
                loss_erm = self.ce(p, y)
                grad_client = autograd.grad(loss_erm, self.model.classifier.parameters(), create_graph=True)
                penalty_value = 0
                for g_client, g_mean in zip(grad_client, self.grad_mean):
                    penalty_value += (g_client - g_mean).pow(2).sum()
                loss = loss_erm + self.penalty_weight * penalty_value
                # ======================
                loss.backward()
                if self.args.clip_grad != None:
                    nn.utils.clip_grad_norm_(self.model.parameters(), max_norm = self.args.clip_grad)
                optimizer.step()
                batch_loss.append(loss.item())
            total_loss = sum(batch_loss) / len(batch_loss)
            epoch_loss.append(total_loss)
            self.logger.info(f"Round {global_round} Client {self.idx} Epoch {iter} Loss {total_loss}")
        # self.save_model(self.args.logdir + '/client_' + str(self.idx) + '.pth')
        return self.model.state_dict(),sum(epoch_loss) / len(epoch_loss)

    def load_gard_mean(self, gm):
        gard_mean = copy.deepcopy(gm)
        self.grad_mean = gard_mean