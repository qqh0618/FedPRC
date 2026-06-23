import copy
import itertools
import time

import torch
import torch.nn as nn
import torch.optim as optim
from core.Client.ClientBase import Client
from utils import Accuracy


class ClientFedProxC(Client):
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

    def update_weights_Prox(self,global_round, lam):  # 训练模型
        self.model.to(self.device)
        self.model.train()
        epoch_loss = []

        # self.print_model()
        global_model = copy.deepcopy(self.model)
        global_model.eval()
        global_weight_collector = list(global_model.parameters())

        optimizer = optim.SGD(itertools.chain(
        self.model.prompt_learner.parameters()
    ), lr=self.args.lr, momentum=0.9, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=self.args.lr_sh_rate, gamma=0.5)
        self.trainloader = self.get_trainloader()
        # self.print_model()
        for iter in range(self.args.local_ep):
            batch_loss = []
            start_time = time.time()
            for batch_idx, (X, y) in enumerate(self.trainloader):
                X = X.to(self.device)
                y = y.to(self.device)
                # 展示图片

                # 转为半精度
                # X = X.half()
                # y = y.half()

                output, _, _ = self.model(X)

                loss = self.ce(output, y)

                fed_prox_reg = 0.0
                for param_index, param in enumerate(self.model.parameters()):
                    fed_prox_reg += ((lam / 2) * torch.norm((param - global_weight_collector[param_index])) ** 2)

                loss.backward()
                if self.args.clip_grad != None:
                    nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.args.clip_grad)
                optimizer.zero_grad()
                optimizer.step()
                batch_loss.append(loss.item())
            # print("一轮训练耗时:", time.time() - start_time, )
            total_loss = sum(batch_loss) / len(batch_loss)
            epoch_loss.append(total_loss)
            self.logger.info(f"Round {global_round} Client {self.idx} Epoch {iter} Loss {total_loss}")
        self.save_model(self.args.logdir + '/client_' + str(self.idx) + '.pth')
        return self.model.state_dict(), sum(epoch_loss) / len(epoch_loss)

    def print_model(self):
        for name, param in self.model.named_parameters():
            print(name, param.requires_grad)
