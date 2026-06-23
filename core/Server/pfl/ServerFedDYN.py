import copy

import torch

from core.Client.pfl.ClientFedDYN import ClientFedDYN
from core.Client.pfl.ClientFedUV import ClientFedUV
from core.Server.ServerBase import Server
from tqdm import tqdm
import numpy as np
from mem_utils import MemReporter
import time

def init_prev_grads(model):
    prev_grads = None
    for param in model.parameters():
        if not isinstance(prev_grads, torch.Tensor):
            prev_grads = torch.zeros_like(param.view(-1))
        else:
            prev_grads = torch.cat((prev_grads, torch.zeros_like(param.view(-1))), dim=0)
    return prev_grads

class ServerFedDYN(Server):
    def __init__(self, args, global_model,Loader_train,Loaders_local_test,Loader_global_test,logger,device, net_idx_dataidx_map):
        super().__init__(args, global_model,Loader_train,Loaders_local_test,Loader_global_test,logger,device, net_idx_dataidx_map)
        # ===============================
        #     feddyn
        self.all_previous_gradient = []
        self.ALPHA = 1e-3
        # ===============================


    def Create_Clints(self):
        for idx in range(self.args.num_clients):
            self.LocalModels.append(ClientFedDYN(self.args, copy.deepcopy(self.global_model),self.net_idx_dataidx_map[idx], idx=idx, logger=self.logger, code_length = self.args.code_len, num_classes = self.args.num_classes, device=self.device))
            
            
    def train(self):
        reporter = MemReporter()
        start_time = time.time()
        train_loss = []
        global_weights = self.global_model.state_dict()

        # ===============================
        #     feddyn
        for i in range(self.args.num_clients):
            self.all_previous_gradient.append(init_prev_grads(self.LocalModels[i].model))
        # ===============================

        for epoch in tqdm(range(self.args.comm_round)):
            test_accuracy = 0
            local_weights, local_losses = [], []
            self.logger.info(f'Global Training Round: {epoch+1}')
            m = max(int(self.args.sampling_rate * self.args.num_clients), 1)
            idxs_users = np.random.choice(range(self.args.num_clients), m, replace=False)

            for idx in idxs_users:
                if self.args.upload_model == True:
                    self.LocalModels[idx].load_model(global_weights)
                self.LocalModels[idx].c_previous_gradient = self.all_previous_gradient[idx]
                w, loss = self.LocalModels[idx].update_weights(global_round=epoch)
                local_losses.append(copy.deepcopy(loss))
                local_weights.append(copy.deepcopy(w))
                acc = self.LocalModels[idx].test_accuracy()
                self.all_previous_gradient[idx] -= self.ALPHA * (self.LocalModels[idx].cur_flat - self.LocalModels[idx].par_flat)
                test_accuracy += acc

            # update global weights
            global_weights = self.average_weights(local_weights, idxs_users)
            self.global_model.load_state_dict(global_weights)

            # print loss
            loss_avg = sum(local_losses) / len(local_losses)
            train_loss.append(loss_avg)
            cur_g_acc = self.global_test_accuracy()
            if cur_g_acc > self.global_best_acc:
                self.global_best_acc = cur_g_acc
                self.Save_CheckPoint(self.args.logdir + '/best_model.pth')
            if test_accuracy / len(idxs_users)  > self.global_best_personal_acc:
                self.global_best_personal_acc = test_accuracy / len(idxs_users)
            self.logger.info(f'Global Training Loss: {loss_avg}')
            self.logger.info(
                f'Personal_Accuracy: {test_accuracy / len(idxs_users) } || Best_Personal_Accuracy: {self.global_best_personal_acc}')
            self.logger.info(f'Global_Accuracy: {cur_g_acc} || Best_Accuracy: {self.global_best_acc}')
        self.test_domain()
        self.logger.info('Training is completed.')
        end_time = time.time()
        self.logger.info(f'Total running time: {end_time - start_time} s || avg: {(end_time - start_time) / self.args.comm_round} s')
        reporter.report()