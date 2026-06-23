import copy

import torch
from torch import autograd
from torch.nn import functional as F
from core.Client.dg.ClientFedIIR import ClientFedIIR
from core.Server.ServerBase import Server
from tqdm import tqdm
import numpy as np
from mem_utils import MemReporter
import time

class ServerFedIIR(Server):
    def __init__(self, args, global_model,Loader_train,Loaders_local_test,Loader_global_test,logger,device, net_idx_dataidx_map):
        super().__init__(args, global_model,Loader_train,Loaders_local_test,Loader_global_test,logger,device, net_idx_dataidx_map)
        # =================
        # FedIIR
        self.ema = 0.95  # 源码lambda r: r.choice([0.90, 0.95, 0.99])
        params = list(self.global_model.classifier.parameters())
        self.grad_mean = tuple(torch.zeros_like(p).to(self.device) for p in params)
        # =================
    def Create_Clints(self):
        for idx in range(self.args.num_clients):
            self.LocalModels.append(ClientFedIIR(self.args, copy.deepcopy(self.global_model),self.net_idx_dataidx_map[idx], idx=idx, logger=self.logger, code_length = self.args.code_len, num_classes = self.args.num_classes, device=self.device))
            
            
    def train(self):
        reporter = MemReporter()
        start_time = time.time()
        train_loss = []
        global_weights = self.global_model.state_dict()
        for epoch in tqdm(range(self.args.comm_round)):
            test_accuracy = 0
            local_weights, local_losses = [], []
            self.logger.info(f'Global Training Round: {epoch+1}')
            m = max(int(self.args.sampling_rate * self.args.num_clients), 1)
            idxs_users = np.random.choice(range(self.args.num_clients), m, replace=False)

            # =====================
            # FedIIR
            self.grad_mean = self.mean_grad(idxs_users)
            # =====================

            for idx in idxs_users:
                if self.args.upload_model == True:
                    self.LocalModels[idx].load_model(global_weights)
                    # ================
                    # FedIIR
                    self.LocalModels[idx].load_gard_mean(self.grad_mean)
                    # ================
                w, loss = self.LocalModels[idx].update_weights(global_round=epoch)
                local_losses.append(copy.deepcopy(loss))
                local_weights.append(copy.deepcopy(w))
                acc = self.LocalModels[idx].test_accuracy()
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
                f'Personal_Accuracy: {test_accuracy / len(idxs_users)} || Best_Personal_Accuracy: {self.global_best_personal_acc}')
            self.logger.info(f'Global_Accuracy: {cur_g_acc} || Best_Accuracy: {self.global_best_acc}')
            self.csv_log(
                {'DataName': f"{self.args.dataset}_{self.args.leave_domain}", 'Round': epoch + 1, 'Loss': loss_avg,
                 'Personal_Acc': test_accuracy / len(idxs_users), 'Global_Acc': cur_g_acc})

        self.test_domain()
        self.logger.info('Training is completed.')
        end_time = time.time()
        self.logger.info(f'Total running time: {end_time - start_time} s || avg: {(end_time - start_time) / self.args.comm_round} s')
        reporter.report()

    def mean_grad(self, sampled_clients):

        total_batch = 0
        grad_sum = tuple(torch.zeros_like(g).to(self.device) for g in self.grad_mean)
        for idx in sampled_clients:

            for x, y in self.LocalModels[idx].trainloader:
                x, y = x.to(self.device), y.to(self.device)
                feature,logits = self.LocalModels[idx].model(x)
                loss = F.cross_entropy(logits, y)
                grad_batch = autograd.grad(loss, self.LocalModels[idx].model.classifier.parameters(), create_graph=False)

                grad_sum = tuple(g1 + g2 for g1, g2 in zip(grad_sum, grad_batch))
                total_batch += 1

        grad_mean_new = tuple(grad / total_batch for grad in grad_sum)
        return tuple(self.ema * g1 + (1 - self.ema) * g2
                     for g1, g2 in zip(self.grad_mean, grad_mean_new))
