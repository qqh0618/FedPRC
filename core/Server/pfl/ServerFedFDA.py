import copy

import torch

from core.Client.pfl.ClientFedFDA import ClientFedFDA
from core.Server.ServerBase import Server
from tqdm import tqdm
import numpy as np
from mem_utils import MemReporter
import time

class ServerFedFDA(Server):
    def __init__(self, args, global_model,Loader_train,Loaders_local_test,Loader_global_test,logger,device, net_idx_dataidx_map):
        super().__init__(args, global_model,Loader_train,Loaders_local_test,Loader_global_test,logger,device, net_idx_dataidx_map)
        # ============================
        # FedFDA
        self.D = 512
        self.global_means = torch.Tensor(torch.rand([self.num_classes, self.D]))
        self.global_covariance = torch.Tensor(torch.eye(self.D))
        self.global_priors = torch.ones(self.num_classes) / self.num_classes
        self.r = 0
        # ============================

    def Create_Clints(self):
        for idx in range(self.args.num_clients):
            self.LocalModels.append(ClientFedFDA(self.args, copy.deepcopy(self.global_model),self.net_idx_dataidx_map[idx], idx=idx, logger=self.logger, code_length = self.args.code_len, num_classes = self.args.num_classes, device=self.device))
            
            
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
            self.seclected_clients = idxs_users
            for idx in idxs_users:
                if self.args.upload_model == True:
                    self.LocalModels[idx].load_model(global_weights)
                    # =======================
                    # FedFDA
                    self.LocalModels[idx].global_means.data = self.global_means.data
                    self.LocalModels[idx].global_covariance.data = self.global_covariance.data
                    if epoch == 1:
                        self.LocalModels[idx].means.data = self.global_means.data
                        self.LocalModels[idx].covariance.data = self.global_covariance.data
                        self.LocalModels[idx].adaptive_means.data = self.global_means.data
                        self.LocalModels[idx].adaptive_covariance.data = self.global_covariance.data
                    # =======================
                w, loss = self.LocalModels[idx].update_weights(global_round=epoch)
                local_losses.append(copy.deepcopy(loss))
                local_weights.append(copy.deepcopy(w))
                acc = self.LocalModels[idx].test_accuracy()
                test_accuracy += acc

            # update global weights
            global_weights = self.aggregate_models(local_weights, idxs_users)
            self.global_model.load_state_dict(global_weights)

            # print loss
            loss_avg = sum(local_losses) / len(local_losses)
            train_loss.append(loss_avg)
            cur_g_acc = self.global_test_accuracy()
            if cur_g_acc > self.global_best_acc:
                self.global_best_acc = cur_g_acc
                self.Save_CheckPoint(self.args.logdir + '/best_model.pth')
            if test_accuracy / len(idxs_users) > self.global_best_personal_acc:
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

    def aggregate_models(self, w, idxs_users):
        model_dict = self.average_weights(w, idxs_users)
        total_samples = sum(self.LocalModels[c].trainloader.dataset.__len__() for c in self.seclected_clients)/self.args.part
        self.global_means.data = torch.zeros_like(self.LocalModels[0].means)
        self.global_covariance.data = torch.zeros_like(self.LocalModels[0].covariance)

        for c in self.seclected_clients:
            self.global_means.data = self.global_means.data + (self.LocalModels[c].num_train / total_samples) * self.LocalModels[c].adaptive_means.data
            self.global_covariance.data = self.global_covariance.data + (
                        self.LocalModels[c].num_train / total_samples) * self.LocalModels[c].adaptive_covariance.data
        return model_dict