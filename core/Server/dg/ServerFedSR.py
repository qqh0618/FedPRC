import copy

import torch
from torch import nn

from core.Client.dg.ClientFedSR import ClientFedSR
from core.Server.ServerBase import Server
from core.Client.ClientFedAvg import ClientFedAvg
from tqdm import tqdm
import numpy as np
from utils import average_weights
from mem_utils import MemReporter
import time
import torch.nn.functional as F
import torch.distributions as distributions

class ServerFedSR(Server):
    def __init__(self, args, global_model, Loader_train, Loaders_local_test, Loader_global_test, logger, device, net_idx_dataidx_map):
        super().__init__(args, global_model, Loader_train, Loaders_local_test, Loader_global_test, logger, device, net_idx_dataidx_map)
        self.probabilistic = True
        self.z_dim = self.args.code_len // 2  # 1/2的特征层长度
        self.global_model.add_module('cls', nn.Linear(self.z_dim, self.num_classes).to(self.device))
        # self.r_mu = nn.Parameter(torch.zeros(self.num_classes, self.z_dim)).to(self.device)
        # self.r_sigma = nn.Parameter(torch.ones(self.num_classes, self.z_dim)).to(self.device)
        # self.C = nn.Parameter(torch.ones([])).to(self.device)
        self.global_model.r_mu = nn.Parameter(torch.zeros(self.num_classes, self.z_dim))
        self.global_model.r_sigma = nn.Parameter(torch.ones(self.num_classes, self.z_dim))
        self.global_model.C = nn.Parameter(torch.ones([]))

    def Create_Clints(self):
        for idx in range(self.args.num_clients):
            self.LocalModels.append(ClientFedSR(self.args, copy.deepcopy(self.global_model),self.net_idx_dataidx_map[idx], idx=idx, logger=self.logger, code_length = self.args.code_len, num_classes = self.args.num_classes, device=self.device))
            
            
    def train(self):
        reporter = MemReporter()
        start_time = time.time()
        train_loss = []

        for epoch in tqdm(range(self.args.comm_round)):
            test_accuracy = 0
            local_weights, local_losses = [], []
            self.logger.info(f'Global Training Round: {epoch+1}')
            m = max(int(self.args.sampling_rate * self.args.num_clients), 1)
            idxs_users = np.random.choice(range(self.args.num_clients), m, replace=False)
            global_weights = self.global_model.state_dict()
            for idx in idxs_users:
                if self.args.upload_model == True:
                    self.LocalModels[idx].load_model(global_weights)
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

    def featurize(self, x, num_samples=1, return_dist=False):
        # if self.args.dataset in[ "pacs", "office_home","caltech10" ]:
        #     num_samples = 3*self.args.meta_num  # 域数量*每个域客户端数量
        # elif self.args.dataset =="office31":
        #     num_samples = 2*self.args.meta_num  # 域数量*每个域客户端数量
        z_params, _ = self.global_model(x)
        z_mu = z_params[:, :self.z_dim]
        z_sigma = F.softplus(z_params[:, self.z_dim:])
        z_dist = distributions.Independent(distributions.normal.Normal(z_mu, z_sigma), 1)
        z = z_dist.rsample([num_samples]).view([-1, self.z_dim])

        if return_dist:
            return z, (z_mu, z_sigma)
        else:
            return z

    def global_test_accuracy(self):
        self.global_model.eval()
        with torch.no_grad():
            total, correct = 0, 0
            for x, y in self.global_testloader:
                x, y = x.to(self.device), y.to(self.device)
                z = self.featurize(x)
                y_pred = self.global_model.cls(z)
                # preds = torch.softmax(self.cls(z), dim=1)
                # preds = preds.view([self.num_samples, -1, self.num_classes]).mean(0)
                y_pred = torch.softmax(y_pred, dim=1)
                y_pred = y_pred.view([self.args.meta_num, -1, self.num_classes]).mean(0)
                y_pred = torch.log(y_pred)
                _, pred = torch.max(y_pred.data, 1)
                total += y.size(0)
                correct += (pred == y).sum().item()
        return correct / total
