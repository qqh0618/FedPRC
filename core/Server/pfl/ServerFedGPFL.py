import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from core.Client.pfl.ClientFedGPFL import ClientFedGPFL
from core.Client.pfl.ClientFedUV import ClientFedUV
from core.Server.ServerBase import Server
from tqdm import tqdm
import numpy as np
from mem_utils import MemReporter
import time


class GCEModule(nn.Module):
    def __init__(self, in_features, num_classes, dev='cpu'):
        super(GCEModule, self).__init__()
        self.in_features = in_features
        self.num_classes = num_classes
        self.embedding = nn.Embedding(num_classes, in_features)
        self.dev = dev

    def forward(self, x, label):
        embeddings = self.embedding(torch.tensor(range(self.num_classes)).cuda())
        cosine = F.linear(F.normalize(x), F.normalize(embeddings))
        one_hot = torch.zeros(cosine.size()).to(label.device)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)

        softmax_value = F.log_softmax(cosine, dim=1)
        softmax_loss = one_hot * softmax_value
        softmax_loss = - torch.mean(torch.sum(softmax_loss, dim=1))

        return softmax_loss


class CoVModule(nn.Module):
    def __init__(self, in_dim):
        super(CoVModule, self).__init__()

        self.Conditional_gamma = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.ReLU(),
            nn.LayerNorm([in_dim]),
        )
        self.Conditional_beta = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.ReLU(),
            nn.LayerNorm([in_dim]),
        )
        self.act = nn.ReLU()

    def forward(self, x, context):
        gamma = self.Conditional_gamma(context)
        beta = self.Conditional_beta(context)

        out = torch.multiply(x, gamma + 1)
        out = torch.add(out, beta)
        out = self.act(out)
        return out


class ServerFedGPFL(Server):
    def __init__(self, args, global_model,Loader_train,Loaders_local_test,Loader_global_test,logger,device, net_idx_dataidx_map):
        super().__init__(args, global_model,Loader_train,Loaders_local_test,Loader_global_test,logger,device, net_idx_dataidx_map)
        self.global_gce = GCEModule(in_features=self.args.code_len, num_classes=self.args.num_classes).to(self.device)
        self.global_cov = CoVModule(in_dim=self.args.code_len).to(self.device)

    def Create_Clints(self):
        for idx in range(self.args.num_clients):
            self.LocalModels.append(ClientFedGPFL(self.args, copy.deepcopy(self.global_model),self.net_idx_dataidx_map[idx], idx=idx, logger=self.logger, code_length = self.args.code_len, num_classes = self.args.num_classes, device=self.device))

            
    def train(self):
        reporter = MemReporter()
        start_time = time.time()
        train_loss = []
        global_weights = self.global_model.state_dict()
        for epoch in tqdm(range(self.args.comm_round)):
            test_accuracy = 0
            local_weights, local_losses = [], []
            local_gces = []
            local_covs = []
            self.logger.info(f'Global Training Round: {epoch+1}')
            m = max(int(self.args.sampling_rate * self.args.num_clients), 1)
            idxs_users = np.random.choice(range(self.args.num_clients), m, replace=False)
            for idx in idxs_users:
                if self.args.upload_model == True:
                    self.LocalModels[idx].load_model(global_weights)
                    self.LocalModels[idx].GCE = self.global_gce
                    self.LocalModels[idx].CoV = self.global_cov

                w, loss = self.LocalModels[idx].update_weights(global_round=epoch)
                local_losses.append(copy.deepcopy(loss))
                local_weights.append(copy.deepcopy(w))
                local_gces.append(copy.deepcopy(self.LocalModels[idx].GCE.state_dict()))
                local_covs.append(copy.deepcopy(self.LocalModels[idx].CoV.state_dict()))
                acc = self.LocalModels[idx].test_accuracy()
                test_accuracy += acc

            # update global weights
            global_weights = self.average_weights(local_weights, idxs_users)
            global_gce_weights = self.average_weights(local_gces, idxs_users)
            global_cov_weights = self.average_weights(local_covs, idxs_users)

            self.global_gce.load_state_dict(global_gce_weights)
            self.global_cov.load_state_dict(global_cov_weights)

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