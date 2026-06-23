import copy

import torch

from core.Client.fl.ClientFedSCE import ClientFedSCE
from core.Server.ServerBase import Server
from tqdm import tqdm
import numpy as np
from utils import average_weights
from mem_utils import MemReporter
import time
import torch.nn as nn

class ServerFedSCE(Server):
    def __init__(self, args, global_model,Loader_train,Loaders_local_test,Loader_global_test,logger,device, net_idx_dataidx_map):
        super().__init__(args, global_model,Loader_train,Loaders_local_test,Loader_global_test,logger,device, net_idx_dataidx_map)

    def Create_Clints(self):
        for idx in range(self.args.num_clients):
            self.LocalModels.append(ClientFedSCE(self.args, copy.deepcopy(self.global_model),self.net_idx_dataidx_map[idx], idx=idx, logger=self.logger, code_length = self.args.code_len, num_classes = self.args.num_classes, device=self.device))
            
            
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
            # global_weights = self.global_model.state_dict()
            for idx in idxs_users:
                if self.args.upload_model == True:
                    self.LocalModels[idx].global_model = self.global_model
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

    def average_weights(self, w, idxs_users):
        """
        average the weights from all local models
        """
        # train_idx
        # 获取全部数据量

        F1_sum = sum([self.LocalModels[r].F1 for r in idxs_users])
        F2_sum = sum([self.LocalModels[r].F2 for r in idxs_users])

        W_list = [self.LocalModels[r].F1 / F1_sum + self.LocalModels[r].F2 / F2_sum for r in idxs_users]
        W_sum = sum(W_list)

        # # ==================================
        # global_para= copy.deepcopy(w[0])
        # # 加权平均
        # for idx in range(len(idxs_users)):  # 可以包含GCE聚合，GCE按照客户端数据量进行加权平均
        #     net_para = w[idx]
        #     if idx == 0:
        #         for key in net_para:
        #             global_para[key] = net_para[key] * fed_avg_freqs[idx]
        #     else:
        #         for key in net_para:
        #             global_para[key] += net_para[key] * fed_avg_freqs[idx]

        #  ==================================
        fedavg_global_params = copy.deepcopy(w[0])
        # d=[]
        for name_param in w[0]:
            list_values_param = []
            for dict_local_params, num_local_data in zip(w, W_list):
                # print(dict_local_params[name_param])
                list_values_param.append(dict_local_params[name_param] * num_local_data)
            # print("list_values_param:",list_values_param)
            value_global_param = sum(list_values_param) / sum(W_list)
            # print("value_global_param:",value_global_param)

            # print("name_param:"+name_param+':',fedavg_global_params[name_param]-value_global_param)

            # print("name_param:"+name_param+':',torch.mean(torch.abs(fedavg_global_params[name_param]-value_global_param)))
            # if name_param[-6:]=="weight":
            # a=1-torch.mean(torch.abs(fedavg_global_params[name_param]-value_global_param))
            # d.append(a.item())
            # d=0.999
            fedavg_global_params[name_param] = value_global_param
        global_para = fedavg_global_params
        return global_para

