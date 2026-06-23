import copy

import torch

from core.Server.ServerBase import Server
from core.Client.ClientFedAWA import ClientFedAWA
from tqdm import tqdm
import numpy as np
from utils import average_weights
from mem_utils import MemReporter
import time
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim

class ServerFedAWA(Server):
    def __init__(self, args, global_model,Loader_train,Loaders_local_test,Loader_global_test,logger,device, net_idx_dataidx_map):
        super().__init__(args, global_model,Loader_train,Loaders_local_test,Loader_global_test,logger,device, net_idx_dataidx_map)
        global size_weights_global
        global global_T_weights
    def Create_Clints(self):
        for idx in range(self.args.num_clients):
            self.LocalModels.append(ClientFedAWA(self.args, copy.deepcopy(self.global_model),self.net_idx_dataidx_map[idx], idx=idx, logger=self.logger, code_length = self.args.code_len, num_classes = self.args.num_classes, device=self.device))
            
            
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
                    self.LocalModels[idx].model = copy.deepcopy(self.global_model)
                w, loss = self.LocalModels[idx].update_weights(global_round=epoch)
                local_losses.append(copy.deepcopy(loss))
                local_weights.append(copy.deepcopy(w))
                acc = self.LocalModels[idx].test_accuracy()
                test_accuracy += acc

            # update global weights
            # global_weights = self.average_weights(local_weights, idxs_users)

            # ====================================
            # fedawa
            client_params = [get_param(self.LocalModels[i].model, clone=True) for i in idxs_users]
            total_data_points = sum([len(self.LocalModels[r].train_idx) for r in idxs_users])
            fed_avg_freqs = [len(self.LocalModels[r].train_idx) / total_data_points for r in idxs_users]
            if epoch==0:
                global_T_weights = torch.tensor(fed_avg_freqs, dtype=torch.float32).to('cuda')

            avg_global_param, global_T_weights = fedawa(self.args, client_params, fed_avg_freqs, self.global_model, epoch,global_T_weights)

            # ====================================
            self.global_model.load_state_dict(avg_global_param)

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

        total_data_points = sum([len(self.LocalModels[r].train_idx) for r in idxs_users])

        fed_avg_freqs = [len(self.LocalModels[r].train_idx) / total_data_points for r in idxs_users]
        fedavg_global_params = copy.deepcopy(w[0])
        # d=[]
        for name_param in w[0]:
            list_values_param = []
            for dict_local_params, num_local_data in zip(w, fed_avg_freqs):
                # print(dict_local_params[name_param])
                list_values_param.append(dict_local_params[name_param] / len(fed_avg_freqs))
            # print("list_values_param:",list_values_param)
            value_global_param = sum(list_values_param)
            fedavg_global_params[name_param] = value_global_param
        global_para = fedavg_global_params

        return global_para

def get_param(model, clone=False):
    state = model.state_dict()
    if clone:
        return {k: v.clone() for k, v in state.items()}
    return state

def _cost_matrix(x, y, dis, p=2):
    d_cosine = nn.CosineSimilarity(dim=-1, eps=1e-8)

    x_col = x.unsqueeze(-2)
    y_lin = y.unsqueeze(-3)
    if dis == 'cos':
        # print('cos_dis')
        C = 1 - d_cosine(x_col, y_lin)
    elif dis == 'euc':
        # print('euc_dis')
        C = torch.mean((torch.abs(x_col - y_lin)) ** p, -1)
    return C

def to_var(x, requires_grad=True):
    if isinstance(x, dict):
        return {k: to_var(v, requires_grad) for k, v in x.items()}
    elif torch.is_tensor(x):
        if torch.cuda.is_available():
            x = x.cuda()
        return Variable(x, requires_grad=requires_grad)
    else:
        return x
# fedgroupavg_para group mean
def fedawa(args, parameters, list_nums_local_data, central_node, rounds, global_T_weight):
    param = get_param(central_node)

    global_params = copy.deepcopy(param)

    flat_w_list = [dict_local_params['flat_w'] for dict_local_params in parameters]

    local_param_list = torch.stack(flat_w_list)

    T_weights = to_var(global_T_weight)

    # if args.server_optimizer == 'sgd':
    Attoptimizer = torch.optim.SGD([T_weights], lr=0.01, momentum=0.9, weight_decay=5e-4)
    # elif args.server_optimizer == 'adam':
    #     Attoptimizer = optim.Adam([T_weights], lr=0.001, betas=(0.5, 0.999))

    print("T_weights_before update:", torch.nn.functional.softmax(T_weights, dim=0))

    # num of server update

    for i in range(1):
        print("server weight update:", i)

        probability_train = torch.nn.functional.softmax(T_weights, dim=0)

        C = _cost_matrix(global_params['flat_w'].detach().unsqueeze(0), local_param_list.detach(), 'cos')

        reg_loss = torch.sum(probability_train * C, dim=(-2, -1))
        print("reg_loss:", reg_loss)

        client_grad = local_param_list - global_params['flat_w']

        column_sum = torch.matmul(probability_train.unsqueeze(0), client_grad)  # weighted sum

        # cosine sim
        # cos_sim = torch.nn.functional.cosine_similarity(client_grad.unsqueeze(0), column_sum.unsqueeze(1), dim=2)
        # print(cos_sim)
        #
        l2_distance = torch.norm(client_grad.unsqueeze(0) - column_sum.unsqueeze(1), p=2, dim=2)

        # cosine sim
        # print("Cos_sim:",cos_sim)
        # sim_loss=-(torch.sum(probability_train*cos_sim, dim=(-2, -1)))
        #
        print("L2_distance:", l2_distance)
        sim_loss = (torch.sum(probability_train * l2_distance, dim=(-2, -1)))

        print("Sim_loss:", sim_loss)

        Loss = sim_loss + reg_loss
        Attoptimizer.zero_grad()
        Loss.backward()
        Attoptimizer.step()
        print("step " + str(i) + " Loss:" + str(Loss))

    global_T_weight = T_weights.data

    print("T_weights_after update:", global_T_weight)

    print("probability_train_after update:", probability_train)

    fedavg_global_params = copy.deepcopy(parameters[0])
    # d=[]

    for name_param in parameters[0]:
        list_values_param = []
        for dict_local_params, num_local_data in zip(parameters, probability_train):
            # print(dict_local_params[name_param])
            list_values_param.append(dict_local_params[name_param] * num_local_data * args.gamma)
        # print("list_values_param:",list_values_param)
        value_global_param = sum(list_values_param) / sum(probability_train)

        fedavg_global_params[name_param] = value_global_param

    return fedavg_global_params, global_T_weight





