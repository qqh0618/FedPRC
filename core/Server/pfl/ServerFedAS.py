import copy
from core.Client.pfl.ClientFedAS import ClientFedAS
from core.Server.ServerBase import Server
from tqdm import tqdm
import numpy as np
from mem_utils import MemReporter
import time

class ServerFedAS(Server):
    def __init__(self, args, global_model,Loader_train,Loaders_local_test,Loader_global_test,logger,device, net_idx_dataidx_map):
        super().__init__(args, global_model,Loader_train,Loaders_local_test,Loader_global_test,logger,device, net_idx_dataidx_map)

        # ===========================
        # FedAS
        self.fim_trace_historys = {}
        # ===========================


    def Create_Clints(self):
        for idx in range(self.args.num_clients):
            self.LocalModels.append(ClientFedAS(self.args, copy.deepcopy(self.global_model),self.net_idx_dataidx_map[idx], idx=idx, logger=self.logger, code_length = self.args.code_len, num_classes = self.args.num_classes, device=self.device))
            
            
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

            self.fim_trace_historys = {}

            for idx in idxs_users:
                if self.args.upload_model == True:
                    self.LocalModels[idx].load_model(global_weights)
                w, loss = self.LocalModels[idx].update_weights(global_round=epoch)
                local_losses.append(copy.deepcopy(loss))
                local_weights.append(copy.deepcopy(w))
                # ===========================
                # FedAS
                self.fim_trace_historys[idx] = self.LocalModels[idx].get_fim_trace_history()
                # ===========================
                acc = self.LocalModels[idx].test_accuracy()
                test_accuracy += acc


            # ===========================
            # FedAS
            self.FIM_weight_list = []
            for id, fth in self.fim_trace_historys.items():
                self.FIM_weight_list.append(fth[-1])
            # ===========================

            # normalization to obtain weight
            # FIM_weight_list = [FIM_value / sum(FIM_weight_list) for FIM_value in FIM_weight_list]
            #
            # for idx in range(len(idxs_users)):  # 可以包含GCE聚合，GCE按照客户端数据量进行加权平均
            #     net_para = local_weights[idx].cpu().state_dict()
            #     if idx == 0:
            #         for key in net_para:
            #             global_para[key] = net_para[key] * FIM_weight_list[idx]
            #     else:
            #         for key in net_para:
            #             global_para[key] += net_para[key] * FIM_weight_list[idx]

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
        self.test_domain()
        self.logger.info('Training is completed.')
        end_time = time.time()
        self.logger.info(f'Total running time: {end_time - start_time} s || avg: {(end_time - start_time) / self.args.comm_round} s')
        reporter.report()

    def average_weights(self, w, idxs_users):
        global_weights = self.global_model.state_dict()
        FIM_weight_list = [FIM_value / sum(self.FIM_weight_list) for FIM_value in self.FIM_weight_list]
        global_para = copy.deepcopy(w[0])
        for idx in range(len(idxs_users)):  # 可以包含GCE聚合，GCE按照客户端数据量进行加权平均
            net_para = w[idx]
            if idx == 0:
                for key in net_para:
                    global_para[key] = net_para[key] * FIM_weight_list[idx]
            else:
                for key in net_para:
                    global_para[key] += net_para[key] * FIM_weight_list[idx]
        return global_para
