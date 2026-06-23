import copy

import torch

from core.Client.prompt.ClientFedOTP import ClientFedOTP
from core.Server.ServerBase import Server
from tqdm import tqdm
import numpy as np
from utils import average_weights, Accuracy
from mem_utils import MemReporter
import time


class ServerFedOTP(Server):
    def __init__(self, args, global_model, Loader_train, Loaders_local_test, Loader_global_test, logger, device,
                 net_idx_dataidx_map):
        super().__init__(args, global_model, Loader_train, Loaders_local_test, Loader_global_test, logger, device,
                         net_idx_dataidx_map)

        # 冻结模型参数
        for name, param in self.global_model.named_parameters():
            # print(name,":",param.size())
            if ("prompt_learner" not in name):
            # if name not in ['prompt_learner.ctx']:
                param.requires_grad_(False)
            else:
                print(name)

        self.local_weights = [[] for i in range(args.num_clients)]
        self.local_weights_0 = [[] for i in range(self.args.num_clients)]
        self.local_weights_1 = [[] for i in range(self.args.num_clients)]
        self.local_weights_per = [{} for i in range(args.num_clients)]



    def Create_Clints(self):
        for idx in range(self.args.num_clients):
            self.LocalModels.append(
                ClientFedOTP(self.args, copy.deepcopy(self.global_model), self.net_idx_dataidx_map[idx], idx=idx,
                                logger=self.logger, code_length=self.args.code_len, num_classes=self.args.num_classes,
                                device=self.device))

    def train(self):
        reporter = MemReporter()
        start_time = time.time()
        train_loss = []

        for epoch in tqdm(range(self.args.comm_round)):
            test_accuracy = 0
            local_weights, local_losses = [], []
            self.logger.info(f'Global Training Round: {epoch + 1}')
            m = max(int(self.args.sampling_rate * self.args.num_clients), 1)
            idxs_users = np.random.choice(range(self.args.num_clients), m, replace=False)
            global_weights = self.global_model.state_dict()
            for idx in idxs_users:
                if self.args.upload_model == True:
                    self.LocalModels[idx].load_model(self.local_weights_per[idx])
                w, loss = self.LocalModels[idx].update_weights(global_round=epoch)

                self.local_weights_0[idx] = copy.deepcopy(w['prompt_learner.ctx'][:self.args.avg_prompt])
                self.local_weights_1[idx] = copy.deepcopy(w['prompt_learner.ctx'][self.args.avg_prompt:self.args.num_prompt])

                local_losses.append(copy.deepcopy(loss))
                local_weights.append(copy.deepcopy(w))
                acc = self.LocalModels[idx].test_accuracy()
                test_accuracy += acc

            # update global weights
            global_prompt_weights = self.average_weights_local(self.local_weights_0, idxs_users)
            # local_prompt_weights = self.average_weights(self.local_weights_1, idxs_users)
            # self.global_model.load_state_dict(global_weights)
            # global_model的prompt_learner.ctx
            # self.global_model.prompt_learner.ctx.data = global_weights
            # global_weights['prompt_learner.ctx'] = torch.cat([global_prompt_weights, local_prompt_weights], dim=0)
            # self.global_model.load_state_dict(global_weights)

            for idx in range(self.args.num_clients):
                self.local_weights_per[idx]['prompt_learner.ctx'] = torch.cat([global_prompt_weights, self.local_weights_1[idx]], dim=0)

            global_weights = self.average_weights(copy.deepcopy(local_weights), idxs_users)
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
                f'Personal_Accuracy: {test_accuracy / len(idxs_users)} || Best_Personal_Accuracy: {self.global_best_personal_acc}')
            self.logger.info(f'Global_Accuracy: {cur_g_acc} || Best_Accuracy: {self.global_best_acc}')
            self.csv_log(
                {'DataName': f"{self.args.dataset}_{self.args.leave_domain}", 'Round': epoch + 1, 'Loss': loss_avg,
                 'Personal_Acc': test_accuracy / len(idxs_users), 'Global_Acc': cur_g_acc})

        self.test_domain()
        self.logger.info('Training is completed.')
        end_time = time.time()
        self.logger.info(
            f'Total running time: {end_time - start_time} s || avg: {(end_time - start_time) / self.args.comm_round} s')
        reporter.report()



    def average_weights_local(self, w, idxs_users):
        """
        average the weights from all local models
        """
        # train_idx
        # 获取全部数据量
        # 应该只聚合local_weight_0

        total_data_points = sum([len(self.LocalModels[r].train_idx) for r in idxs_users])

        fed_avg_freqs = [len(self.LocalModels[r].train_idx) / total_data_points for r in idxs_users]
        fedavg_global_params = copy.deepcopy(w[0])
        # d=[]
        for name_param in w[0]:
            list_values_param = []
            for dict_local_params, num_local_data in zip(w, fed_avg_freqs):
                # list_values_param.append(dict_local_params[name_param] * num_local_data)
                list_values_param.append(dict_local_params* num_local_data)
            value_global_param = sum(list_values_param) / sum(fed_avg_freqs)

            fedavg_global_params = value_global_param
        global_para = fedavg_global_params

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

        return global_para


    def global_test_accuracy(self):
        all_domain_acc = []
        for domain, dataloader in self.global_testloader.items():
            accuracy = 0
            cnt = 0
            domain_acc = []

            for batch_idx, (X, y) in enumerate(dataloader):
                X = X.to(self.device)
                y = y.to(self.device)
                # X = X.half()
                p,_,_ = self.global_model.forward_otp(X)
                y_pred = p.argmax(1)

                accuracy += Accuracy(y, y_pred)
                cnt += 1
            domain_acc.append(accuracy / cnt)
            self.logger.info(f'|| Server Domain:: {domain} Accuracy: {accuracy / cnt} ||')
            # self.logger.info(f'|| Server Domain Avg: {sum(domain_acc) / len(domain_acc)}')
            all_domain_acc.append(accuracy / cnt)

        self.logger.info(f'|| Server Domain Avg: {sum(all_domain_acc) / len(all_domain_acc)}')
        return sum(all_domain_acc) / len(all_domain_acc)