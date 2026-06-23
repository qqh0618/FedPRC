"""
2025/4/30 18:57
while True:
    leanring
本文件由my_ywj首次创建编写
"""
import copy

import torch

from core.Client.dg.ClientFedGA import ClientFedGA
from core.Server.ServerBase import Server
from tqdm import tqdm
import numpy as np
from mem_utils import MemReporter
import time

import torch.nn.functional as F


class Classification(object):
    def __init__(self):
        self.init()

    def init(self):
        self.pred_list = []
        self.label_list = []
        self.correct_count = 0
        self.total_count = 0
        self.loss = 0

    def update(self, pred, label, easy_model=False):
        pred = pred.cpu()
        label = label.cpu()

        if easy_model:
            pass
        else:
            loss = F.cross_entropy(pred, label).item() * len(label)
            self.loss += loss
            pred = pred.data.max(1)[1]
        self.pred_list.extend(pred.numpy())
        self.label_list.extend(label.numpy())
        self.correct_count += pred.eq(label.data.view_as(pred)).sum()
        self.total_count += len(label)

    def results(self):
        result_dict = {}
        result_dict['acc'] = float(self.correct_count) / float(self.total_count)
        result_dict['loss'] = float(self.loss) / float(self.total_count)
        self.init()
        return result_dict


class ServerFedGA(Server):
    def __init__(self, args, global_model, Loader_train, Loaders_local_test, Loader_global_test, logger, device, net_idx_dataidx_map):
        super().__init__(args, global_model, Loader_train, Loaders_local_test, Loader_global_test, logger, device, net_idx_dataidx_map)
        # =======================
        # FedDG-GA
        self.weight_dict = {}
        self.site_results_before_avg = {}
        self.site_results_after_avg = {}
        self.metric = Classification()
        for site_name in range(self.args.num_clients):
            self.weight_dict[site_name] = 1. / 3.
            self.site_results_before_avg[site_name] = None
            self.site_results_after_avg[site_name] = None

        # =======================

    def Create_Clints(self):
        for idx in range(self.args.num_clients):
            self.LocalModels.append(ClientFedGA(self.args, copy.deepcopy(self.global_model), self.net_idx_dataidx_map[idx], idx=idx, logger=self.logger,
                                                 code_length=self.args.code_len, num_classes=self.args.num_classes,
                                                 device=self.device))

    def train(self):
        reporter = MemReporter()
        start_time = time.time()
        train_loss = []
        global_weights = self.global_model.state_dict()

        # ======================
        # FedDG-GA
        for idx in range(self.args.num_clients):
            if self.args.upload_model == True:
                self.LocalModels[idx].load_model(global_weights)

                self.LocalModels[idx].load_metrics(self.metric)
        # ======================

        for epoch in tqdm(range(self.args.comm_round)):
            test_accuracy = 0
            local_numbers, local_weights, local_losses =[], [], []
            self.logger.info(f'Global Training Round: {epoch + 1}')
            m = max(int(self.args.sampling_rate * self.args.num_clients), 1)
            idxs_users = np.random.choice(range(self.args.num_clients), m, replace=False)
            for idx in idxs_users:
                w, loss = self.LocalModels[idx].update_weights(global_round=epoch)
                # ======================
                # FedDG-GA
                self.site_results_before_avg[idx] = self.LocalModels[idx].site_evaluation()
                # ======================

                local_losses.append(copy.deepcopy(loss))
                local_weights.append(copy.deepcopy(w))
                acc = self.LocalModels[idx].test_accuracy()
                test_accuracy += acc
            # ======================
            # FedDG-GA
            global_weights = self.average_weights(local_weights, idxs_users)
            self.global_model.load_state_dict(global_weights)

            for idx in idxs_users:
                self.LocalModels[idx].load_model(global_weights)
                self.LocalModels[idx].load_metrics(self.metric)

            for idx in idxs_users:
                self.site_results_after_avg[idx] = self.LocalModels[idx].site_evaluation()

            self.site_evaluation()
            self.weight_dict = refine_weight_dict_by_GA(self.weight_dict, self.site_results_before_avg, self.site_results_after_avg,
                                                   0.2 - (epoch - 1) * 0.2)
            # ======================

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
            self.csv_log(
                {'DataName': f"{self.args.dataset}_{self.args.leave_domain}", 'Round': epoch + 1, 'Loss': loss_avg,
                 'Personal_Acc': test_accuracy / len(idxs_users), 'Global_Acc': cur_g_acc})

        self.test_domain()
        self.logger.info('Training is completed.')
        end_time = time.time()
        self.logger.info(
            f'Total running time: {end_time - start_time} s || avg: {(end_time - start_time) / self.args.comm_round} s')
        reporter.report()

    def site_evaluation(self):
        # 客户端测试
        self.global_model.eval()
        with torch.no_grad():
            # for domain, dataloader in self.global_testloader.items():
            for imgs, labels in self.global_testloader:
                imgs = imgs.cuda()
                _, output = self.global_model(imgs)
                self.metric.update(output, labels)
        results_dict = self.metric.results()

        return results_dict



def refine_weight_dict_by_GA(weight_dict, site_before_results_dict, site_after_results_dict, step_size=0.1,
                             fair_metric='loss'):
    # if fair_metric == 'acc':
    #     signal = -1.0
    # elif fair_metric == 'loss':
    #     signal = 1.0
    # else:
    #     raise ValueError('fair_metric must be acc or loss')
    signal = -1.0
    value_list = []
    for site_name in site_before_results_dict.keys():
        value_list.append(
            site_after_results_dict[site_name][fair_metric] - site_before_results_dict[site_name][fair_metric])

    value_list = np.array(value_list)

    step_size = 1. / 3. * step_size
    norm_gap_list = value_list / np.max(np.abs(value_list))

    for i, site_name in enumerate(weight_dict.keys()):
        weight_dict[site_name] += signal * norm_gap_list[i] * step_size

    weight_dict = weight_clip(weight_dict)

    return weight_dict


def weight_clip(weight_dict):
    new_total_weight = 0.0
    for key_name in weight_dict.keys():
        weight_dict[key_name] = np.clip(weight_dict[key_name], 0.0, 1.0)
        new_total_weight += weight_dict[key_name]

    for key_name in weight_dict.keys():
        weight_dict[key_name] /= new_total_weight

    return weight_dict

