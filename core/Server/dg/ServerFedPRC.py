import copy

import torch


from core.Client.dg.ClientPRC import ClientFedPRC

from core.Server.ServerBase import Server
from core.Client.ClientFedAvg import ClientFedAvg
from tqdm import tqdm
import numpy as np
from utils import average_weights, Accuracy
from mem_utils import MemReporter
import time
import torch.nn as nn
import torch.nn.functional as F

from numpy.linalg import norm, pinv
from scipy.special import logsumexp, softmax
from sklearn.covariance import EmpiricalCovariance
# 利用clip的tokenizer创建一个语义标签分类器模型
class TextClassifier(nn.Module):
    def __init__(self, args, code_length, num_classes):
        super().__init__()
        self.args = args
        self.code_length = code_length
        self.num_classes = num_classes
        self.linear = nn.Linear(code_length, num_classes)

    def forward(self, x):
        return self.linear(x)

def compute_mahalanobis_bias(client_feat, global_mean, global_cov_inv):
    """
    client_feat: np.array of shape (N, D)
    global_mean: np.array of shape (D,)
    global_cov_inv: np.array of shape (D, D)
    """
    delta = client_feat - global_mean  # (N, D)
    distances = np.einsum('nd,df,nf->n', delta, global_cov_inv, delta)  # 快速计算 N 个样本的距离
    return distances.mean()  # 可用于客户端指标或正则化


def vim_get_feature_space(features, global_net):
    # 当前客户端数据在全局模型的特征空间
    method = 'ViM'
    # print(f'\n{method}')

    feature_id_train = np.concatenate([features.cpu().detach().numpy()], axis=0)
    # feature_id_train = np.squeeze(features)
    # print(f'{feature_id_train.shape=}')

    w = global_net.proto_classifier.proto.numpy()
    b = np.zeros(w.shape)
    # print(f'{w.shape=}, {b.shape=}')

    # 计算logit
    logit_id_train = feature_id_train @ w + b
    # print('computing softmax...')

    # 计算softmax
    # softmax_id_train = softmax(logit_id_train, axis=-1)

    # 很神奇的一步，类似于奇异值分解
    # u = -np.matmul(pinv(w), b)  # 这个U很重要

    result = []
    if feature_id_train.shape[-1] >= 2048:
        DIM = 1000
    elif feature_id_train.shape[-1] >= 768:
        DIM = 74
    else:
        DIM = 20
        # DIM = 128
        # DIM = 256
    # print(f'{DIM=}')

    # 计算主要空间
    # print('computing principal space...')
    ec = EmpiricalCovariance(assume_centered=True)
    ec.fit(feature_id_train)
    eig_vals, eigen_vectors = np.linalg.eig(ec.covariance_)
    NS = np.ascontiguousarray(
        (eigen_vectors.T[np.argsort(eig_vals * -1)[:DIM]]).T)

    # print('computing alpha...')
    vlogit_id_train = norm(np.matmul(feature_id_train, NS), axis=-1)
    alpha = logit_id_train.max(axis=-1).mean() / vlogit_id_train.mean()
    # print(f'{alpha=:.4f}')

    energy_id_train = logsumexp(vlogit_id_train, axis=-1)
    score_id = -vlogit_id_train + energy_id_train  # 偏离程度

    return alpha
    # --------------------------------------------------------


class ServerFedVDDG(Server):
    def __init__(self, args, global_model, Loader_train, Loaders_local_test, Loader_global_test, logger, device,
                 net_idx_dataidx_map):
        super().__init__(args, global_model, Loader_train, Loaders_local_test, Loader_global_test, logger, device,
                         net_idx_dataidx_map)
        # ===============
        # WJDG
        self.global_U = None
        self.global_mean = None
        self.global_std = None
        # ===============
        self.global_model.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def Create_Clints(self):
        for idx in range(self.args.num_clients):
            self.LocalModels.append(
                ClientFedPRC(self.args, copy.deepcopy(self.global_model), self.net_idx_dataidx_map[idx], idx=idx,
                              logger=self.logger, code_length=self.args.code_len, num_classes=self.args.num_classes,
                              device=self.device))

    def train(self):
        reporter = MemReporter()
        start_time = time.time()
        train_loss = []
        global_weights = self.global_model.state_dict()
        for epoch in tqdm(range(self.args.comm_round)):
            test_accuracy = 0
            local_weights, local_losses = [], []
            self.logger.info(f'Global Training Round: {epoch + 1}')
            m = max(int(self.args.sampling_rate * self.args.num_clients), 1)
            idxs_users = np.random.choice(range(self.args.num_clients), m, replace=False)

            # ==========================
            # WJDA
            local_U = []
            # ==========================

            for idx in idxs_users:
                if self.args.upload_model == True:
                    self.LocalModels[idx].load_model(global_weights)

                    # ==================
                    # WJDA
                    self.LocalModels[idx].load_global_subspace(self.global_mean, self.global_std)
                    self.LocalModels[idx].model.proto_classifier.load_proto(self.global_model.proto_classifier.proto)
                    # ==================

                w, loss = self.LocalModels[idx].update_weights(global_round=epoch)

                # ======================================
                # WJDA
                local_U.append(self.LocalModels[idx].get_domain_prototype(self.args.k))
                # ======================================

                local_losses.append(copy.deepcopy(loss))
                local_weights.append(copy.deepcopy(w))
                acc = self.LocalModels[idx].test_accuracy()
                test_accuracy += acc

            # ==========================
            # WJDA
            self.aggregate_global_subspace(local_U)
            # self.aggregate_global_svd(local_U)
            # ==========================

            # update global weights
            global_weights = self.average_weights(local_weights, idxs_users)
            # global_weights = self.average_weights_svd(local_weights, idxs_users)
            self.global_model.load_state_dict(global_weights)

            # print loss
            loss_avg = sum(local_losses) / len(local_losses)
            train_loss.append(loss_avg)
            cur_g_acc = self.global_test_accuracy()
            if cur_g_acc > self.global_best_acc:
                self.global_best_acc = cur_g_acc
                # self.Save_CheckPoint(self.args.logdir + '/best_model.pth')
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

    # =========================================
    # WJDA
    def aggregate_global_subspace(self, list_U):
        """
        聚合全局主子空间（简化为平均）
        list_U: List of [D, k]
        return: U_global [D, k]
        """
        global_means = []
        global_stds = []
        for mean, std in list_U:
            global_means.append(mean)
            global_stds.append(std)

        self.global_mean = torch.stack(global_means).mean(dim=0)
        self.global_std = torch.stack(global_stds).mean(dim=0)

        # return torch.stack(list_U).mean(dim=0)


    def global_test_accuracy(self):
        # 域适应
        self.global_model.eval()
        accuracy = 0
        cnt = 0
        domain_acc = []
        if self.args.leave_domain:
            # 域泛化
            # for domain, dataloader in self.global_testloader.items():
            for batch_idx, (X, y) in enumerate(self.global_testloader):
                X = X.to(self.device)
                y = y.to(self.device)
                z, p = self.global_model(X)
                # y_pred = p.argmax(1)
                z = self.global_model.linear_proto(z)
                z_norm = torch.div(z, torch.norm(z, p=2, dim=1, keepdim=True))
                output_local = torch.matmul(z_norm, self.global_model.proto_classifier.proto.to(self.device))
                p = self.global_model.scaling * output_local
                y_pred = p.argmax(1)

                accuracy += Accuracy(y, y_pred)
                cnt += 1
            domain_acc.append(accuracy / cnt)
            self.logger.info(f'|| Server Domain: {self.args.test_domain} Accuracy: {accuracy / cnt} ||')
            # self.logger.info(f'|| Server Domain Avg: {sum(domain_acc) / len(domain_acc)}')
            return sum(domain_acc) / len(domain_acc)
        else:
            # 域偏移
            for domain, dataloader in self.global_testloader.items():
                for batch_idx, (X, y) in enumerate(dataloader):
                    X = X.to(self.device)
                    y = y.to(self.device)
                    z, p = self.global_model(X)
                    # y_pred = p.argmax(1)
                    z_norm = torch.div(z, torch.norm(z, p=2, dim=1, keepdim=True))
                    output_local = torch.matmul(z_norm, self.global_model.proto_classifier.proto.to(self.device))
                    p = self.global_model.scaling * output_local
                    y_pred = p.argmax(1)

                    accuracy += Accuracy(y, y_pred)
                    cnt += 1
                domain_acc.append(accuracy / cnt)
                self.logger.info(f'|| Server Domain: {domain} Accuracy: {accuracy / cnt} ||')
            self.logger.info(f'|| Server Domain Avg: {sum(domain_acc) / len(domain_acc)}')
            return sum(domain_acc) / len(domain_acc)
   
