import copy

import torch
from torch.utils.data import TensorDataset, DataLoader


from core.Client.dg.ClientGS import ClientFedGS
from core.Client.dg.ClientPRC import ClientFedVDDG

from core.Server.ServerBase import Server
from core.Client.ClientFedAvg import ClientFedAvg
from tqdm import tqdm
import numpy as np

from model.simpletext import CustomTextClassifier
from utils import average_weights, Accuracy
from mem_utils import MemReporter
import time
import torch.nn as nn
import torch.nn.functional as F

import clip

def compute_mahalanobis_bias(client_feat, global_mean, global_cov_inv):
    """
    client_feat: np.array of shape (N, D)
    global_mean: np.array of shape (D,)
    global_cov_inv: np.array of shape (D, D)
    """
    delta = client_feat - global_mean  # (N, D)
    distances = np.einsum('nd,df,nf->n', delta, global_cov_inv, delta)  # 快速计算 N 个样本的距离
    return distances.mean()  # 可用于客户端指标或正则化



class ServerFedGS(Server):
    def __init__(self, args, global_model, Loader_train, Loaders_local_test, Loader_global_test, logger, device,
                 net_idx_dataidx_map):
        super().__init__(args, global_model, Loader_train, Loaders_local_test, Loader_global_test, logger, device,
                         net_idx_dataidx_map)
        # ===============
        # WJDA
        self.global_U = None
        self.global_mean = None
        self.global_std = None
        # ===============
        # self.global_model.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        #
        # self.text = clip.tokenize(["a photo of a dog",
        #                       "a photo of a elephant",
        #                       "a photo of a giraffe",
        #                       "a photo of a guitar",
        #                       "a photo of a horse",
        #                       "a photo of a house",
        #                       "a photo of a person",
        #                       ]).to(self.device)
        # self.global_model.text_encoder = CustomTextClassifier(self.text, num_classes=self.args.num_classes).to(self.device)
        # self.pre_train()
        #加载训练好的pt模型
        self.global_model.load_state_dict(torch.load("persuo_global_model.pt"))
        self.global_test_accuracy()

    def Create_Clints(self):
        for idx in range(self.args.num_clients):
            self.LocalModels.append(
                ClientFedGS(self.args, copy.deepcopy(self.global_model), self.net_idx_dataidx_map[idx], idx=idx,
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
            # ==========================

            # update global weights
            global_weights = self.average_weights(local_weights, idxs_users)
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

    # =========================================

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
                # z = z/torch.norm(z, dim=1, keepdim=True)
                # text_f = self.global_model.text_encoder(self.text)
                # text_f = text_f/torch.norm(text_f, dim=1, keepdim=True)
                # logit_scale = self.global_model.logit_scale.exp()
                # logits_per_image = logit_scale * z @ text_f.T
                # p =  F.softmax(logits_per_image, dim=1)

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
                    # z_norm = torch.div(z, torch.norm(z, p=2, dim=1, keepdim=True))
                    # output_local = torch.matmul(z_norm, self.global_model.proto_classifier.proto.to(self.device))
                    # p = self.global_model.scaling * output_local
                    y_pred = p.argmax(1)

                    accuracy += Accuracy(y, y_pred)
                    cnt += 1
                domain_acc.append(accuracy / cnt)
                self.logger.info(f'|| Server Domain: {domain} Accuracy: {accuracy / cnt} ||')
            self.logger.info(f'|| Server Domain Avg: {sum(domain_acc) / len(domain_acc)}')
            return sum(domain_acc) / len(domain_acc)

    # def average_weights(self, w, idxs_users, list_U=None):
    #     # list_U是所有客户端的均值和标准差
    #     total_data_points = sum([len(self.LocalModels[r].train_idx) for r in idxs_users])
    #
    #     # 计算每个客户端原型与全局均值的距离（加上eps避免除0）
    #     distances = torch.tensor([
    #         torch.norm(list_U[i][0] - self.global_mean, p=1).item() + 1e-8
    #         for i in range(len(idxs_users))
    #     ])
    #
    #     # 反距离归一化权重（越接近全局均值，权重越大）
    #     weights = 1.0 / distances
    #     weights = weights / weights.sum()
    #
    #     # 初始化全局模型参数
    #     global_para = copy.deepcopy(w[0])
    #     # 加权平均
    #     for idx in range(len(idxs_users)):
    #         net_para = w[idx]
    #         if idx == 0:
    #             for key in net_para:
    #                 global_para[key] = net_para[key] * weights[idx]
    #         else:
    #             for key in net_para:
    #                 global_para[key] += net_para[key] * weights[idx]
    #
    #     return global_para

    def pre_train(self):
        self.global_model.train()
        self.ce = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.global_model.parameters(), lr=self.args.lr)

        data_loader = self.generate_trainloader(num_classes=self.args.num_classes)
        for iter in range(100):
            batch_loss = []
            for batch_idx, (X, y) in tqdm(enumerate(data_loader)):
                X = X.to(self.device)
                y = y.to(self.device)
                z, p = self.global_model(X)
                loss = self.ce(p, y)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                batch_loss.append(loss.item())
            self.logger.info(f'|| Global Pre-Train Epoch: {iter} Loss: {sum(batch_loss) / len(batch_loss)} ||')
            print(f'|| Global Pre-Train Epoch: {iter} Loss: {sum(batch_loss) / len(batch_loss)} ||')
        torch.save(self.global_model.state_dict(), f'persuo_global_model.pt')
    def generate_trainloader(self, feat_in=256, num_classes=10):

        a = np.random.random(size=(feat_in, num_classes))
        P, _ = np.linalg.qr(a)
        P = torch.tensor(P).float()

        I = torch.eye(num_classes)
        one = torch.ones(num_classes, num_classes)
        M = np.sqrt(num_classes / (num_classes - 1)) * torch.matmul(P, I - ((1 / num_classes) * one))

        # 求每一类的mean和std
        g_mean = torch.mean(M, dim=0)
        g_std = torch.std(M, dim=0)

        # 利用mean和std对每一类生成10000个224维的样本
        X = torch.randn(1000, feat_in*3)
        Xs = []
        ys = []
        for idx, (mean, std) in enumerate(zip(g_mean, g_std)):
            ts = X * std + mean
            ts = ts.reshape(1000, 3, np.sqrt(feat_in).astype(int), np.sqrt(feat_in).astype(int))
            Xs.append(ts)
            ys.append(torch.full((1000,), idx, dtype=torch.long))



        X = torch.cat(Xs, dim=0)
        y = torch.cat(ys, dim=0)
        return DataLoader(
            dataset=TensorDataset(X, y),
            batch_size=self.args.batch_size,
            shuffle=True,
        )



