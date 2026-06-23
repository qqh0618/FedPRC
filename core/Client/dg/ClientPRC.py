import copy

import torch.nn as nn
import torch.optim as optim
import torch
import torch.nn.functional as F
import numpy as np
from core.Client.ClientBase import Client
from utils import Accuracy



import torch
import torch.nn as nn
import torch.nn.functional as F





class DisReg(nn.Module):
    def __init__(self, args, n_classes):
        super().__init__()  # 调用父类(nn.Module)的构造函数
        self.args = args  # 保存传入的参数
        self.n_classes = n_classes  # 保存类别数
        self.soft = nn.Softmax(dim=1)  # 创建一个Softmax层，用于计算输入向量的softmax

        # 创建一个单位矩阵，大小为n_classes x n_classes
        tester = torch.eye(self.n_classes)
        # 计算单位矩阵的标准差的平均值，用于后续损失计算
        self.batch_gamma = tester.std(dim=0).mean().item()

    def forward(self, X, Y, pro1):  # x是pred，Y是真实target，pro1是feature
        # 如果输入X或Y的维度大于2，将它们重塑为二维张量
        if len(X.shape) > 2:
            X = torch.reshape(X, (X.shape[0], np.prod(X.shape[1:])))
            Y = torch.reshape(Y, (Y.shape[0], np.prod(Y.shape[1:])))

        # 计算pro1中所有成对点之间的欧氏距离的平方
        pdist_x = torch.pdist(pro1, p=2).pow(2)   # 计算两行之间的欧氏距离的平方
        # 计算非零距离的中位数，作为均匀性损失的尺度因子
        sigma_unif_x = torch.median(pdist_x[pdist_x != 0])   # 计算非零距离的中位数

        # 计算均匀性损失，即所有成对距离的指数加权平均值
        unif_loss = pdist_x.mul(-1 / sigma_unif_x).exp().mean()   # 均值越均匀效果越好
        loss = 0.1 * unif_loss  # 均匀性损失的权重

        return loss


def coral_loss(f_s, f_t):
    # f: [B, d]
    # align second-order moments between source and target
    d = f_s.size(1)
    # covariances
    Cs = (f_s - f_s.mean(0)).T @ (f_s - f_s.mean(0)) / (f_s.size(0)-1)
    Ct = (f_t - f_t.mean(0)).T @ (f_t - f_t.mean(0)) / (f_t.size(0)-1)
    return ((Cs - Ct)**2).sum() / (4 * d * d)


class ClientFedPRC(Client):
    """
    This class is for train the local model with input global model(copied) and output the updated weight
    args: argument
    Loader_train,Loader_val,Loaders_test: input for training and inference
    user: the index of local model
    idxs: the index for data of this local model
    logger: log the loss and the process
    """
    def __init__(self, args, model, local_idx_dataidx_map, idx, logger, code_length, num_classes, device):
        super().__init__(args, model,local_idx_dataidx_map, idx, logger, code_length, num_classes, device)
        # =====================
        # WJDA
        self.local_U = None
        self.global_U = None
        self.dkl = nn.KLDivLoss(reduction='batchmean')
        self.uvreg = DisReg(self.args,num_classes)
        self.T = 3
        self.global_mean = None
        self.global_std = None
        # self.proto_classifier = Proto_Classifier(self.args.code_len, self.args.num_classes)
        self.text_feature = torch.from_numpy(np.load("data/pacs/text_features.npy")).to(self.device)
        # =====================


    def update_weights(self, global_round):   # 训练模型
        self.model.to(self.device)
        self.model.train()
        epoch_loss = []
        optimizer = self.sgd
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=self.args.lr_sh_rate, gamma=0.5)
        # proto不参与反向传播和梯度优化
        for param in self.model.proto_classifier.parameters():
            param.requires_grad = False
        # 选取完整dataloader的10%训练
        self.trainloader = self.get_trainloader()
        for i in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (X, y) in enumerate(self.trainloader):
                X = X.to(self.device)
                y = y.to(self.device)
                optimizer.zero_grad()

                f, pred = self.model(X)

                global_bias = 0
                if self.global_mean is not None:
                    global_f = self.style_transfer(f, self.args.style)
                    soft_g_f = F.softmax(global_f)*0.5
                    global_bias = coral_loss(soft_g_f, f)
                    # f = f+global_f

                if self.global_mean is not None:
                    f = f + self.global_mean
                f = self.model.linear_proto(f)
                f_norm = torch.norm(f, dim=1, keepdim=True)
                f = torch.div(f, f_norm)
                f = torch.nan_to_num(f)
                out = self.model.proto_classifier.proto
                output_local = torch.matmul(f, out.to(self.device))
                output_local = self.model.scaling * output_local

                uv_loss = self.uvreg(output_local, y, f)

                loss = self.ce(output_local, y)+self.args.uv*uv_loss+self.args.coral*global_bias 

                loss.backward()
                if self.args.clip_grad != None:
                    nn.utils.clip_grad_norm_(self.model.parameters(), max_norm = self.args.clip_grad)
                optimizer.step()
                batch_loss.append(loss.item())
            total_loss = sum(batch_loss) / len(batch_loss)
            epoch_loss.append(total_loss)

            self.logger.info(f"Round {global_round} Client {self.idx} Epoch {i} Loss {total_loss}")


        self.save_model(self.args.logdir + '/client_' + str(self.idx) + '.pth')
        return self.model.state_dict(), sum(epoch_loss) / len(epoch_loss)

    def test_accuracy(self):
        self.model.eval()
        accuracy = 0
        cnt = 0

        for batch_idx, (X, y) in enumerate(self.testloader):
            X = X.to(self.device)
            y = y.to(self.device)
            z, _ = self.model(X)

            z = self.model.linear_proto(z)
            z_norm = torch.div(z, torch.norm(z, p=2, dim=1, keepdim=True))
            output_local = torch.matmul(z_norm, self.model.proto_classifier.proto.to(self.device))
            p = self.model.scaling * output_local
            y_pred = p.argmax(1)
            accuracy += Accuracy(y, y_pred)
            cnt += 1
        return accuracy / cnt

    def get_domain_prototype(self, k):
        with torch.no_grad():
            self.model.to(self.device)
            self.model.eval()
            # 初始化原型向量

            for batch_idx, (X, y) in enumerate(self.trainloader):
                X = X.to(self.device)
                f, _ = self.model(X)
                feat_sum, feat_square_sum, count = self.sum_feature(f)
                feat_mean = feat_sum / float(count)
                feat_var = feat_square_sum / float(count) - feat_mean ** 2
                feat_std = torch.sqrt(feat_var + 1e-5)
                style_stat = [feat_mean, feat_std]
                style_stat = [stat.to(self.device) for stat in style_stat]
            return style_stat

    def estimate_local_subspace_prototype(self, k):
        """
        还可以考虑用原型方法来估计子空间特征矩阵
        :param k:
        :return:
        """
        with torch.no_grad():
            self.model.to(self.device)
            self.model.eval()
            features = []
            labels = []
            # 初始化原型向量
            prototypes = torch.zeros((self.args.num_classes, 512)).to(self.device)
            for batch_idx, (X, y) in enumerate(self.trainloader):
                X = X.to(self.device)
                f, _ = self.model(X)
                # 用于计算协方差矩阵
                features.append(f.cpu()/torch.norm(f.cpu(), dim=1, keepdim=True))
                # 计算原型向量
                for i in range(len(y)):
                    prototypes[y[i]] += f[i]
            # 计算主轴空间矩阵
            features = torch.cat(features, dim=0)
            features = features - torch.mean(features, dim=0)
            return prototypes


    def load_global_subspace(self, mean, std):
        """
        加载全局子空间矩阵
        :param subspace_matrix:
        :return:
        """
        # self.global_U = subspace_matrix.to(self.device)
        if mean is not None:
            # print('load global subspace')
            self.global_mean = mean
            self.global_std = std



    def sum_feature(self, feat):
        """
        获取当前特征的域知识
        :param feat:
        :return:
        """
        feat = feat.detach()
        size = feat.shape
        assert (len(size) == 2)
        N, C = size
        count = N
        feat = feat.transpose(1, 0)

        feat_sum = feat.reshape(C, -1).sum(axis=1)
        feat_square = feat ** 2
        feat_square_sum = feat_square.reshape(C, -1).sum(axis=1)
        # import pdb; pdb.set_trace()
        return feat_sum, feat_square_sum, count

    def style_transfer(self, f, alpha=1.0,
                       interpolation_weights=None):
        assert (0.0 <= alpha <= 1.0)

        feat = self.adaIN_StyleStat_ContentFeat(f)
        feat = feat * alpha + f * (1 - alpha)
        return feat

    def adaIN_StyleStat_ContentFeat(self, content_feat):
        size = content_feat.size()
        style_mean, style_std = self.global_mean,self.global_std
        content_mean, content_std = self.calc_mean_std(content_feat)

        normalized_feat = (content_feat - content_mean.expand(
            size)) / content_std.expand(size)
        # normalized_feat = torch.div(content_feat, torch.norm(content_feat, p=2, dim=1, keepdim=True))
        return normalized_feat * style_std.expand(size) + style_mean.expand(size)

    def calc_mean_std(self, feat, eps=1e-5):
        # eps is a small value added to the variance to avoid divide-by-zero.
        size = feat.size()
        assert (len(size) == 2)
        feat_mean = feat.mean(dim=0, keepdim=True)  # [1, C]
        feat_std = feat.std(dim=0, unbiased=False, keepdim=True) + eps  # [1, C]
        return feat_mean, feat_std