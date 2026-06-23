import torch.nn as nn
import torch.optim as optim
import torch
import numpy as np
import torch.nn.functional as F

from core.Client.ClientBase import Client


class UVReg(nn.Module):
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
        pdist_x = torch.pdist(pro1, p=2).pow(2)
        # 计算非零距离的中位数，作为均匀性损失的尺度因子
        sigma_unif_x = torch.median(pdist_x[pdist_x != 0])

        # 计算均匀性损失，即所有成对距离的指数加权平均值
        unif_loss = pdist_x.mul(-1 / sigma_unif_x).exp().mean()

        # 计算X的softmax输出
        logsoft_out = self.soft(X)
        # 计算softmax输出的标准差
        logsoft_out_std = logsoft_out.std(dim=0)
        # 计算标准差损失，鼓励softmax输出的标准差大于self.batch_gamma
        std_loss = torch.mean(F.relu(self.batch_gamma - logsoft_out_std))

        # 计算最终损失，为标准差损失和均匀性损失的加权和
        loss = (
                0.5 * std_loss  # 标准差损失的权重
                + 2.5 * unif_loss  # 均匀性损失的权重
        )

        # 返回最终损失和一个包含不同损失组件的数组
        # 数组中的前两个和最后两个元素被设置为0，可能是因为格式要求或未使用
        return loss


class ClientFedUV(Client):
    """
    This class is for train the local model with input global model(copied) and output the updated weight
    args: argument 
    Loader_train,Loader_val,Loaders_test: input for training and inference
    user: the index of local model
    idxs: the index for data of this local model
    logger: log the loss and the process
    """
    def __init__(self, args, model, local_idx_dataidx_map, idx, logger, code_length, num_classes, device):
        super().__init__(args, model, local_idx_dataidx_map, idx, logger, code_length, num_classes, device)
        self.uv_criterion = UVReg(args, num_classes)

    def update_weights(self, global_round):   # 训练模型
        self.model.to(self.device)
        self.model.train()
        epoch_loss = []
        optimizer = self.optim
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=self.args.lr_sh_rate, gamma=0.5)
        self.trainloader = self.get_trainloader()
        for iter in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (X, y) in enumerate(self.trainloader):
                X = X.to(self.device)
                y = y.to(self.device)
                optimizer.zero_grad()
                f, p = self.model(X)

                loss_uv = self.uv_criterion(p, y, f)
                loss = self.ce(p, y)

                loss = loss + loss_uv
                loss.backward()
                if self.args.clip_grad != None:
                    nn.utils.clip_grad_norm_(self.model.parameters(), max_norm = self.args.clip_grad)
                optimizer.step()
                batch_loss.append(loss.item())
            total_loss = sum(batch_loss) / len(batch_loss)
            epoch_loss.append(total_loss)
            self.logger.info(f"Round {global_round} Client {self.idx} Epoch {iter} Loss {total_loss}")
        self.save_model(self.args.logdir + '/client_' + str(self.idx) + '.pth')
        return self.model.state_dict(),sum(epoch_loss) / len(epoch_loss)
