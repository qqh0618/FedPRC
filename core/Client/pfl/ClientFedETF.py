import torch.nn as nn
import torch.optim as optim
import torch
import numpy as np
import torch.nn.functional as F

from core.Client.ClientBase import Client
from utils import Accuracy


def balanced_softmax_loss(logits, labels, sample_per_class, reduction="mean"):
    """Compute the Balanced Softmax Loss between `logits` and the ground truth `labels`.
    Args:
      labels: A int tensor of size [batch].
      logits: A float tensor of size [batch, no_of_classes].
      sample_per_class: A int tensor of size [no of classes].
      reduction: string. One of "none", "mean", "sum"
    Returns:
      loss: A float tensor. Balanced Softmax Loss.
    """
    spc = sample_per_class.type_as(logits)
    spc = spc.unsqueeze(0).expand(logits.shape[0], -1)
    logits = logits + spc.log()
    loss = F.cross_entropy(input=logits, target=labels, reduction=reduction)
    return loss


def generate_sample_per_class(local_data, num_classes=10):
    sample_per_class = torch.tensor([0 for _ in range(num_classes)])

    for idx, (data, target) in enumerate(local_data):
        sample_per_class += torch.tensor([sum(target==i) for i in range(num_classes)])

    sample_per_class = torch.where(sample_per_class > 0, sample_per_class, 1)

    return sample_per_class


class ClientFedETF(Client):
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
        # =======================
        #     FedETF
        self.etf_proto = False
        self.linear_proto = False
        self.balancedloss = True
        # =======================

    def update_weights(self, global_round):   # 训练模型
        self.model.to(self.device)
        self.model.train()
        epoch_loss = []
        optimizer = self.optim
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=self.args.lr_sh_rate, gamma=0.5)
        self.trainloader = self.get_trainloader()


        # =========================================
        # FedETF
        if self.etf_proto:
            for name, param in self.model.named_parameters():
                param.requires_grad = False
            self.model.proto_classifier.proto.requires_grad = True
            optimizer = torch.optim.SGD([self.model.proto_classifier.proto], lr=0.1)
        if self.linear_proto:
            for name, param in self.model.named_parameters():
                if 'linear_proto' in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False

        sample_per_class = generate_sample_per_class(self.trainloader, self.num_classes)

        # net.proto_classifier.load_proto(global_model.proto_classifier.proto)

        # =========================================

        for iter in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (X, y) in enumerate(self.trainloader):
                X = X.to(self.device)
                y = y.to(self.device)
                optimizer.zero_grad()
                f, p = self.model(X)
                f = self.model.linear_proto(f)
                # 归一化特征
                f = torch.div(f, torch.norm(f, p=2, dim=1, keepdim=True))

                # ========================================

                output_local = torch.matmul(f, self.model.proto_classifier.proto.to(self.device))
                output_local = self.model.scaling * output_local
                if self.balancedloss:
                    loss = balanced_softmax_loss(output_local, y, sample_per_class)
                else:
                    # For local personalization
                    loss = self.ce(output_local, y)
                # ========================================
                loss = loss
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

    def test_accuracy(self):
        # 传统联邦学习
        self.model.eval()
        accuracy = 0
        cnt = 0

        for batch_idx, (X, y) in enumerate(self.testloader):
            X = X.to(self.device)
            y = y.to(self.device)
            f, _ = self.model(X)
            f = self.model.linear_proto(f)
            # 归一化特征
            f = torch.div(f, torch.norm(f, p=2, dim=1, keepdim=True))
            p = self.model.scaling * torch.matmul(f,
                                                         self.model.proto_classifier.proto.to(self.device))

            y_pred = p.argmax(1)
            accuracy += Accuracy(y, y_pred)
            cnt += 1
        return accuracy / cnt