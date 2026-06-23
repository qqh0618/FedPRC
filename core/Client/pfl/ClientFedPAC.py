import copy

import torch.nn as nn
import torch.optim as optim
import torch
import numpy as np
import torch.nn.functional as F

from core.Client.ClientBase import Client


def set_params(net, global_model, finetune=False):
    """
        Reference: Be careful with the state_dict[key].
        https://discuss.pytorch.org/t/how-to-copy-a-modified-state-dict-into-a-models-state-dict/64828/4.
    """

    for key in global_model.state_dict().keys():  # 头部个性化
        if key not in ['head.weight', 'head.bias']:
            net.state_dict()[key]=global_model.state_dict()[key]
        # net.state_dict()[key].copy_(global_model.state_dict()[key])



class ClientFedPAC(Client):
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
        self.mse = nn.MSELoss().to(device)
        self.global_protos = None
        self.local_proto = None
        self.v = None
        self.h = None
        self.sizes_label = None
    def update_weights(self, global_round):   # 训练模型
        self.model.to(self.device)
        self.model.train()
        epoch_loss = []
        optimizer = self.optim
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=self.args.lr_sh_rate, gamma=0.5)
        self.trainloader = self.get_trainloader()

        self.v, self.h = self.statistics_extraction(self.model, self.trainloader)

        local_protos1 = self.get_local_protos(self.model, self.trainloader)
        self.sizes_label = self.size_label(self.trainloader).to(self.device)

        for iter in range(1):
            batch_loss = []
            for batch_idx, (X, y) in enumerate(self.trainloader):
                X = X.to(self.device)
                y = y.to(self.device)
                optimizer.zero_grad()
                f, p = self.model(X)

                loss = self.ce(p, y)

                loss = loss
                loss.backward()
                if self.args.clip_grad != None:
                    nn.utils.clip_grad_norm_(self.model.parameters(), max_norm = self.args.clip_grad)
                optimizer.step()
                batch_loss.append(loss.item())
            total_loss = sum(batch_loss) / len(batch_loss)
            epoch_loss.append(total_loss)
            self.logger.info(f"Round {global_round} Client {self.idx} Epoch {iter} Loss {total_loss}")

        # FedPAC
        for name, param in self.model.named_parameters():  # 再训练主干
            if name in ['head.weight', 'head.bias']:
                param.requires_grad = False
            else:
                param.requires_grad = True
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, self.model.parameters()), lr=self.args.lr, momentum=self.args.rho,
                              weight_decay=0.1)


        for iter in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (X, y) in enumerate(self.trainloader):
                X = X.to(self.device)
                y = y.to(self.device)
                optimizer.zero_grad()
                f, protos = self.model(X)

                loss = self.ce(p, y)

                protos = F.leaky_relu(protos)

                protos_new = protos.clone().detach()
                for i in range(len(y)):
                    yi = y[i].item()
                    if yi in self.global_protos:
                        protos_new[i] = self.global_protos[yi].detach()
                    else:
                        protos_new[i] = local_protos1[yi].detach()
                loss1 = self.mse(protos_new, protos)

                loss = loss + loss1.item()
                loss.backward()
                if self.args.clip_grad != None:
                    nn.utils.clip_grad_norm_(self.model.parameters(), max_norm = self.args.clip_grad)
                optimizer.step()
                batch_loss.append(loss.item())
            total_loss = sum(batch_loss) / len(batch_loss)
            epoch_loss.append(total_loss)
            self.logger.info(f"Round {global_round} Client {self.idx} Epoch {iter} Loss {total_loss}")

        self.local_proto = self.get_local_protos(self.model, self.trainloader)

        self.save_model(self.args.logdir + '/client_' + str(self.idx) + '.pth')
        return self.model.state_dict(),sum(epoch_loss) / len(epoch_loss)

    def statistics_extraction(self, net, train_data):
        model = copy.deepcopy(net)
        cls_keys = ['head.weight', 'head.bias']
        g_params = model.state_dict()[cls_keys[0]] if isinstance(cls_keys, list) else model.state_dict()[cls_keys]
        d = g_params[0].shape[0]
        feature_dict = {}
        with torch.no_grad():
            for inputs, labels in train_data:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs, _, features = model(inputs)
                feat_batch = features.clone().detach()
                for i in range(len(labels)):
                    yi = labels[i].item()
                    if yi in feature_dict.keys():
                        feature_dict[yi].append(feat_batch[i, :])
                    else:
                        feature_dict[yi] = [feat_batch[i, :]]
        for k in feature_dict.keys():
            feature_dict[k] = torch.stack(feature_dict[k])

        py = self.prior_label(train_data)
        py2 = py.mul(py)
        v = 0
        h_ref = torch.zeros((self.num_classes, d), device=self.device)
        for k in range(self.num_classes):
            if k in feature_dict.keys():
                feat_k = feature_dict[k]
                num_k = feat_k.shape[0]
                feat_k_mu = feat_k.mean(dim=0)
                h_ref[k] = py[k] * feat_k_mu
                v += (py[k] * torch.trace((torch.mm(torch.t(feat_k), feat_k) / num_k))).item()
                v -= (py2[k] * (torch.mul(feat_k_mu, feat_k_mu))).sum().item()
        v = v / torch.tensor(len(train_data.dataset)).to(self.device).item()

        return v, h_ref

    def prior_label(self, dataset):
        py = torch.zeros(self.num_classes)
        total = len(dataset.dataset)
        data_loader = iter(dataset)
        iter_num = len(data_loader)
        for it in range(iter_num):
            images, labels = next(data_loader)
            for i in range(self.num_classes):
                py[i] = py[i] + (i == labels).sum()
        py = py / (total)
        return py

    def get_local_protos(self, net, train_data):
        model = copy.deepcopy(net)
        local_protos_list = {}
        for inputs, labels in train_data:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            outputs, _, features = model(inputs)
            protos = features.clone().detach()
            for i in range(len(labels)):
                if labels[i].item() in local_protos_list.keys():
                    local_protos_list[labels[i].item()].append(protos[i, :])
                else:
                    local_protos_list[labels[i].item()] = [protos[i, :]]
        local_protos = self.get_protos(local_protos_list)
        return local_protos

    def get_protos(self, protos):
        """
        Returns the average of the feature embeddings of samples from per-class.
        """
        protos_mean = {}
        for [label, proto_list] in protos.items():
            proto = 0 * proto_list[0]
            for i in proto_list:
                proto += i
            protos_mean[label] = proto / len(proto_list)

        return protos_mean

    def size_label(self, dataset):
        py = torch.zeros(self.num_classes)
        total = len(dataset.dataset)
        data_loader = iter(dataset)
        iter_num = len(data_loader)
        for it in range(iter_num):
            images, labels = next(data_loader)
            for i in range(self.num_classes):
                py[i] = py[i] + (i == labels).sum()
        py = py / (total)
        size_label = py * total
        return size_label