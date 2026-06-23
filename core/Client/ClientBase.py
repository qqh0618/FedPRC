import time

import numpy as np
import torch
import scipy
from torch.utils.data import Dataset
import torch
import copy
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from datautil.domainsampling import get_loader
from utils import Accuracy,soft_predict

class Client(object):
    """
    This class is for train the local model with input global model(copied) and output the updated weight
    args: argument 
    Loader_train,Loader_val,Loaders_test: input for training and inference
    user: the index of local model
    idxs: the index for data of this local model
    logger: log the loss and the process
    """
    def __init__(self, args, model, local_idx_dataidx_map, idx, logger, code_length, num_classes, device):
        self.args = args
        self.logger = logger
        self.local_idx_dataidx_map = local_idx_dataidx_map
        self.parse_data()
        self.idx = idx
        self.ce = nn.CrossEntropyLoss() 
        self.device = device
        self.code_length = code_length
        self.kld = nn.KLDivLoss()
        self.mse = nn.MSELoss()
        self.model = copy.deepcopy(model)
        self.num_classes = num_classes
        self.adam = optim.Adam(self.model.parameters(), lr=self.args.lr)
        self.sgd = optim.SGD(self.model.parameters(), lr=self.args.lr, momentum=0.9,weight_decay=1e-5)
        self.optim = self.sgd
        self.global_model = copy.deepcopy(model)

        # CLIP专用
        if self.args.dataset == 'pacs':
            self.y_labels = [
                "a photo of dog",
                "a photo of elephant",
                "a photo of giraffe",
                'a photo of guitar',
                'a photo of horse',
                'a photo of house',
                'a photo of person',
            ]
    def test_accuracy(self):
        self.model.eval()
        accuracy = 0
        cnt = 0
        if self.args.model=='clip':
            for X, y in self.testloader:
                X = X.to(self.device)
                # X = X.half()
                y = y.to(self.device)
                p, _, _ = self.model.forward(X)
                y_pred = p.argmax(1)
                accuracy += Accuracy(y, y_pred)
                cnt += 1
            return accuracy / cnt
        else:
            if self.args.alg=='FedDDDA' or self.args.alg =='FedAGG':
                for batch_idx, (X, y) in enumerate(self.testloader):
                    X = X.to(self.device)
                    y = y.to(self.device)
                    z, _ = self.model(X)

                    z_norm = torch.div(z,torch.norm(z,p=2,dim=1, keepdim=True))
                    output_local = torch.matmul(z_norm, self.model.scaling.to(torch.float32)@ self.model.proto_classifier.proto.to(self.device))
                    p =  output_local
                    y_pred = p.argmax(1)
                    accuracy += Accuracy(y, y_pred)
                    cnt += 1
                return accuracy / cnt
            elif self.args.alg =='StableFDG':
                for batch_idx, (X, y) in enumerate(self.testloader):
                    X = X.to(self.device)
                    y = y.to(self.device)
                    p = self.model(X,y)
                    # y_pred = p.argmax(1)
                    if isinstance(p, tuple):
                        pred = p[0].max(1)[1]
                    else:
                        pred = p.max(1)[1]
                    accuracy += Accuracy(y,pred)
                    cnt += 1
                return accuracy/cnt
            else:
                for batch_idx, (X, y) in enumerate(self.testloader):
                    X = X.to(self.device)
                    # X = X.half()
                    y = y.to(self.device)
                    _, p = self.model(X)
                    y_pred = p.argmax(1)
                    accuracy += Accuracy(y, y_pred)
                    cnt += 1
                return accuracy/cnt

    def load_model(self, global_weights):
        self.model.load_state_dict(global_weights)
        self.global_model.load_state_dict(global_weights)

    def save_model(self, path):
        torch.save(self.model.state_dict(), path)


    def parse_data(self):
        for k,v in self.local_idx_dataidx_map.items():
            self.dataset = k
            self.train_idx = v['train']
            self.test_idx = v['test']
            self.testloader = get_loader(self.args, self.dataset, self.test_idx, train=False, args_test=None)
            self.trainloader = get_loader(self.args, self.dataset, self.train_idx, train=True, args_test=None)

    def get_trainloader(self):
        # 随机选取10%的数据
        start_time = time.time()
        num_samples = len(self.train_idx)
        num_samples_to_select = max(2, int(num_samples * self.args.part))
        # 必须保证除以batch_size后余数不等于1
        if num_samples_to_select % self.args.batch_size == 1:
            num_samples_to_select += 1
        selected_idxs = np.random.choice(self.train_idx, num_samples_to_select)
        # self.logger.info(f"Client {self.idx} selected {len(selected_idxs)} samples from {num_samples} samples || {selected_idxs}")
        dl = get_loader(self.args, self.dataset, selected_idxs, train=True, args_test=None)
        return dl