import torch.nn as nn
import torch.optim as optim
import torch
import numpy as np
import torch.nn.functional as F

from core.Client.ClientBase import Client

from collections import Counter


class ClientFedNH(Client):
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

    def update_weights(self, global_round):   # 训练模型
        self.model.to(self.device)
        self.model.train()
        epoch_loss = []
        optimizer = self.optim
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=self.args.lr_sh_rate, gamma=0.5)
        self.trainloader = self.get_trainloader()

        # ------------------------
        # fednh专属
        count_by_class = Counter(self.trainloader.dataset.targets)
        temp = [count_by_class[cls] if cls in count_by_class.keys() else 1e-12 for cls in
                range(self.num_classes)]
        count_by_class_full = torch.tensor(temp).to(self.device)
        # ------------------------

        for iter in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (X, y) in enumerate(self.trainloader):
                X = X.to(self.device)
                y = y.to(self.device)
                optimizer.zero_grad()
                f, p = self.model(X)

                # --------------------------------------------------------------------------
                # 正确的nh版本，多一个对原型的nor处理 todo:测试时也需要需求logit方式
                feats_norm = torch.norm(f, p=2, dim=1, keepdim=True).clamp(min=1e-12)
                feats_embedding = torch.div(f, feats_norm)
                prototype = self.model.prototype  # 多了一个
                norm_prototype = torch.norm(prototype, p=2, dim=1, keepdim=True).clamp(min=1e-12)  # 又多一个
                normalized_prototype = torch.div(prototype, norm_prototype)  # 再多一个
                logits = torch.matmul(feats_embedding, normalized_prototype.T)
                y = self.model.scaling * logits
                # --------------------------------------------------------------------------

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
        self.save_model(self.args.logdir + '/client_' + str(self.idx) + '.pth')
        return self.model.state_dict(),sum(epoch_loss) / len(epoch_loss)
