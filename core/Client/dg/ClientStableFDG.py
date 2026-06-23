import copy

import torch
from torch.nn import  functional as F
import torch.nn as nn
import torch.optim as optim
from core.Client.ClientBase import Client


class ClientStableFDG(Client):
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
        # ===============
        # StableFDG
        self.local_style = None
        # ===============

    def update_weights(self, global_round):   # 训练模型
        self.model.to(self.device)
        self.model.train()
        epoch_loss = []
        optimizer = self.optim
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=self.args.lr_sh_rate, gamma=0.5)
        part = 0.1
        self.trainloader = self.get_trainloader()
        for iter in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (X, y) in enumerate(self.trainloader):
                X = X.to(self.device)
                y = y.to(self.device)
                optimizer.zero_grad()

                # ====================
                # StableFDG, 核心在于AFH模块和OMA模块，需要适应我们当前场景
                # 推理和正常模型一样
                original_batch = X.size(0)
                label_ori = y.clone().detach()

                feat, flat_feat = self.model.forward_cb(X, y)

                output, label = self.model(X, label=y,
                                            supplemental_samples=(feat, y, flat_feat),
                                            style=self.local_style)

                loss = F.cross_entropy(output, label)

                # ====================
                # _,p = self.model(X)
                # loss = self.ce(p,y)
                loss.backward()
                if self.args.clip_grad != None:
                    nn.utils.clip_grad_norm_(self.model.parameters(), max_norm = self.args.clip_grad)
                optimizer.step()
                batch_loss.append(loss.item())
            total_loss = sum(batch_loss) / len(batch_loss)
            epoch_loss.append(total_loss)
            self.logger.info(f"Round {global_round} Client {self.idx} Epoch {iter} Loss {total_loss}")
        # self.save_model(self.args.logdir + '/client_' + str(self.idx) + '.pth')
        return self.model.state_dict(),sum(epoch_loss) / len(epoch_loss)


    def style_collect(self, train_loader=None, local_model=None):

        samples = []
        num_sample_per_domain = 0

        if local_model is None:
            initial_model = copy.deepcopy(self.model)
        else:
            initial_model = local_model

        self.trainloader = self.get_trainloader()
        with torch.no_grad():
            for i, (X, y) in enumerate(self.trainloader):
                X = X.to(self.device)
                y = y.to(self.device)
                feat, _ = initial_model.forward_cb(X, y)

                mu = feat[0].mean(dim=[2, 3], keepdim=True)
                var = feat[0].var(dim=[2, 3], keepdim=True)
                sig = (var + 1e-6).sqrt()

                mu_batch = torch.squeeze(mu)
                sig_batch = torch.squeeze(sig)

                B = mu_batch.size(0)
                C = mu_batch.size(1)

                for i in range(mu_batch.size(0)):

                    if num_sample_per_domain == 0:
                        concate = torch.cat((mu_batch[i].detach().cpu(), sig_batch[i].detach().cpu()), dim=0)
                        samples = concate.resize(1, C * 2)
                    else:
                        concate = torch.cat((mu_batch[i].detach().cpu(), sig_batch[i].detach().cpu()), dim=0)
                        samples = torch.cat(
                            (samples, concate.resize(1, C * 2)), dim=0)

                    num_sample_per_domain += 1

        sample_style_mean = torch.mean(samples, 0)
        sample_style_sqrt = (torch.var(samples,0) + 1e-6).sqrt()

        return [sample_style_mean, sample_style_sqrt]

    def load_style(self, style):
        self.local_style = copy.deepcopy(style)
