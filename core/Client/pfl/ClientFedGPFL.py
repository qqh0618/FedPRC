import copy

import torch.nn as nn
import torch.optim as optim
import torch
import numpy as np
import torch.nn.functional as F

from core.Client.ClientBase import Client





class ClientFedGPFL(Client):
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
        # ----------------------------------------------------------------
        # gpfl 设置
        self.feature_dim = list(self.model.classifier.parameters())[0].shape[1]
        self.GCE = None
        self.CoV = None

        # -----------------------------------------------------------------

    def update_weights(self, global_round):   # 训练模型
        self.model.to(self.device)
        self.model.train()
        epoch_loss = []
        optimizer = self.optim

        GCE_opt = optim.SGD(self.GCE.parameters(), lr=self.args.lr_sh_rate, weight_decay=0.0)
        CoV_opt = optim.SGD(self.CoV.parameters(), lr=self.args.lr_sh_rate, weight_decay=0.0)
        self.GCE_frozen = copy.deepcopy(self.GCE)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=self.args.lr_sh_rate, gamma=0.5)
        self.trainloader = self.get_trainloader()
        sample_per_class = torch.zeros(self.args.num_classes).to(self.device)
        for x, y in self.trainloader:
            for yy in y:
                sample_per_class[yy.item()] += 1
        sample_per_class = sample_per_class / torch.sum(
            sample_per_class)
        generic_conditional_input = torch.zeros(self.feature_dim).to(self.device)
        personalized_conditional_input = torch.zeros(self.feature_dim).to(self.device)
        try:
            generic_conditional_input, personalized_conditional_input = self.setGCE(sample_per_class,
                                                                               generic_conditional_input,
                                                                               personalized_conditional_input)
        except:
            generic_conditional_input = torch.zeros(self.feature_dim).to(self.device)
            personalized_conditional_input = torch.zeros(self.feature_dim).to(self.device)

        for iter in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (X, y) in enumerate(self.trainloader):
                X = X.to(self.device)
                y = y.to(self.device)
                optimizer.zero_grad()
                GCE_opt.zero_grad()
                CoV_opt.zero_grad()
                feats, p = self.model(X)

                # ------------------------------
                # gpfl
                feat_P = self.CoV(feats, personalized_conditional_input)
                feat_G = self.CoV(feats, generic_conditional_input)

                p = self.model.classifier(feat_P)
                softmax_loss = self.GCE(feat_G, y)
                # ------------------------------

                loss = self.ce(p, y)

                loss = loss + softmax_loss

                emb = torch.zeros_like(feats)
                for i, yy in enumerate(y):
                    emb[i, :] = self.GCE_frozen.embedding(yy).detach().data
                loss += torch.norm(feat_G - emb, 2) * 0.001

                loss.backward()
                if self.args.clip_grad != None:
                    nn.utils.clip_grad_norm_(self.model.parameters(), max_norm = self.args.clip_grad)
                optimizer.step()
                GCE_opt.step()
                CoV_opt.step()
                batch_loss.append(loss.item())
            total_loss = sum(batch_loss) / len(batch_loss)
            epoch_loss.append(total_loss)
            self.logger.info(f"Round {global_round} Client {self.idx} Epoch {iter} Loss {total_loss}")
        self.save_model(self.args.logdir + '/client_' + str(self.idx) + '.pth')
        return self.model.state_dict(),sum(epoch_loss) / len(epoch_loss)

    def setGCE(self, sample_per_class, generic_conditional_input, personalized_conditional_input):
        embeddings = self.GCE.embedding(torch.tensor(range(self.args.num_classes), device=self.device))
        for l, emb in enumerate(embeddings):  # 计算全局GCE
            generic_conditional_input.data += emb / self.args.num_classes
            personalized_conditional_input.data += emb * sample_per_class[l]

        return generic_conditional_input, personalized_conditional_input