import time

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.distributions as distributions
from core.Client.ClientBase import Client

class ClientFedSR(Client):
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
        # ============================
        # parser.add_argument('--L2R_coeff', type=float, default=1e-2)
        # parser.add_argument('--CMI_coeff', type=float, default=5e-4)
        self.probabilistic = True
        self.z_dim = self.args.code_len//2  # 1/2的特征层长度
        # self.cls = nn.Linear(self.z_dim, self.num_classes).to(self.device)
        # self.cls = self.model.cls
        # self.r_mu = nn.Parameter(torch.zeros(self.num_classes, self.z_dim)).to(self.device)
        # self.r_sigma = nn.Parameter(torch.ones(self.num_classes, self.z_dim)).to(self.device)
        # self.C = nn.Parameter(torch.ones([])).to(self.device)
        self.L2R_coeff = 1e-2
        self.CMI_coeff = 1e-3
        # ============================
    def update_weights(self, global_round):   # 训练模型
        self.model.to(self.device)
        self.model.train()
        epoch_loss = []
        optimizer = self.optim
        # optimizer.add_param_group({'params':[self.model.r_mu,self.model.r_sigma,self.model.C],'lr':self.args.lr,'momentum':0.9})
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=self.args.lr_sh_rate, gamma=0.5)
        self.trainloader = self.get_trainloader()


        for iter in range(self.args.local_ep):
            batch_loss = []
            start_time = time.time()
            for batch_idx, (X, y) in enumerate(self.trainloader):
                X = X.to(self.device)
                y = y.to(self.device)
                # 转为半精度
                # X = X.half()
                # y = y.half()
                optimizer.zero_grad()
                if not self.probabilistic:
                    _, p = self.model(X)
                    loss = self.ce(p, y)
                else:
                    z, (z_mu, z_sigma) = self.featurize(X, return_dist=True)
                    logits = self.model.cls(z)
                    loss = self.ce(logits, y)

                    obj = loss

                    regL2R = torch.zeros_like(obj)
                    regCMI = torch.zeros_like(obj)
                    regNegEnt = torch.zeros_like(obj)
                    if self.L2R_coeff != 0.0:
                        regL2R = z.norm(dim=1).mean()
                        obj = obj + self.L2R_coeff * regL2R
                    if self.CMI_coeff != 0.0:
                        r_sigma_softplus = F.softplus(self.model.r_sigma)
                        r_mu = self.model.r_mu[y]
                        r_sigma = r_sigma_softplus[y]
                        z_mu_scaled = z_mu * self.model.C
                        z_sigma_scaled = z_sigma * self.model.C
                        regCMI = torch.log(r_sigma) - torch.log(z_sigma_scaled) + \
                                 (z_sigma_scaled ** 2 + (z_mu_scaled - r_mu) ** 2) / (2 * r_sigma ** 2) - 0.5
                        regCMI = regCMI.sum(1).mean()
                        obj = obj + self.CMI_coeff * regCMI

                # z_dist = distributions.Independent(distributions.normal.Normal(z_mu, z_sigma), 1)
                # mix_coeff = distributions.categorical.Categorical(X.new_ones(X.shape[0]))
                # mixture = distributions.mixture_same_family.MixtureSameFamily(mix_coeff, z_dist)
                # log_prob = mixture.log_prob(z)
                # regNegEnt = log_prob.mean()


                obj.backward()
                if self.args.clip_grad != None:
                    nn.utils.clip_grad_norm_(self.model.parameters(), max_norm = self.args.clip_grad)
                optimizer.step()
                batch_loss.append(loss.item())
            # print("一轮训练耗时:", time.time() - start_time, )
            total_loss = sum(batch_loss) / len(batch_loss)
            epoch_loss.append(total_loss)
            self.logger.info(f"Round {global_round} Client {self.idx} Epoch {iter} Loss {total_loss}")
        # self.save_model(self.args.logdir + '/client_' + str(self.idx) + '.pth')
        return self.model.state_dict(), sum(epoch_loss) / len(epoch_loss)

    def featurize(self, x, num_samples=1, return_dist=False):
        # if self.args.dataset in[ "pacs", "office_home","caltech10" ]:
        #     num_samples = 3*self.args.meta_num  # 域数量*每个域客户端数量
        # elif self.args.dataset =="office31":
        #     num_samples = 2*self.args.meta_num  # 域数量*每个域客户端数量


        if not self.probabilistic:
            return self.model(x)
        else:
            z_params, _ = self.model(x)

            z_mu = z_params[:, :self.z_dim]
            z_sigma = F.softplus(z_params[:, self.z_dim:])
            z_dist = distributions.Independent(distributions.normal.Normal(z_mu, z_sigma), 1)
            z = z_dist.rsample([num_samples]).view([-1, self.z_dim])

            if return_dist:
                return z, (z_mu, z_sigma)
            else:
                return z

    def test_accuracy(self):
        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_idx, (X, y) in enumerate(self.testloader):
                X = X.to(self.device)
                y = y.to(self.device)
                z, (z_mu, z_sigma) = self.featurize(X, return_dist=True)
                logits = self.model.cls(z)
                # preds = torch.softmax(self.cls(z), dim=1)
                # preds = preds.view([self.num_samples, -1, self.num_classes]).mean(0)
                logits = torch.softmax(logits, dim=1)
                logits = logits.view([self.args.meta_num, -1, self.num_classes]).mean(0)
                logits = torch.log(logits)
                _, predicted = torch.max(logits.data, 1)
                total += y.size(0)
                correct += (predicted == y).sum().item()
        return 100 * correct / total
