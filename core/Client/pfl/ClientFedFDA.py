from copy import deepcopy
from sklearn.model_selection import StratifiedKFold
from statsmodels.stats.correlation_tools import cov_nearest
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchmin

from core.Client.ClientBase import Client
import torch
class ClientFedFDA(Client):
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
        # ===========================
        # FedFDA
        self.D = 512
        self.eps = 1e-4
        self.means_beta = torch.ones(size=(self.num_classes,)) * 0.5  # 均值及方差
        self.cov_beta = torch.Tensor([0.5])
        # local statistics
        self.means = torch.Tensor(torch.rand([self.num_classes, self.D]))
        self.covariance = torch.Tensor(torch.eye(self.D))
        counts, priors = self.get_label_distribution("train")
        self.priors = priors.cpu()
        self.priors = self.priors + self.eps
        self.priors = self.priors / self.priors.sum()
        self.class_counts = counts.cpu()
        # global statistics
        self.global_means = deepcopy(self.means)
        self.global_covariance = deepcopy(self.covariance)
        # interpolated statistics
        self.adaptive_means = deepcopy(self.means)
        self.adaptive_covariance = deepcopy(self.covariance)
        self.num_train = len(self.trainloader.dataset)
        # interpolation term solver
        self.single_beta = False
        self.local_beta = False
        self.num_cv_folds = 2
        self.min_samples = self.num_cv_folds
        if self.local_beta:
            self.single_beta = True
        # ===========================
    def update_weights(self, global_round):   # 训练模型
        self.model.to(self.device)
        self.model.train()
        epoch_loss = []
        optimizer = self.optim
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=self.args.lr_sh_rate, gamma=0.5)
        self.trainloader = self.get_trainloader()
        # =================================
        # FedFDA
        self.means = self.means.to(self.device)
        self.covariance = self.covariance.to(self.device)
        self.adaptive_means = self.adaptive_means.to(self.device)
        self.adaptive_covariance = self.adaptive_covariance.to(self.device)
        self.global_means = self.global_means.to(self.device)
        self.global_covariance = self.global_covariance.to(self.device)

        #
        self.set_lda_weights(self.global_means, self.global_covariance, self.priors)
        for param in self.model.classifier[0].parameters():
            param.requires_grad_(False)

        feats_ep = []
        labels_ep = []
        # =================================
        for iter in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (X, y) in enumerate(self.trainloader):
                X = X.to(self.device)
                y = y.to(self.device)
                optimizer.zero_grad()
                feats, p = self.model(X)
                loss = self.ce(p, y)
                loss.backward()
                if self.args.clip_grad != None:
                    nn.utils.clip_grad_norm_(self.model.parameters(), max_norm = self.args.clip_grad)
                # =============================
                # FedFDA
                feats_ep.append(feats.detach())
                labels_ep.append(y.cpu().numpy())
                # =============================
                optimizer.step()
                batch_loss.append(loss.item())
            total_loss = sum(batch_loss) / len(batch_loss)
            epoch_loss.append(total_loss)


            self.logger.info(f"Round {global_round} Client {self.idx} Epoch {iter} Loss {total_loss}")
        # =============================
        # FedFDA
        feats_ep = torch.cat(feats_ep, dim=0)
        labels_ep = np.concatenate(labels_ep, axis=0)
        self.solve_beta(feats_ep, labels_ep)  # 1
        self.model.to(self.device)

        means_mle, scatter_mle, priors, counts = self.compute_mle_statistics(feats=feats_ep, labels=labels_ep)
        cov_mle = (scatter_mle / (np.sum(counts) - 1)) + 1e-4 + torch.eye(self.D).to(self.device)
        cov_psd = cov_nearest(cov_mle.cpu().numpy(), method="clipped")
        cov_psd = torch.Tensor(cov_psd).to(self.device)
        means_mle = torch.stack(
            [means_mle[i] if means_mle[i] is not None else self.global_means[i] for i in range(self.num_classes)])

        # =============================

        self.update(means_mle, cov_psd)
        self.save_model(self.args.logdir + '/client_' + str(self.idx) + '.pth')
        return self.model.state_dict(),sum(epoch_loss) / len(epoch_loss)

    def get_label_distribution(self, split="train"):
        if split == "train":
            loader = self.trainloader
        else:
            loader = self.testloader
        labels = []
        for i, (x, y) in enumerate(loader):
            labels.append(y.cpu().numpy())
        labels = np.concatenate(labels)
        label_counts = np.bincount(labels, minlength=self.num_classes)
        label_counts = torch.Tensor(label_counts).to(self.device)
        priors = np.bincount(labels, minlength=self.num_classes) / float(len(labels))
        priors = torch.Tensor(priors).float().to(self.device)
        return label_counts, priors


    def set_lda_weights(self, means=None, covariance=None, priors=None, use_lstsq=True):
        if means is None:
            means = self.means
        if covariance is None:
            covariance = self.covariance
        if priors is None:
            priors = self.priors
        with torch.no_grad():
            means = means.to(self.device)
            covariance = covariance.to(self.device)
            priors = priors.to(self.device)
            covariance = (1-self.eps)*covariance + self.eps * torch.trace(covariance)/self.D * torch.eye(self.D).cuda()
            if use_lstsq:
                coefs = torch.linalg.lstsq(covariance, means.T)[0].T
            else:
                coefs = torch.matmul(torch.linalg.inv(covariance), means.T).T  # 求逆矩阵, 获得一个和均值想同维度的矩阵

            intercepts = -0.5 * torch.diag(torch.matmul(means, coefs.T)) + torch.log(priors).to(self.device)  # 求均值的对角线元素
            self.model.classifier[0].weight.data = coefs.detach()
            self.model.classifier[0].bias.data = intercepts.detach()

    def solve_beta(self, feats, labels, seed=0):
        if self.local_beta:  # use only local
            self.means_beta = torch.ones(1).to(self.device)
            self.cov_beta = torch.ones(1).to(self.device)
            return
        vals, counts = np.unique(labels, return_counts=True)
        pruned_feats = deepcopy(feats)
        pruned_labels = deepcopy(labels)
        # remove classes with < K samples, as we cannot do StratifiedKFold
        for v, c in zip(vals, counts):
            if c < self.num_cv_folds:
                pruned_feats = pruned_feats[pruned_labels != v]
                pruned_labels = pruned_labels[pruned_labels != v]
        try:
            skf = StratifiedKFold(n_splits=self.num_cv_folds, random_state=seed, shuffle=True)
            l_feats_te, l_labels_te = [], []
            l_means, l_covs = [], []
            for i, (train_index, test_index) in enumerate(skf.split(pruned_feats, pruned_labels)):
                feats_tr, labels_tr = pruned_feats[train_index], pruned_labels[train_index]
                feats_te, labels_te = pruned_feats[test_index], pruned_labels[test_index]
                means_tr, scatter_tr, priors, counts = self.compute_mle_statistics(feats=feats_tr, labels=labels_tr)
                cov_tr = (scatter_tr / (np.sum(counts)-1)) + 1e-4 + torch.eye(self.D).to(self.device)
                cov_psd = cov_nearest(cov_tr.cpu().numpy(), method="clipped")
                cov_psd = torch.Tensor(cov_psd).to(self.device)
                means = torch.stack([means_tr[i] if means_tr[i] is not None else self.global_means[i] for i in range(self.num_classes)])
                l_means.append(means)
                l_covs.append(cov_psd)
                l_feats_te.append(feats_te)
                l_labels_te.append(labels_te)
        except: # not enough data; use local stats (this should only happen on a few emnist clients)
            self.means_beta = torch.zeros(1).to(self.device)
            self.cov_beta = torch.zeros(1).to(self.device)
            return

        loss_torch = lambda a: torch.sum(torch.stack([self.beta_classifier(a.clip(0,1),
                                                    l_means[i],
                                                    l_covs[i],
                                                    l_feats_te[i],
                                                    l_labels_te[i], )
                                                    for i in range(self.num_cv_folds)]))

        try:
            if self.single_beta:
                x = torchmin.minimize(loss_torch, x0=0.5*torch.ones(size=(1,)).cuda(), method="l-bfgs", max_iter=10, options={"gtol": 1e-3}).x.cpu().clip(0,1)
                self.means_beta = torch.ones_like(self.means_beta) * x[0]
                self.cov_beta = x[0]
            else:
                x = torchmin.minimize(loss_torch, x0=0.5*torch.ones(size=(2,)).cuda(), method="l-bfgs", max_iter=10, options={"gtol": 1e-3}).x.cpu().clip(0,1)
                self.means_beta = torch.ones_like(self.means_beta) * x[0]
                self.cov_beta = x[1]
        except:  # if optimization fails, use last used value of beta
            pass

    def compute_mle_statistics(self, split="train", feats=None, labels=None):
        self.model = self.model.to(self.device)
        means = [None] * self.num_classes
        if feats is None:
            feats, labels = self.compute_feats(split=split)  # 获取特征和标签
        counts = np.bincount(labels, minlength=self.num_classes)
        priors = np.bincount(labels, minlength=self.num_classes) / float(len(labels))
        priors = torch.Tensor(priors).float().to(self.device)

        for i, y in enumerate(np.unique(labels)):
            if len(labels == y) > self.min_samples:
                means[y] = torch.mean(feats[labels == y], axis=0)

        feats_centered = []
        for i, y in enumerate(np.unique(labels)):
            means_y = means[y] if means[y] is not None else self.global_means[y]
            f = feats[labels == y] - means_y
            feats_centered.append(f)

        feats_centered = torch.cat(feats_centered, dim=0)
        scatter = torch.mm(feats_centered.t(), feats_centered)
        # self.model = self.model.to("cpu")
        return means, scatter, priors, counts

    def compute_feats(self, split="train"):
        with torch.no_grad():
            if split == "train":
                loader = self.trainloader
            else:
                loader = self.testloader
            feats = []
            labels = []
            self.model.eval()
            self.model.to(self.device)
            for i, (x, y) in enumerate(loader):
                x = x.to(self.device)
                feat, _ = self.model.forward(x, return_feat=True)
                feats.append(feat)
                labels.append(y.cpu().numpy())
            feats = torch.cat(feats, axis=0)
            labels = np.concatenate(labels, axis=0)
        return feats, labels

    def beta_classifier(self, beta, means_local, cov_local, feats, labels):
        if self.single_beta:
            means = beta * means_local + (1-beta) * self.global_means
            cov = beta * cov_local + (1-beta) * self.global_covariance
        else:
            means = beta[0] * means_local + (1-beta[0]) * self.global_means
            cov = beta[-1] * cov_local + (1-beta[-1]) * self.global_covariance
        y_pred = self.lda_classify(feats, means=means, covariance=cov, priors=self.priors, use_lstsq=True)
        return torch.nn.functional.cross_entropy(y_pred, torch.LongTensor(labels).cuda())

    def lda_classify(self, Z, means=None, covariance=None, priors=None, use_lstsq=True):
        if priors is None:
            priors = self.priors
        covariance = (1-self.eps)*covariance + self.eps * torch.trace(covariance)/self.D * torch.eye(self.D).cuda()
        if use_lstsq:
            coefs = torch.linalg.lstsq(covariance, means.T)[0].T
        else:
            coefs = torch.matmul(torch.linalg.inv(covariance), means.T).T
        intercepts = -0.5 * torch.diag(torch.matmul(means, coefs.T)) + torch.log(priors).to(self.device)
        return Z @ coefs.T + intercepts

    def update(self, means_mle, cov_mle):
        self.means_beta = self.means_beta.to(self.device)
        self.cov_beta = self.cov_beta.to(self.device)
        self.means.data = means_mle.data
        self.covariance.data = cov_mle.data
        self.adaptive_means.data = self.means_beta.unsqueeze(1) * means_mle.data + (1-self.means_beta.unsqueeze(1)) * self.global_means.data
        self.adaptive_covariance.data = self.cov_beta * cov_mle.data + (1-self.cov_beta) * self.global_covariance.data
        self.means_beta = self.means_beta.cpu()
        self.cov_beta = self.cov_beta.cpu()