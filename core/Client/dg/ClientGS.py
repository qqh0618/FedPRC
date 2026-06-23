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
import clip

# class UVRegDG(nn.Module):
#     """
#     UVRegDG: improved UVReg for Federated Domain Generalization.
#     - X: logits (batch, n_classes)
#     - Y: labels (batch,)  (int64)  -- required for class-conditional prototype alignment
#     - pro1: features (batch, feat_dim)
#     - domain: domain ids for each sample in the batch (batch,) - ints from 0..D-1
#     """
#     def __init__(
#         self,
#         n_classes: int,
#         feat_dim: int = None,
#         subsample_pairs: int = 1024,
#         weight_unif: float = 2.5,
#         weight_std: float = 0.5,
#         weight_proto: float = 1.0,
#         use_worst_domain: bool = False,
#         eps: float = 1e-8,
#     ):
#         super().__init__()
#         self.n_classes = n_classes
#         self.subsample_pairs = subsample_pairs
#         self.soft = nn.Softmax(dim=1)
#         self.eps = eps
#
#         # weights for each loss term
#         self.w_unif = weight_unif
#         self.w_std = weight_std
#         self.w_proto = weight_proto
#
#         # how to aggregate domain losses in this local step (False: mean, True: worst-domain)
#         self.use_worst_domain = use_worst_domain
#
#         # precompute batch_gamma baseline from identity as original code (scalar)
#         tester = torch.eye(self.n_classes)
#         self.register_buffer('batch_gamma', tester.std(dim=0).mean().detach())
#
#     @staticmethod
#     def _pairwise_sq_dists(x):
#         """Compute pairwise squared euclidean distances via (x - x)^2 trick.
#            x assumed (n, d) and already normalized if desired.
#         """
#         # x @ x^T gives dot products
#         # use: ||a-b||^2 = ||a||^2 + ||b||^2 - 2 a.b
#         xx = (x * x).sum(dim=1, keepdim=True)  # (n,1)
#         dists = xx + xx.t() - 2.0 * (x @ x.t())
#         # numerical stability
#         dists = torch.clamp(dists, min=0.0)
#         return dists
#
#     def _subsample_pairwise(self, feats):
#         """Return vector of pairwise squared distances (upper-triangle) possibly sub-sampled to limit O(B^2)."""
#         n = feats.shape[0]
#         if n < 2:
#             return feats.new_zeros(0)
#         # decide all pairs or subsample indices
#         # if subsample_pairs <= 0: use all pairs
#         if self.subsample_pairs <= 0 or n*(n-1)//2 <= self.subsample_pairs:
#             pd = torch.pdist(feats, p=2).pow(2)  # original style: upper-triangle vector
#             return pd
#         # else: randomly sample pairs
#         i = torch.randint(0, n, (self.subsample_pairs,), device=feats.device)
#         j = torch.randint(0, n, (self.subsample_pairs,), device=feats.device)
#         # avoid too many identical pairs
#         valid = (i != j)
#         if not valid.all():
#             i = i[valid]
#             j = j[valid]
#         # clamp length
#         d = (feats[i] - feats[j]).pow(2).sum(dim=1)
#         return d
#
#     def forward(self, X, Y, pro1, domain=None):
#         """
#         X: logits (B, C)
#         Y: labels (B,) int
#         pro1: features (B, F)
#         domain: (B,) int domain id per sample. If None, treat whole batch as single domain.
#         """
#         # flatten X if needed (keep as logits)
#         if X.dim() > 2:
#             X = X.view(X.shape[0], -1)
#         # flatten Y if needed
#         if Y is not None and Y.dim() > 1:
#             Y = Y.view(Y.shape[0], -1).squeeze(-1)
#
#         device = X.device
#         B = X.shape[0]
#
#         # 1) normalize features to unit norm (helps uniformity metrics)
#         pro = F.normalize(pro1, p=2, dim=1, eps=self.eps)
#
#         # ---------- Uniformity losses ----------
#         # compute pairwise squared dists (subsampled)
#         pair_d2 = self._subsample_pairwise(pro)  # vector of squared dists
#         # robust scale: median of non-zero distances or fallback to mean
#         nonzero = pair_d2[pair_d2 > self.eps]
#         if nonzero.numel() == 0:
#             sigma = pair_d2.mean().clamp(min=self.eps)
#         else:
#             sigma = nonzero.median()
#         sigma = sigma.detach() + self.eps
#
#         # global unif loss (like original)
#         unif_loss_global = (pair_d2.mul(-1.0 / sigma).exp().mean()) if pair_d2.numel() > 0 else torch.tensor(0.0, device=device)
#
#         # domain-wise unif: encourage each domain's features to also be spread out (prevents domain collapse)
#         if domain is None:
#             unif_loss_domain = unif_loss_global
#         else:
#             domains = torch.unique(domain)
#             domain_losses = []
#             for d in domains:
#                 mask = (domain == d)
#                 if mask.sum() < 2:
#                     continue
#                 feats_d = pro[mask]
#                 pd_d = self._subsample_pairwise(feats_d)
#                 if pd_d.numel() == 0:
#                     continue
#                 nz = pd_d[pd_d > self.eps]
#                 if nz.numel() == 0:
#                     sig_d = pd_d.mean().clamp(min=self.eps)
#                 else:
#                     sig_d = nz.median()
#                 domain_losses.append(pd_d.mul(-1.0 / (sig_d + self.eps)).exp().mean())
#             unif_loss_domain = torch.stack(domain_losses).mean() if len(domain_losses) > 0 else torch.tensor(0.0, device=device)
#
#         # combine uniformity: a weighted combination (give more importance to domain-wise when many domains)
#         unif_loss = 0.5 * unif_loss_global + 0.5 * unif_loss_domain
#
#         # ---------- Class-conditional prototype alignment ----------
#         # For each (domain, class) compute centroid; then compute pairwise distances across domains for same class.
#         proto_loss_terms = []
#         if Y is not None and domain is not None:
#             domains = torch.unique(domain)
#             classes = torch.unique(Y)
#             for c in classes:
#                 # find per-domain centroids for class c
#                 per_domain_centroids = []
#                 for d in domains:
#                     mask = (Y == c) & (domain == d)
#                     if mask.sum() == 0:
#                         continue
#                     feats_cd = pro[mask]
#                     centroid = feats_cd.mean(dim=0)
#                     centroid = F.normalize(centroid, p=2, dim=0, eps=self.eps)
#                     per_domain_centroids.append(centroid)
#                 if len(per_domain_centroids) <= 1:
#                     continue
#                 centroids = torch.stack(per_domain_centroids, dim=0)  # (k, feat)
#                 # minimize variance between centroids for this class (i.e., align same-class across domains)
#                 # equivalent to mean pairwise squared distances among centroids
#                 dmat = self._pairwise_sq_dists(centroids)
#                 # take upper triangle mean
#                 mean_pair = dmat.sum() / (dmat.shape[0]**2 + self.eps)  # normalized proxy
#                 proto_loss_terms.append(mean_pair)
#         proto_loss = torch.stack(proto_loss_terms).mean() if len(proto_loss_terms) > 0 else torch.tensor(0.0, device=device)
#
#         # ---------- Prediction diversity (std loss), class-aware ----------
#         soft_out = self.soft(X)  # (B, C)
#         # compute per-class std over samples that belong to that class OR over whole batch? we'll do over whole batch probability for each class,
#         # but to avoid forcing classes not present in batch, we'll mask classes absent in batch.
#         class_present = torch.zeros(self.n_classes, dtype=torch.bool, device=device)
#         if Y is not None:
#             present = torch.unique(Y)
#             class_present[present.long()] = True
#         # compute std per class over batch
#         std_per_class = soft_out.std(dim=0, unbiased=False)
#         # only penalize classes that are present in this local batch (can't force variance for absent classes)
#         relevant_std = std_per_class[class_present] if class_present.any() else std_per_class
#         if relevant_std.numel() == 0:
#             std_loss = torch.tensor(0.0, device=device)
#         else:
#             std_loss = F.relu(self.batch_gamma - relevant_std).mean()
#
#         # ---------- Domain-robust aggregation (optional) ----------
#         # Optionally, compute per-domain combined loss and take max (worst-domain) to be robust.
#         # Here we create per-domain scalar losses for reporting / worst-domain option.
#         if domain is not None:
#             domains = torch.unique(domain)
#             per_domain_vals = []
#             for d in domains:
#                 mask = (domain == d)
#                 # domain unif (if computed earlier)
#                 # for simplicity use domain unif per d if available via recompute
#                 if mask.sum() < 2:
#                     per_domain_unif = torch.tensor(0.0, device=device)
#                 else:
#                     feats_d = pro[mask]
#                     pd_d = self._subsample_pairwise(feats_d)
#                     nz = pd_d[pd_d > self.eps]
#                     if nz.numel() == 0:
#                         sig_d = pd_d.mean().clamp(min=self.eps)
#                     else:
#                         sig_d = nz.median()
#                     per_domain_unif = pd_d.mul(-1.0 / (sig_d + self.eps)).exp().mean() if pd_d.numel() > 0 else torch.tensor(0.0, device=device)
#                 # prediction std on domain
#                 if mask.sum() == 0:
#                     per_domain_std = torch.tensor(0.0, device=device)
#                 else:
#                     per_domain_soft = soft_out[mask]
#                     per_domain_std = per_domain_soft.std(dim=0, unbiased=False).mean()
#                 per_domain_val = self.w_unif * per_domain_unif + self.w_std * F.relu(self.batch_gamma - per_domain_std).mean()
#                 per_domain_vals.append(per_domain_val)
#             if len(per_domain_vals) > 0:
#                 per_domain_vals = torch.stack(per_domain_vals)
#                 if self.use_worst_domain:
#                     domain_robust_term = per_domain_vals.max()
#                 else:
#                     domain_robust_term = per_domain_vals.mean()
#             else:
#                 domain_robust_term = torch.tensor(0.0, device=device)
#         else:
#             domain_robust_term = torch.tensor(0.0, device=device)
#
#         # ---------- combine final loss ----------
#         # main combination: uniformity + std + prototype alignment + domain_robust (optional)
#         loss = self.w_unif * unif_loss + self.w_std * std_loss + self.w_proto * proto_loss
#
#
#
#         return loss

class UVRegDG(nn.Module):
    """
    UVRegDG: improved UVReg for Federated Domain Generalization.
    - X: logits (batch, n_classes)
    - Y: labels (batch,)  (int64)  -- required for class-conditional prototype alignment
    - pro1: features (batch, feat_dim)
    - domain: domain ids for each sample in the batch (batch,) - ints from 0..D-1
    """
    def __init__(
        self,
        n_classes: int,
        feat_dim: int = None,
        subsample_pairs: int = 1024,
        weight_unif: float = 2.5,
        weight_std: float = 0.5,
        weight_proto: float = 1.0,
        use_worst_domain: bool = False,
        eps: float = 1e-8,
    ):
        super().__init__()
        self.n_classes = n_classes
        self.subsample_pairs = subsample_pairs
        self.soft = nn.Softmax(dim=1)
        self.eps = eps

        # weights for each loss term
        self.w_unif = weight_unif
        self.w_std = weight_std
        self.w_proto = weight_proto

        # how to aggregate domain losses in this local step (False: mean, True: worst-domain)
        self.use_worst_domain = use_worst_domain

        # precompute batch_gamma baseline from identity as original code (scalar)
        tester = torch.eye(self.n_classes)
        self.register_buffer('batch_gamma', tester.std(dim=0).mean().detach())

    @staticmethod
    def _pairwise_sq_dists(x):
        """Compute pairwise squared euclidean distances via (x - x)^2 trick.
           x assumed (n, d) and already normalized if desired.
        """
        # x @ x^T gives dot products
        # use: ||a-b||^2 = ||a||^2 + ||b||^2 - 2 a.b
        xx = (x * x).sum(dim=1, keepdim=True)  # (n,1)
        dists = xx + xx.t() - 2.0 * (x @ x.t())
        # numerical stability
        dists = torch.clamp(dists, min=0.0)
        return dists

    def _subsample_pairwise(self, feats):
        """Return vector of pairwise squared distances (upper-triangle) possibly sub-sampled to limit O(B^2)."""
        n = feats.shape[0]
        if n < 2:
            return feats.new_zeros(0)
        # decide all pairs or subsample indices
        # if subsample_pairs <= 0: use all pairs
        if self.subsample_pairs <= 0 or n*(n-1)//2 <= self.subsample_pairs:
            pd = torch.pdist(feats, p=2).pow(2)  # original style: upper-triangle vector
            return pd
        # else: randomly sample pairs
        i = torch.randint(0, n, (self.subsample_pairs,), device=feats.device)
        j = torch.randint(0, n, (self.subsample_pairs,), device=feats.device)
        # avoid too many identical pairs
        valid = (i != j)
        if not valid.all():
            i = i[valid]
            j = j[valid]
        # clamp length
        d = (feats[i] - feats[j]).pow(2).sum(dim=1)
        return d

    def forward(self, X, Y, pro1, domain=None):
        """
        X: logits (B, C)
        Y: labels (B,) int
        pro1: features (B, F)
        domain: (B,) int domain id per sample. If None, treat whole batch as single domain.
        """
        # flatten X if needed (keep as logits)
        if X.dim() > 2:
            X = X.view(X.shape[0], -1)
        # flatten Y if needed
        if Y is not None and Y.dim() > 1:
            Y = Y.view(Y.shape[0], -1).squeeze(-1)

        device = X.device
        B = X.shape[0]

        # 1) normalize features to unit norm (helps uniformity metrics)
        pro = F.normalize(pro1, p=2, dim=1, eps=self.eps)

        # ---------- Uniformity losses ----------
        # compute pairwise squared dists (subsampled)
        pair_d2 = self._subsample_pairwise(pro)  # vector of squared dists
        # robust scale: median of non-zero distances or fallback to mean
        nonzero = pair_d2[pair_d2 > self.eps]
        if nonzero.numel() == 0:
            sigma = pair_d2.mean().clamp(min=self.eps)
        else:
            sigma = nonzero.median()
        sigma = sigma.detach() + self.eps

        # global unif loss (like original)
        unif_loss_global = (pair_d2.mul(-1.0 / sigma).exp().mean()) if pair_d2.numel() > 0 else torch.tensor(0.0, device=device)

        # domain-wise unif: encourage each domain's features to also be spread out (prevents domain collapse)
        if domain is None:
            unif_loss_domain = unif_loss_global
        else:
            domains = torch.unique(domain)
            domain_losses = []
            for d in domains:
                mask = (domain == d)
                if mask.sum() < 2:
                    continue
                feats_d = pro[mask]
                pd_d = self._subsample_pairwise(feats_d)
                if pd_d.numel() == 0:
                    continue
                nz = pd_d[pd_d > self.eps]
                if nz.numel() == 0:
                    sig_d = pd_d.mean().clamp(min=self.eps)
                else:
                    sig_d = nz.median()
                domain_losses.append(pd_d.mul(-1.0 / (sig_d + self.eps)).exp().mean())
            unif_loss_domain = torch.stack(domain_losses).mean() if len(domain_losses) > 0 else torch.tensor(0.0, device=device)

        # combine uniformity: a weighted combination (give more importance to domain-wise when many domains)
        unif_loss = 0.5 * unif_loss_global + 0.5 * unif_loss_domain

        # ---------- Class-conditional prototype alignment ----------
        # For each (domain, class) compute centroid; then compute pairwise distances across domains for same class.
        proto_loss_terms = []
        if Y is not None and domain is not None:
            domains = torch.unique(domain)
            classes = torch.unique(Y)
            for c in classes:
                # find per-domain centroids for class c
                per_domain_centroids = []
                for d in domains:
                    mask = (Y == c) & (domain == d)
                    if mask.sum() == 0:
                        continue
                    feats_cd = pro[mask]
                    centroid = feats_cd.mean(dim=0)
                    centroid = F.normalize(centroid, p=2, dim=0, eps=self.eps)
                    per_domain_centroids.append(centroid)
                if len(per_domain_centroids) <= 1:
                    continue
                centroids = torch.stack(per_domain_centroids, dim=0)  # (k, feat)
                # minimize variance between centroids for this class (i.e., align same-class across domains)
                # equivalent to mean pairwise squared distances among centroids
                dmat = self._pairwise_sq_dists(centroids)
                # take upper triangle mean
                mean_pair = dmat.sum() / (dmat.shape[0]**2 + self.eps)  # normalized proxy
                proto_loss_terms.append(mean_pair)
        proto_loss = torch.stack(proto_loss_terms).mean() if len(proto_loss_terms) > 0 else torch.tensor(0.0, device=device)

        # ---------- Prediction diversity (std loss), class-aware ----------
        soft_out = self.soft(X)  # (B, C)
        # compute per-class std over samples that belong to that class OR over whole batch? we'll do over whole batch probability for each class,
        # but to avoid forcing classes not present in batch, we'll mask classes absent in batch.
        class_present = torch.zeros(self.n_classes, dtype=torch.bool, device=device)
        if Y is not None:
            present = torch.unique(Y)
            class_present[present.long()] = True
        # compute std per class over batch
        std_per_class = soft_out.std(dim=0, unbiased=False)
        # only penalize classes that are present in this local batch (can't force variance for absent classes)
        relevant_std = std_per_class[class_present] if class_present.any() else std_per_class
        if relevant_std.numel() == 0:
            std_loss = torch.tensor(0.0, device=device)
        else:
            std_loss = F.relu(self.batch_gamma - relevant_std).mean()


        # ---------- combine final loss ----------
        # main combination: uniformity + std + prototype alignment + domain_robust (optional)
        loss = self.w_unif * unif_loss + self.w_std * std_loss + self.w_proto * proto_loss



        return loss


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

from collections import OrderedDict

class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)

class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class Proto_Classifier(nn.Module):
    def __init__(self, feat_in, num_classes):
        super(Proto_Classifier, self).__init__()
        P = self.generate_random_orthogonal_matrix(feat_in, num_classes)
        I = torch.eye(num_classes)
        one = torch.ones(num_classes, num_classes)
        M = np.sqrt(num_classes / (num_classes-1)) * torch.matmul(P, I-((1/num_classes) * one))

        self.proto = M

    def generate_random_orthogonal_matrix(self, feat_in, num_classes):
        a = np.random.random(size=(feat_in, num_classes))
        P, _ = np.linalg.qr(a)
        P = torch.tensor(P).float()
        assert torch.allclose(torch.matmul(P.T, P), torch.eye(num_classes), atol=1e-06), torch.max(torch.abs(torch.matmul(P.T, P) - torch.eye(num_classes)))
        return P

    def load_proto(self, proto):
        self.proto = copy.deepcopy(proto)

    def forward(self, label):
        # produce the prototypes w.r.t. the labels
        target = self.proto[:, label].T ## B, d  output: B, d

        return target

class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])

    def forward(self, x: torch.Tensor):
        return self.resblocks(x)





class ClientFedGS(Client):
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
        self.uvreg = UVReg(self.args,num_classes)
        self.T = 3
        self.global_mean = None
        self.global_std = None
        # self.proto_classifier = Proto_Classifier(self.args.code_len, self.args.num_classes)
        self.text_feature = torch.from_numpy(np.load("data/pacs/text_features.npy")).to(self.device)
        self.text = clip.tokenize(["a photo of a dog",
                                    "a photo of a elephant",
                                    "a photo of a giraffe",
                                    "a photo of a guitar",
                                    "a photo of a horse",
                                    "a photo of a house",
                                    "a photo of a person",
                                    ]).to(self.device)
        # =====================


    def update_weights(self, global_round):   # 训练模型
        self.model.to(self.device)
        self.model.train()
        epoch_loss = []
        optimizer = self.sgd
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=self.args.lr_sh_rate, gamma=0.5)
        # optimizer.add_param_group({'params': self.model.text_encoder.parameters()})
        # proto不参与反向传播和梯度优化
        # for param in self.model.proto_classifier.parameters():
        #     param.requires_grad = False
        # 选取完整dataloader的10%训练

        self.trainloader = self.get_trainloader()
        for iter in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (X, y) in enumerate(self.trainloader):
                X = X.to(self.device)
                y = y.to(self.device)
                optimizer.zero_grad()

                f, pred = self.model(X)

                text_f = self.model.text_encoder(self.text)

                f = f/torch.norm(f, dim=1, keepdim=True)
                text_f = text_f/torch.norm(text_f, dim=1, keepdim=True)

                logit_scale = self.model.logit_scale.exp()
                logits_per_image = logit_scale * f @ text_f.T
                loss = self.ce(logits_per_image, y)

                loss.backward()
                if self.args.clip_grad != None:
                    nn.utils.clip_grad_norm_(self.model.parameters(), max_norm = self.args.clip_grad)
                optimizer.step()
                batch_loss.append(loss.item())
            total_loss = sum(batch_loss) / len(batch_loss)
            epoch_loss.append(total_loss)
            self.logger.info(f"Round {global_round} Client {self.idx} Epoch {iter} Loss {total_loss}")


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
            z = z/torch.norm(z, dim=1, keepdim=True)
            text_f = self.model.text_encoder(self.text)
            text_f = text_f/torch.norm(text_f, dim=1, keepdim=True)
            p = self.model.logit_scale.exp() * z @ text_f.T
            p =  F.softmax(p, dim=1)

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
            # U, _, _ = torch.svd(features)
            # subspace_matrix = U[:, :k]
            # =========================================================
            # DA 不一定靠谱好使，目前可解释性还不清楚
            # 计算原型向量     # 这种方式可以替代原有的ETF的计算方式
            # for i in range(self.args.num_classes):
            #     prototypes[i] = prototypes[i] / torch.norm(prototypes[i])

            # ================================
            # 计算原型均值和协方差矩阵
            # 计算均值
            # mean_prototypes = torch.mean(prototypes, dim=0)

            # ================================
            # 求原型向量与主轴空间矩阵的内积
            # for i in range(self.args.num_classes):
            #     prototypes[i] = torch.matmul(subspace_matrix, prototypes[i])
            #     prototypes[i] = prototypes[i] / torch.norm(prototypes[i])
            # # 求主轴空间矩阵与原型向量的外积
            # subspace_matrix = torch.matmul(subspace_matrix, prototypes.t())
            # subspace_matrix = subspace_matrix / torch.norm(subspace_matrix)

            # 计算原型
            # =========================================================


            # return subspace_matrix, prototypes
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