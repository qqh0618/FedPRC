import math

import torch
from torch import nn
import torch.nn.functional as F
import torchvision.models as models
import copy
import numpy as np
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


class ETFHead(nn.Module):
    def __init__(self, feat_in, num_classes):
        super(ETFHead, self).__init__()
        P = self.generate_random_orthogonal_matrix(feat_in, num_classes)
        I = torch.eye(num_classes)
        one = torch.ones(num_classes, num_classes)
        M = np.sqrt(num_classes / (num_classes-1)) * torch.matmul(P, I-((1/num_classes) * one))

        self.proto = M

        etf_rect = torch.ones((1, num_classes), dtype=torch.float32)
        self.etf_rect = etf_rect
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


# class ProtoNorm(nn.Module):
#     """
#     ProtoNorm: class-prototype-guided normalization, meant as a replacement for BatchNorm.
#     - feat_dim: number of channels (C)
#     - num_classes: number of classes
#     - eps, momentum: same meaning as BatchNorm
#     - affine: whether to learn gamma/beta (per-channel)
#     - track_running_stats: whether to keep running stats per class
#     Usage:
#         m = ProtoNorm(C, num_classes)
#         out = m(x, labels)  # during training requires labels (LongTensor shape [B])
#         m.eval(); out = m(x, labels=None)  # during eval labels optional; uses running stats
#     """
#     def __init__(self, feat_dim, num_classes, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True):
#         super(ProtoNorm, self).__init__()
#         self.feat_dim = feat_dim
#         self.num_classes = num_classes
#         self.eps = eps
#         # momentum used similar to BatchNorm: running = (1 - momentum) * running + momentum * batch_stat
#         self.momentum = momentum
#         self.affine = affine
#         self.track_running_stats = track_running_stats
#
#         # learnable affine parameters (per-channel)
#         if self.affine:
#             self.gamma = nn.Parameter(torch.ones(feat_dim))
#             self.beta = nn.Parameter(torch.zeros(feat_dim))
#         else:
#             self.register_parameter('gamma', None)
#             self.register_parameter('beta', None)
#
#         # running stats per class: running_mean_residual and running_var (shape: num_classes x feat_dim)
#         if self.track_running_stats:
#             self.register_buffer('running_mean', torch.zeros(num_classes, feat_dim))
#             self.register_buffer('running_var', torch.ones(num_classes, feat_dim))
#             self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))
#         else:
#             self.register_parameter('running_mean', None)
#             self.register_parameter('running_var', None)
#             self.register_parameter('num_batches_tracked', None)
#
#         # prototype matrix (feat_dim, num_classes). Initialize randomly similarly to your Proto_Classifier.
#         # Allow user to set via load_proto
#         P = self._generate_random_orthogonal_matrix(feat_dim, num_classes)
#         I = torch.eye(num_classes)
#         one = torch.ones(num_classes, num_classes)
#         M = np.sqrt(num_classes / (num_classes-1)) * torch.matmul(P, I-((1/num_classes) * one))
#         self.register_buffer('proto', M.float())  # (feat_dim, num_classes)
#
#     def _generate_random_orthogonal_matrix(self, feat_in, num_classes):
#         a = np.random.random(size=(feat_in, num_classes))
#         P, _ = np.linalg.qr(a)
#         P = torch.tensor(P).float()
#         return P
#
#     def load_proto(self, proto_tensor):
#         """
#         proto_tensor: tensor shaped (feat_dim, num_classes)
#         """
#         assert proto_tensor.shape == (self.feat_dim, self.num_classes)
#         self.proto = proto_tensor.float().to(self.running_mean.device if self.track_running_stats else torch.device('cpu'))
#
#     def forward(self, x, labels=None):
#         """
#         x: Tensor, shape either (B, C) or (B, C, H, W)
#         labels: LongTensor shape (B,) giving class id for each sample. If None in training mode -> error.
#                 In eval mode, labels optional (we'll use running stats based on label if provided; else fallback to per-sample prototype only).
#         returns y same shape as x.
#         """
#         if x.dim() not in (2, 4):
#             raise ValueError("ProtoNorm supports 2D (B,C) or 4D (B,C,H,W) tensors.")
#
#         is_4d = (x.dim() == 4)
#         if is_4d:
#             B, C, H, W = x.shape
#             x_flat = x.view(B, C, -1)  # B, C, S where S = H*W
#             S = H * W
#         else:
#             B, C = x.shape
#             x_flat = x.view(B, C, 1)   # treat as single spatial location
#             S = 1
#
#         # labels required during training
#         if self.training and labels is None:
#             raise ValueError("ProtoNorm in training mode requires labels (LongTensor of shape [B]).")
#
#         device = x.device
#         dtype = x.dtype
#
#         # get per-sample prototype: shape (B, C)
#         # proto is stored as (C, num_classes)
#         proto = self.proto.to(device=device, dtype=dtype)
#         if labels is not None:
#             # ensure labels on device
#             labels = labels.to(device)
#             # gather prototype for each sample
#             # proto[:, labels] -> (C, B). transpose -> (B, C)
#             proto_per_sample = proto[:, labels].T  # B, C
#         else:
#             # if labels None (eval w/o labels), fallback to zero prototype (or mean prototype?)
#             proto_per_sample = torch.zeros(B, C, device=device, dtype=dtype)
#
#         # expand proto to match spatial elements: (B, C, S)
#         proto_per_sample_exp = proto_per_sample.view(B, C, 1).expand(-1, -1, S)  # B, C, S
#
#         # residual r = x - proto
#         r = x_flat - proto_per_sample_exp  # B, C, S
#
#         # compute per-class stats across samples and spatial dims
#         # We'll compute sum and count per class using scatter_add
#         if labels is not None:
#             labels_flat = labels.view(-1)  # B
#         else:
#             # if no labels, treat all samples as class 0 to compute some stat (but usually eval uses running stats)
#             labels_flat = torch.zeros(B, dtype=torch.long, device=device)
#
#         # Prepare per-sample sums across spatial dims: sum over S -> shape (B, C)
#         r_sum_per_sample = r.sum(dim=2)  # B, C
#         r_sq_sum_per_sample = (r * r).sum(dim=2)  # B, C
#         counts_per_sample = torch.full((B,), S, dtype=torch.long, device=device)  # every sample contributes S elements
#
#         # accumulate per-class:
#         class_sum = torch.zeros(self.num_classes, C, device=device, dtype=dtype)  # num_classes x C
#         class_sq_sum = torch.zeros(self.num_classes, C, device=device, dtype=dtype)
#         class_count = torch.zeros(self.num_classes, device=device, dtype=torch.long)
#
#         # use scatter_add for sums
#         class_sum = class_sum.index_add(0, labels_flat, r_sum_per_sample)
#         class_sq_sum = class_sq_sum.index_add(0, labels_flat, r_sq_sum_per_sample)
#         class_count = class_count.index_add(0, labels_flat, counts_per_sample)
#
#         # compute class means and vars where count > 0
#         # to avoid divide by zero, create mask
#         count_float = class_count.float().view(self.num_classes, 1)  # num_classes x 1
#         has_count = (class_count > 0)
#         batch_mean = torch.zeros_like(class_sum)  # num_classes x C
#         batch_var = torch.zeros_like(class_sum)
#
#         if has_count.any():
#             idx = has_count.nonzero(as_tuple=False).squeeze(1)
#             batch_mean[idx] = class_sum[idx] / count_float[idx]
#             # E[x^2] - E[x]^2
#             ex2 = class_sq_sum[idx] / count_float[idx]
#             batch_var[idx] = ex2 - batch_mean[idx] * batch_mean[idx]
#             # numerical safeguard: var >= 0
#             batch_var[idx] = torch.clamp(batch_var[idx], min=0.0)
#
#         # Update running stats (only for classes present in batch)
#         if self.training and self.track_running_stats:
#             with torch.no_grad():
#                 # increment batch counter
#                 self.num_batches_tracked += 1
#                 for c in range(self.num_classes):
#                     if has_count[c]:
#                         self.running_mean[c] = (1 - self.momentum) * self.running_mean[c] + self.momentum * batch_mean[c]
#                         # unbiased-ish update like BN uses: use batch_var computed as above
#                         self.running_var[c] = (1 - self.momentum) * self.running_var[c] + self.momentum * batch_var[c]
#
#         # Now compute normalization for each sample. For sample i with label l:
#         # use stats = batch stats if available for its class, else running stats (if tracked), else zeros/ones fallback
#         # Build per-sample mean and var tensors: (B, C)
#         per_sample_mean = torch.zeros(B, C, device=device, dtype=dtype)
#         per_sample_var = torch.zeros(B, C, device=device, dtype=dtype)
#
#         for i in range(B):
#             l = int(labels_flat[i].item())
#             if has_count[l]:
#                 per_sample_mean[i] = batch_mean[l]
#                 per_sample_var[i] = batch_var[l]
#             elif self.track_running_stats:
#                 per_sample_mean[i] = self.running_mean[l].to(dtype=dtype)
#                 per_sample_var[i] = self.running_var[l].to(dtype=dtype)
#             else:
#                 # fallback
#                 per_sample_mean[i].zero_()
#                 per_sample_var[i].fill_(1.0)
#
#         # reshape means to (B, C, 1)
#         per_sample_mean_exp = per_sample_mean.view(B, C, 1)
#         per_sample_var_exp = per_sample_var.view(B, C, 1)
#
#         # normalize residuals: (r - mean) / sqrt(var + eps)
#         invstd = torch.rsqrt(per_sample_var_exp + self.eps)
#         r_norm = (r - per_sample_mean_exp) * invstd  # B, C, S
#
#         # apply affine
#         if self.affine:
#             gamma = self.gamma.to(device=device, dtype=dtype).view(1, C, 1)
#             beta = self.beta.to(device=device, dtype=dtype).view(1, C, 1)
#             y = r_norm * gamma + beta
#         else:
#             y = r_norm
#
#         # reconstruct original shape
#         if is_4d:
#             y = y.view(B, C, H, W)
#         else:
#             y = y.view(B, C)
#
#         return y

class ProtoNorm(nn.Module):
    def __init__(self, feat_in):
        super(ProtoNorm, self).__init__()
        P = self.generate_random_orthogonal_matrix(feat_in, feat_in)
        I = torch.eye(feat_in)
        one = torch.ones(feat_in, feat_in)
        M = np.sqrt(feat_in / (feat_in-1)) * torch.matmul(P, I-((1/feat_in) * one))

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


# class ProtoNorm2(nn.Module):
#     def __init__(self, feat_in):
#         super().__init__()
#         self.temperature = 0.1
#         self.adaptive_weight = nn.Parameter(torch.ones(feat_in))
#         a = torch.randn(feat_in, feat_in)
#         self.s, _, _ = np.linalg.svd(a)
#
#     def forward(self, x):
#         # 计算特征注意力, SE注意力机制
#         x = F.max_pool2d(x, kernel_size=2)
#
#
#         return out


class HUGClassifier(nn.Module):
    def __init__(self, feat_in, num_classes):
        super().__init__()
        # all_classifier = Parameter(torch.Tensor(num_c, 512).cuda())
        # stdv = 1. / math.sqrt(all_classifier.size(1))
        # all_classifier.data.uniform_(-stdv, stdv)
        # all_classifier.requires_grad = True

        self.all_classifier = nn.Parameter(torch.Tensor(num_classes, feat_in).cuda())
        stdv = 1. / math.sqrt(self.all_classifier.size(1))
        self.all_classifier.data.uniform_(-stdv, stdv)
        self.all_classifier.requires_grad = True


    def forward(self, x):
        out = torch.matmul(x, self.all_classifier.T)

        return out

class GEMNet(nn.Module):
    def __init__(self, res101, img_size, c, w, h,
                 attritube_num, cls_num, ucls_num, attr_group, w2v,
                 scale=20.0, device=None):

        super(GEMNet, self).__init__()
        self.device = device

        self.img_size = img_size
        # self.prototype_shape = prototype_shape
        self.attritube_num = attritube_num

        self.feat_channel = c
        self.feat_w = w
        self.feat_h = h

        self.ucls_num = ucls_num
        self.scls_num = cls_num - ucls_num
        self.attr_group = attr_group

        self.w2v_att = torch.from_numpy(w2v).float().to(self.device)  # 312 * 300
        assert self.w2v_att.shape[0] == self.attritube_num

        if scale<=0:
            self.scale = nn.Parameter(torch.ones(1) * 20.0)
        else:
            self.scale = nn.Parameter(torch.tensor(scale), requires_grad=False)


        self.backbone = res101

        # self.prototype_vectors = nn.Parameter(nn.init.normal_(torch.empty(self.prototype_shape)), requires_grad=True)  # a, c

        self.W = nn.Parameter(nn.init.normal_(torch.empty(self.w2v_att.shape[1], self.feat_channel)),
                               requires_grad=True) # 300 * 2048


        self.V = nn.Parameter(nn.init.normal_(torch.empty(self.feat_channel, self.attritube_num)), requires_grad=True)

        # loss
        self.Reg_loss = nn.MSELoss()
        self.CLS_loss = nn.CrossEntropyLoss()



    def conv_features(self, x):
        '''
        the feature input to prototype layer
        '''
        x = self.backbone(x)
        return x

    def base_module(self, x, seen_att):

        N, C, W, H = x.shape
        global_feat = F.avg_pool2d(x, kernel_size=(W, H))
        global_feat = global_feat.view(N, C)
        gs_feat = torch.einsum('bc,cd->bd', global_feat, self.V)

        gs_feat_norm = torch.norm(gs_feat, p=2, dim = 1).unsqueeze(1).expand_as(gs_feat)
        gs_feat_normalized = gs_feat.div(gs_feat_norm + 1e-5)

        temp_norm = torch.norm(seen_att, p=2, dim=1).unsqueeze(1).expand_as(seen_att)
        seen_att_normalized = seen_att.div(temp_norm + 1e-5)

        cos_dist = torch.einsum('bd,nd->bn', gs_feat_normalized, seen_att_normalized)
        score = cos_dist * self.scale

        return score


    def attentionModule(self, x):

        N, C, W, H = x.shape
        x = x.reshape(N, C, W * H)  # N, V, r=WH

        query = torch.einsum('lw,wv->lv', self.w2v_att, self.W) # L * V

        atten_map = torch.einsum('lv,bvr->blr', query, x) # batch * L * r

        atten_map = F.softmax(atten_map, -1)

        x = x.transpose(2, 1) # batch, WH=r, V
        part_feat = torch.einsum('blr,brv->blv', atten_map, x) # batch * L * V
        part_feat = F.normalize(part_feat, dim=-1)

        atten_map = atten_map.view(N, -1, W, H)
        atten_attr = F.max_pool2d(atten_map, kernel_size=(W,H))
        atten_attr = atten_attr.view(N, -1)

        return part_feat, atten_map, atten_attr, query

    def attr_decorrelation(self, query):

        loss_sum = 0

        for key in self.attr_group:
            group = self.attr_group[key]
            proto_each_group = query[group]  # g1 * v
            channel_l2_norm = torch.norm(proto_each_group, p=2, dim=0)
            loss_sum += channel_l2_norm.mean()

        loss_sum = loss_sum.float()/len(self.attr_group)

        return loss_sum

    def CPT(self, atten_map):
        """

        :param atten_map: N, L, W, H
        :return:
        """

        N, L, W, H = atten_map.shape
        xp = torch.tensor(list(range(W))).long().unsqueeze(1).to(self.device)
        yp = torch.tensor(list(range(H))).long().unsqueeze(0).to(self.device)

        xp = xp.repeat(1, H)
        yp = yp.repeat(W, 1)

        atten_map_t = atten_map.view(N, L, -1)
        value, idx = atten_map_t.max(dim=-1)

        tx = idx // H
        ty = idx - H * tx

        xp = xp.unsqueeze(0).unsqueeze(0)
        yp = yp.unsqueeze(0).unsqueeze(0)
        tx = tx.unsqueeze(-1).unsqueeze(-1)
        ty = ty.unsqueeze(-1).unsqueeze(-1)

        pos = (xp - tx) ** 2 + (yp - ty) ** 2

        loss = atten_map * pos

        loss = loss.reshape(N, -1).mean(-1)
        loss = loss.mean()

        return loss

    def forward(self, x, att=None, label=None, seen_att=None):

        feat = self.conv_features(x)  # N， 2048， 14， 14

        score = self.base_module(feat, seen_att)  # N, d
        if not self.training:
            return score

        part_feat, atten_map, atten_attr, query = self.attentionModule(feat)

        Lcls = self.CLS_loss(score, label)
        Lreg = self.Reg_loss(atten_attr, att)

        if self.attr_group is not None:
            Lad = self.attr_decorrelation(query)
        else:
            Lad = torch.tensor(0).float().to(self.device)

        Lcpt = self.CPT(atten_map)
        scale = self.scale.item()

        loss_dict = {
            'Reg_loss': Lreg,
            'Cls_loss': Lcls,
            'AD_loss': Lad,
            'CPT_loss': Lcpt,
            'scale': scale
        }

        return loss_dict

    def getAttention(self, x):
        feat = self.conv_features(x)
        part_feat, atten_map, atten_attr, query = self.attentionModule(feat)
        return atten_map



class EncoderFemnist(nn.Module):
    def __init__(self, code_length):
        super(EncoderFemnist, self).__init__()
        self.conv1 = nn.Conv2d(3, 10, kernel_size=3)
        self.conv2 = nn.Conv2d(10,20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(int(500), code_length)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, x.shape[1]*x.shape[2]*x.shape[3])
        z = F.relu(self.fc1(x))
        return z

class CNNFemnist(nn.Module):
    def __init__(self, args,code_length=50,num_classes = 62):
        super(CNNFemnist, self).__init__()
        self.code_length = code_length
        self.num_classes = num_classes
        self.feature_extractor = EncoderFemnist(self.code_length)
        self.classifier = nn.Sequential(nn.Dropout(0.2),
                                        nn.Linear(self.code_length, self.num_classes),
                                        nn.LogSoftmax(dim=1))
        self.proto_classifier = Proto_Classifier(self.code_length, num_classes=num_classes)
        self.scaling = torch.nn.Parameter(torch.tensor([1.0]))

    def forward(self, x):
        z = self.feature_extractor(x)
        p = self.classifier(z)
        return z, p


class EncoderAGG(nn.Module):
    def __init__(self, code_length):
        super(EncoderAGG, self).__init__()
        self.conv1 = nn.Conv2d(3, 10, kernel_size=3)
        self.conv2 = nn.Conv2d(10,20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(int(500), code_length)


    def forward(self, x,labels=None):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, x.shape[1]*x.shape[2]*x.shape[3])
        z = F.relu(self.fc1(x))
        return z

class AGGCNN(nn.Module):
    def __init__(self, args, code_length=50,num_classes = 62):
        super(AGGCNN, self).__init__()
        self.code_length = code_length
        self.num_classes = 20
        self.feature_extractor = EncoderAGG(self.code_length)
        self.classifier = nn.Sequential(nn.Dropout(0.2),
                                        nn.Linear(self.code_length, self.num_classes),
                                        nn.Sigmoid())
        self.proto_classifier = Proto_Classifier(self.code_length, num_classes=num_classes)
        self.scaling = torch.nn.Parameter(torch.tensor([1.0]))
        a = np.random.random(size=(code_length, num_classes))
        s, _, _ = np.linalg.svd(a)

        # 替代scaling
        self.scaling = torch.nn.Parameter(torch.tensor(s))

    def forward(self, x):
        z = self.feature_extractor(x)
        p = self.classifier(z)
        return z,p



class ResNetCifar10(nn.Module):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(ResNetCifar10, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        # x = self.fc(x)

        return x

    def forward(self, x):
        return self._forward_impl(x)


def ResNet18_cifar10(**kwargs):
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return ResNetCifar10(BasicBlock, [2, 2, 2, 2], **kwargs)



# class ResNet18(nn.Module):
#     def __init__(self, args, code_length=64, num_classes = 10):
#         super(ResNet18, self).__init__()
#         self.code_length = code_length
#         self.num_classes = num_classes
#
#         # self.feature_extractor = models.resnet18(num_classes=self.code_length)
#         self.feature_extractor = ResNet18_cifar10()
#         self.classifier = nn.Sequential(
#                                 nn.Linear(512, self.num_classes))
#     def forward(self,x):
#         z = self.feature_extractor(x)
#         p = self.classifier(z)
#         return z, p


class MultiHeadFeatureRotation(nn.Module):
    def __init__(self, feature_dim=512, num_heads=8):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_heads = num_heads
        self.head_dim = feature_dim // num_heads

        assert feature_dim % num_heads == 0, "feature_dim must be divisible by num_heads"

        # 每个头有自己的旋转参数
        self.rotation_angles = nn.Parameter(
            torch.randn(num_heads, self.head_dim // 2) * 0.1
        )

    def forward(self, x):
        """
        x: [batch_size, feature_dim]
        """
        batch_size = x.shape[0]

        # 重塑为多头形式 [batch_size, num_heads, head_dim]
        x_multihead = x.view(batch_size, self.num_heads, self.head_dim)

        # 应用每个头的旋转
        rotated_heads = []
        for head_idx in range(self.num_heads):
            head_x = x_multihead[:, head_idx, :]  # [batch_size, head_dim]

            # 重塑为复数形式
            head_reshaped = head_x.view(batch_size, self.head_dim // 2, 2)

            # 获取该头的旋转角度
            angles = self.rotation_angles[head_idx]  # [head_dim//2]
            cos_vals = torch.cos(angles).unsqueeze(0).unsqueeze(-1)  # [1, head_dim//2, 1]
            sin_vals = torch.sin(angles).unsqueeze(0).unsqueeze(-1)  # [1, head_dim//2, 1]

            # 应用旋转
            head_rotated = torch.stack([
                head_reshaped[..., 0] * cos_vals - head_reshaped[..., 1] * sin_vals,
                head_reshaped[..., 0] * sin_vals + head_reshaped[..., 1] * cos_vals
            ], dim=-1)

            rotated_heads.append(head_rotated.view(batch_size, self.head_dim))

        # 合并所有头
        output = torch.cat(rotated_heads, dim=1)
        return output


class ResNet18(nn.Module):
    def __init__(self, args=None, code_length=64, num_classes=10):
        super(ResNet18, self).__init__()
        self.code_length = code_length
        self.num_classes = num_classes
        # model = models.resnet18(weights='IMAGENET1K_V1')  # 域泛化用预训练模型
        model = models.resnet18()  # 域泛化用预训练模型
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, num_classes)
        modules = list(model.children())[:-1]  # 移除最后的全连接层
        self.feature_extractor = nn.Sequential(*modules)
        # self.feature_extractor = self.model
        print(num_features)
        self.classifier = nn.Sequential(
            nn.Linear(num_features, self.num_classes))
        # self.domain_classifier = DomainAdversarial(self.code_length, num_domains=num_classes)
        self.linear_proto = nn.Linear(num_features, code_length)

        self.proto_classifier = Proto_Classifier(code_length, num_classes=num_classes)

        self.proto_classifier2 = Proto_Classifier(code_length, num_classes=num_classes)
        self.linear_proto2 = nn.Linear(num_features, code_length)

        self.scaling = torch.nn.Parameter(torch.tensor([1.0]))
        self.scaling2 = torch.nn.Parameter(torch.tensor([1.0]))
        # FedBABU
        temp = self.classifier[0].state_dict()['weight']
        self.prototype = nn.Parameter(temp)
        a = np.random.random(size=(code_length, num_classes))
        s, _, _ = np.linalg.svd(a)
        self.rotation = nn.Linear(code_length, code_length, bias=False)

        self.etf = ETFHead(code_length, num_classes)
        self.hug_classifier = HUGClassifier(num_features, num_classes)
        # 替代scaling
        # self.scaling = torch.nn.Parameter(torch.tensor(s))

        # self.rotation = MultiHeadFeatureRotation(feature_dim=code_length)
        # print(self.classifier[0].weight.data)

    def forward(self, x):
        z = self.feature_extractor(x)
        z = torch.flatten(z, 1)
        p = self.classifier(z)
        return z, p



class ResNet181(nn.Module):
    def __init__(self, args,code_length=64,num_classes = 10):
        super(ResNet181, self).__init__()
        self.code_length = code_length
        self.num_classes = num_classes
        # 输入为32*32
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, self.code_length)

        )
        self.classifier = nn.Sequential(
            nn.Linear(self.code_length, self.num_classes)
        )
    def forward(self, x):
        z = self.feature_extractor(x)
        p = self.classifier(z)
        return z, p

class ShuffLeNet(nn.Module):
    def __init__(self, args,code_length=64,num_classes = 10):
        super(ShuffLeNet, self).__init__()
        self.code_length = code_length
        self.num_classes = num_classes
        self.feature_extractor = models.shufflenet_v2_x1_0(num_classes=self.code_length)
        self.classifier =  nn.Sequential(
                                nn.Linear(self.code_length, self.num_classes))
    def forward(self,x):
        z = self.feature_extractor(x)
        p = self.classifier(z)
        return z,p

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


def ResNet18_cifar10(**kwargs):
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return ResNetCifar10(BasicBlock, [2, 2, 2, 2], **kwargs)



def ResNet50_cifar10(**kwargs):
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return ResNetCifar10(Bottleneck, [3, 4, 6, 3], **kwargs)