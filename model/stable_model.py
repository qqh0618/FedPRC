"""
2025/5/7 0:10
while True:
    leanring
本文件由my_ywj首次创建编写
"""
import random

import torch
from torch.nn import functional as F
import torch.nn as nn
import random
from sklearn.cluster import KMeans
import numpy as np

#Style Sharing
def style_insert(x, style=None, num_blocks=None):

    if style is None:
        return x

    else:
        if random.random() > 0.5:
            return x

        else:
            share_number = 4 # half of the batch
            B = x.size(0)
            mu = x.mean(dim=[2, 3], keepdim=True)
            var = x.var(dim=[2, 3], keepdim=True)
            sig = (var + 1e-6).sqrt()
            mu, sig = mu.detach(), sig.detach()
            x_normed = (x - mu) / sig

            if num_blocks == 8: # ResNet-18
                chan = 128
            else: # ResNet-50
                chan = 512

            stat = torch.cat((mu, sig), dim=1)
            stat = stat.view(B, chan)
            stat = kmeans_plus(stat, x.device, share_number)

            mu_sel = stat[:,:int(chan/2)].view(x.shape[0] - share_number, int(chan/2), 1, 1)
            sig_sel = stat[:,int(chan/2):].view(x.shape[0] - share_number, int(chan/2), 1, 1)
            style1 = style[0].to(x.device)

            try:
                style2 = style[1].to(x.device)

            except:
                style2 = 0

            mu_mean_other = style1[:int(chan/2)]
            sig_mean_other = style1[int(chan/2):]

            if torch.sum(style2) == 0:
                new_mu = mu_mean_other.repeat(share_number, 1)  # B,C,1,1
                new_sig = sig_mean_other.repeat(share_number, 1)

            else:
                mu_sig_other = style2[:int(chan/2)]
                sig_sig_other = style2[int(chan/2):]

                mu_sig_other = mu_sig_other.view(1, int(chan/2))
                sig_sig_other = sig_sig_other.view(1, int(chan/2))

                mu_sig_other = mu_sig_other.repeat(share_number, 1)  # B,C,1,1
                sig_sig_other = sig_sig_other.repeat(share_number, 1)

                new_mu = torch.randn_like(mu_sig_other) * mu_sig_other + mu_mean_other
                new_sig = torch.randn_like(sig_sig_other) * sig_sig_other + sig_mean_other

            new_sig = new_sig * (new_sig>=0)

            mu_new = torch.vstack((mu_sel, new_mu.view(share_number, int(chan/2), 1, 1)))
            sig_new = torch.vstack((sig_sel, new_sig.view(share_number, int(chan/2), 1, 1)))

            return x_normed * sig_new + mu_new


def kmeans_plus(x, device, share_number):
    x = x.data.cpu().numpy()

    model = KMeans(n_clusters=x.shape[0] - share_number, init='k-means++', max_iter=1, n_init=1)
    model.fit(x)

    center = model.cluster_centers_
    stat = torch.tensor(center).to(device)

    return stat




def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False
    )


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes,
            planes,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(
            planes, planes * self.expansion, kernel_size=1, bias=False
        )
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

# Attention-based Feature Highlighter
class AFH(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dk, dv, Nh, relative):
        super(AFH, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dk = dk
        self.dv = dv
        self.Nh = Nh
        self.relative = relative

        self.qkv_conv = nn.Conv2d(self.in_channels, 2 * self.dk + self.dv, kernel_size=1)
        self.attn_out = nn.Conv2d(1, 1, 1)

    def forward(self, x, label, supplemental_samples=None, tmp_=False, layer=2):

        batch, _, height, width = x.size()

        if self.training:
            if supplemental_samples is not None:
                flat_q_m, flat_k_m, flat_v_m = supplemental_samples[2][0]

            unique_label = torch.unique(label)
            unique_label = unique_label.cpu().numpy()
            ind_per_label = {}
            num_per_label_dic = {}
            for i in unique_label:
                ind_tmp = (label==i).nonzero().view(-1).cpu().numpy()
                ind_per_label[i] = ind_tmp
                num_per_label_dic[i] = ind_tmp.shape[0]

            pair_index = []
            pair_index_1s = []
            pair_index_1s_class = []

            for j in range(batch):

                if num_per_label_dic[label[j].item()] == 1:
                    add_ind = ind_per_label[label[j].item()][0]
                    pair_index_1s.append(add_ind)
                    pair_index_1s_class.append(label[j].item())

                else:
                    tmp = (ind_per_label[label[j].item()]==j).nonzero()[0]
                    add_ind = ind_per_label[label[j].item()][(tmp+1) % num_per_label_dic[label[j].item()]][0]

                pair_index.append(add_ind)
        else:
            pair_index = [k for k in range(batch)]

        flat_q, flat_k, flat_v, q, k, v = self.compute_flat_qkv(x, self.dk, self.dv, self.Nh)

        if tmp_ is True:
            return x, [flat_q, flat_k, flat_v]

        if self.training:
            if len(pair_index_1s) != 0:
                flat_q[pair_index_1s] = flat_q_m[pair_index_1s_class]

        flat_q = flat_q / flat_q.norm(dim=2)[:,:,None,:]
        flat_k = flat_k / flat_k.norm(dim=2)[:,:,None,:]

        logits = torch.matmul(((flat_q[pair_index] + flat_q)/2).transpose(2, 3), flat_k)

        if self.relative:
            h_rel_logits, w_rel_logits = self.relative_logits(q)
            logits += h_rel_logits
            logits += w_rel_logits

        attn_out = torch.reshape(logits, (batch, self.Nh, height * width, height, width))
        attn_out = self.combine_heads_2d(attn_out)  # (batch, out_channels, height, width)

        attn_out = torch.mean(attn_out, dim=[1], keepdim=True)

        out = F.softmax((attn_out).reshape(attn_out.size(0), attn_out.size(1), -1), 2).view_as(attn_out)

        return torch.cat((x/49, x * out), dim=1)


    def compute_flat_qkv(self, x, dk, dv, Nh):
        N, _, H, W = x.size()
        qkv = self.qkv_conv(x)
        q, k, v = torch.split(qkv, [dk, dk, dv], dim=1)
        q = self.split_heads_2d(q, Nh)
        k = self.split_heads_2d(k, Nh)
        v = self.split_heads_2d(v, Nh)

        dkh = dk // Nh
        q = q * (dkh ** -0.5)
        flat_q = torch.reshape(q, (N, Nh, dk // Nh, H * W)) # flatten HW
        flat_k = torch.reshape(k, (N, Nh, dk // Nh, H * W))
        flat_v = torch.reshape(v, (N, Nh, dv // Nh, H * W))
        return flat_q, flat_k, flat_v, q, k, v

    def split_heads_2d(self, x, Nh):
        batch, channels, height, width = x.size()
        ret_shape = (batch, Nh, channels // Nh, height, width) # split head
        split = torch.reshape(x, ret_shape)
        return split

    def combine_heads_2d(self, x):
        batch, Nh, dv, H, W = x.size()
        ret_shape = (batch, Nh * dv, H, W)
        return torch.reshape(x, ret_shape)

    def relative_logits(self, q):
        B, Nh, dk, H, W = q.size()
        q = torch.transpose(q, 2, 4).transpose(2, 3)

        key_rel_w = nn.Parameter(torch.randn((2 * W - 1, dk), requires_grad=True)).to(q.device)
        rel_logits_w = self.relative_logits_1d(q, key_rel_w, H, W, Nh, "w")

        key_rel_h = nn.Parameter(torch.randn((2 * H - 1, dk), requires_grad=True)).to(q.device)
        rel_logits_h = self.relative_logits_1d(torch.transpose(q, 2, 3), key_rel_h, W, H, Nh, "h")

        return rel_logits_h, rel_logits_w

    def relative_logits_1d(self, q, rel_k, H, W, Nh, case):
        rel_logits = torch.einsum('bhxyd,md->bhxym', q, rel_k)
        rel_logits = torch.reshape(rel_logits, (-1, Nh * H, W, 2 * W - 1))
        rel_logits = self.rel_to_abs(rel_logits)

        rel_logits = torch.reshape(rel_logits, (-1, Nh, H, W, W))
        rel_logits = torch.unsqueeze(rel_logits, dim=3)
        rel_logits = rel_logits.repeat((1, 1, 1, H, 1, 1))

        if case == "w":
            rel_logits = torch.transpose(rel_logits, 3, 4)
        elif case == "h":
            rel_logits = torch.transpose(rel_logits, 2, 4).transpose(4, 5).transpose(3, 5)
        rel_logits = torch.reshape(rel_logits, (-1, Nh, H * W, H * W))
        return rel_logits

    def rel_to_abs(self, x):
        B, Nh, L, _ = x.size()

        col_pad = torch.zeros((B, Nh, L, 1)).to(x.device)
        x = torch.cat((x, col_pad), dim=3)

        flat_x = torch.reshape(x, (B, Nh, L * 2 * L))
        flat_pad = torch.zeros((B, Nh, L - 1)).to(x.device)
        flat_x_padded = torch.cat((flat_x, flat_pad), dim=2)

        final_x = torch.reshape(flat_x_padded, (B, Nh, L + 1, 2 * L - 1))
        final_x = final_x[:, :, :L, L - 1:]
        return final_x



class ResNet(nn.Module):

    def __init__(
        self,
        block,
        layers,
        ms_class=None,
        ms_layers=[],
        ms_p=0.5,
        ms_a=0.1,
        **kwargs
    ):
        self.inplanes = 64
        super().__init__()

        # backbone network
        self.conv1 = nn.Conv2d(
            3, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.bn1 = nn.BatchNorm2d(64)

        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)

        self._out_features = 512 * block.expansion
        self.num_blocks = sum(layers)

        ######################################################################################################
        self.cad3 = AFH(in_channels=512, out_channels=30, kernel_size=1, dk=30, dv=30, Nh=1,
                        relative=False)  # 64 128 512
        self.style = style_insert
        self.OMA = OMA(p=0.5, alpha=0.1)
        self.ms_layers = ["layer1", "layer2", "layer3"]
        ######################################################################################################

        self.ms_layers = ms_layers
        self.vis = 0
        self._init_params()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def _init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode="fan_out", nonlinearity="relu"
                )
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def featuremaps(self, x, label, supplemental_samples, style, StableFDG_param):

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)

        # Style sharing
        x = self.style(x, style, self.num_blocks)

        self.OMA.indicator = 0

        #Style exploration
        if "layer1" in self.ms_layers:
            x, label = self.OMA(x, label, supplemental_samples, param=StableFDG_param, layer_mix=1)

        x = self.layer2(x)

        # Style exploration
        if "layer2" in self.ms_layers:
            x, label = self.OMA(x, label, supplemental_samples, param=StableFDG_param, layer_mix=2)

        x = self.layer3(x)

        # Style exploration
        if "layer3" in self.ms_layers:
            x, label = self.OMA(x, label, supplemental_samples, param=StableFDG_param, layer_mix=3)

        x = self.layer4(x)

        # 3-3 Attention module
        x = self.cad3(x, label, supplemental_samples=supplemental_samples, layer=4, tmp_=False)

        return x, label

    def forward(self, x, label, supplemental_samples=None, style=None, StableFDG_param=None):
        f, label = self.featuremaps(x, label, supplemental_samples, style, StableFDG_param)

        # Normal
        # v = self.global_avgpool(f)

        # 3-3 Attention Module
        v = torch.sum(f.view(f.size(0), f.size(1), -1), dim=2)

        return v.view(v.size(0), -1), label


    def featuremaps_cb(self, x, label):
        feat = []
        feat_attn = []
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)

        feat.append(x)
        x = self.layer2(x)

        feat.append(x)
        x = self.layer3(x)
        feat.append(x)

        x = self.layer4(x)

        # Feature extraction
        x, flat_feat = self.cad3(x, label, tmp_=True)
        feat_attn.append(flat_feat)

        return x, feat, feat_attn

    def forward_cb(self, x, label):
        f, feat, flat_feat = self.featuremaps_cb(x, label)
        # v = self.global_avgpool(f)

        # return v.view(v.size(0), -1), feat, flat_feat
        return feat, flat_feat





"""
Residual network configurations:
--
resnet18: block=BasicBlock, layers=[2, 2, 2, 2]
resnet34: block=BasicBlock, layers=[3, 4, 6, 3]
resnet50: block=Bottleneck, layers=[3, 4, 6, 3]
resnet101: block=Bottleneck, layers=[3, 4, 23, 3]
resnet152: block=Bottleneck, layers=[3, 8, 36, 3]
"""



#Style Exploration
class OMA(nn.Module):

    def __init__(self, p=0.5, alpha=0.1, eps=1e-6, mix="random"):

        super().__init__()
        self.p = p
        self.beta = torch.distributions.Beta(alpha, alpha)
        self.beta_mixup = torch.distributions.Beta(0.1, 0.1)
        self.eps = eps
        self.alpha = alpha
        self.mix = mix
        self._activated = True
        self.indicator = 0
        self.when = 0

    def __repr__(self):
        return (
            f"MixStyle(p={self.p}, alpha={self.alpha}, eps={self.eps}, mix={self.mix})"
        )

    def set_activation_status(self, status=True):
        self._activated = status

    def update_mix_method(self, mix="random"):
        self.mix = mix

    def to_one_hot(self, inp, num_classes):
        y_onehot = torch.FloatTensor(inp.size(0), num_classes)
        y_onehot.zero_()
        y_onehot.scatter_(1, inp.unsqueeze(1).cpu(), 1)
        return y_onehot.to(inp.device)

    def forward(self, x, label, supplemental_samples=None, param=None, layer_mix=1):

        if not self.training or not self._activated:
            return x, label

        if random.random() > 0.5:
            return x, label

        self.indicator += 1

        exploration_level = 3.0
        oversampling_size = 32

        if self.indicator == 1:
            if supplemental_samples is not None:
                feat_sup = supplemental_samples[0][layer_mix-1]
                label_sup = supplemental_samples[1]

            B_ori = x.size(0)

            unique_label = torch.unique(label)
            unique_label = unique_label.cpu().numpy()
            ind_per_label = {}
            num_per_label_dic = {}
            for i in unique_label:
                ind_tmp = (label==i).nonzero().view(-1).cpu().numpy()
                ind_per_label[i] = ind_tmp
                num_per_label_dic[i] = ind_tmp.shape[0]

            class_sup = [k for k, v in num_per_label_dic.items() if v == 1]

            # Only for Office-Home dataset
            if len(class_sup) != 0:
                if len(class_sup) <= oversampling_size:
                    supplemental_index = [(label_sup==i).nonzero()[0].item() for i in class_sup]
                    x = torch.vstack([x, feat_sup[supplemental_index]])
                    label = torch.hstack([label, label_sup[supplemental_index]])

                else:
                    supplemental_index = [(label_sup==i).nonzero()[0].item() for i in class_sup]
                    sample_ind = np.random.choice(supplemental_index, oversampling_size, replace=False).tolist()
                    x = torch.vstack([x, feat_sup[sample_ind]])
                    label = torch.hstack([label, label_sup[sample_ind]])

                for i in unique_label:
                    ind_tmp = (label == i).nonzero().view(-1).cpu().numpy()
                    ind_per_label[i] = ind_tmp
                    num_per_label_dic[i] = ind_tmp.shape[0]

            num_per_label = np.array([i for i in num_per_label_dic.values()])
            batch_limit = oversampling_size - len(class_sup)
            index_class = []
            add_index = []

            if batch_limit > 0:

                while True:
                    unique_num_samples = np.unique(num_per_label)
                    k = unique_num_samples[0]
                    candi = (num_per_label == k).nonzero()[0]
                    if candi.shape[0] >= batch_limit:
                        index_a = np.random.choice(candi, batch_limit, replace=False).tolist()
                        index_class += list(unique_label[index_a])
                        break
                    else:
                        index_a = np.random.choice(candi, candi.shape[0], replace=False).tolist()
                        index_class += list(unique_label[index_a])
                        num_per_label += (num_per_label == k)
                        batch_limit -= candi.shape[0]

                for i in index_class:
                    add_index  += np.random.choice(ind_per_label[i], 1, replace=False).tolist()

            mu = x.mean(dim=[2, 3], keepdim=True)
            var = x.var(dim=[2, 3], keepdim=True)
            sig = (var + self.eps).sqrt()

            x_normed = (x - mu) / sig
            x_normed[B_ori:] = x_normed[B_ori:]
            x_norm_oversampled = x_normed[add_index]

            x_normed = torch.vstack([x_normed, x_norm_oversampled])
            label = torch.hstack([label, label[add_index]])

            # Style Exploration
            mu_mean = mu.mean(dim=[0], keepdim=True)
            sig_mean = sig.mean(dim=[0], keepdim=True)

            mu_mix = (mu[add_index] - mu_mean) * exploration_level + mu[add_index]
            sig_mix = (sig[add_index] - sig_mean) * exploration_level + sig[add_index]
            # sig_mix = (sig_mix >= 0) * sig_mix

            mu_extended = torch.vstack([mu,  mu_mix])
            sig_extended = torch.vstack([sig,  sig_mix])

        else:
            mu = x.mean(dim=[2, 3], keepdim=True)
            var = x.var(dim=[2, 3], keepdim=True)
            sig = (var + self.eps).sqrt()

            x_normed = (x - mu) / sig
            mu_extended = mu
            sig_extended = sig

        B = x_normed.size(0)

        # Mixing the Styles
        mu_extended, sig_extended = mu_extended.detach(), sig_extended.detach()
        lmda = self.beta.sample((B, 1, 1, 1))
        lmda = lmda.to(x.device)

        perm = torch.randperm(B)
        mu2, sig2 = mu_extended[perm], sig_extended[perm]
        mu_mix = mu_extended * lmda + mu2 * (1 - lmda)
        sig_mix = sig_extended * lmda + sig2 * (1 - lmda)

        return x_normed*sig_mix + mu_mix, label


def resnet18_OMA_ms_l123(pretrained=True, **kwargs):
    model = ResNet(
        block=BasicBlock,
        layers=[2, 2, 2, 2],
        ms_class=(OMA, ),
        ms_layers=["layer1", "layer2", "layer3"],
    )

    return model

############################################################################################################
def resnet18(pretrained=True, **kwargs):
    model = ResNet(block=BasicBlock, layers=[2, 2, 2, 2])


    return model




class SimpleNet(nn.Module):
    """A simple neural network composed of a CNN backbone
    and optionally a head such as mlp for classification.
    """

    def __init__(self, num_classes, **kwargs):
        super().__init__()
        self.backbone = resnet18_OMA_ms_l123()

        self.classifier = None
        if num_classes > 0:
            # self.classifier = nn.Linear(fdim, num_classes)

            # For Attention Module
            self.classifier = nn.Linear(512*2, num_classes)


    def forward(self, x, label, supplemental_samples=None, style=None, StableFDG_param=None):
        f, label = self.backbone(x, label, supplemental_samples, style)
        y = self.classifier(f)


        return y, label

    def forward_cb(self, x, label=None, return_feature=False):
        feat, flat_feat = self.backbone.forward_cb(x, label)

        return feat, flat_feat



#######################################################################################