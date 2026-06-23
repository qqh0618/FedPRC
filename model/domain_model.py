"""
2025/5/3 10:43
while True:
    leanring
本文件由my_ywj首次创建编写
"""
import copy

import torch
from torch import nn
import torch.nn.functional as F
import torchvision.models as models
import numpy as np

from model.densent import densenet121
from model.efficent import efficientnet_b0
from model.mobilenet import mobilenet_v3_small
from model.models import ResNet18_cifar10
from model.resnet import resnet18
from model.shufflenet import shufflenet_v2_x0_5
import clip
from contextlib import contextmanager
from six import add_metaclass

from model.vggmodel import vgg11
class EncoderFemnist(nn.Module):
    def __init__(self, code_length):
        super(EncoderFemnist, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=3)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(int(320), code_length)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, x.shape[1] * x.shape[2] * x.shape[3])
        z = F.relu(self.fc1(x))
        return z


class CNNFemnist(nn.Module):
    def __init__(self, args, code_length=50, num_classes=62):
        super(CNNFemnist, self).__init__()
        self.code_length = code_length
        self.num_classes = num_classes
        self.feature_extractor = EncoderFemnist(self.code_length)
        self.classifier = nn.Sequential(nn.Dropout(0.2),
                                        nn.Linear(self.code_length, self.num_classes),
                                        nn.LogSoftmax(dim=1))

    def forward(self, x):
        z = self.feature_extractor(x)
        p = self.classifier(z)
        return z, p


class GradientReversalFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambda_ * grad_output, None


def grad_reverse(x, lambda_=1.0):
    return GradientReversalFunction.apply(x, lambda_)


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



class DomainAdversarial(nn.Module):
    def __init__(self, feature_dim, num_domains):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.ReLU(),
            nn.Linear(128, num_domains)
        )

    def forward(self, features, lambda_):
        rev = grad_reverse(features, lambda_)
        return self.classifier(rev)


# Initialization function for modules
class PatchModules(type):
    def __call__(cls, *args, **kwargs):
        net = type.__call__(cls, *args, **kwargs)
        w_modules_names = []

        for m in net.modules():
            for n, p in m.named_parameters(recurse=False):
                if p is not None:
                    w_modules_names.append((m, n))

        net._weights_module_names = tuple(w_modules_names)
        ws = tuple(m._parameters[n].detach() for m, n in w_modules_names)

        net._weights_numels = tuple(w.numel() for w in ws)
        net._weights_shapes = tuple(w.shape for w in ws)
        with torch.no_grad():
            flat_w = torch.cat([w.reshape(-1) for w in ws], 0)

        for m, n in net._weights_module_names:
            delattr(m, n)
            m.register_buffer(n, None)

        net.register_parameter('flat_w', nn.Parameter(flat_w, requires_grad=True))

        return net


@add_metaclass(PatchModules)
class ReparamModule(nn.Module):
    def _apply(self, *args, **kwargs):
        rv = super(ReparamModule, self)._apply(*args, **kwargs)
        return rv

    def get_param(self, clone=False):
        state = self.state_dict()
        if clone:
            return {k: v.clone() for k, v in state.items()}
        return state

    def load_param(self, param_dict):
        self.load_state_dict(param_dict, strict=False)
        for name, module in self.named_modules():
            if isinstance(module, nn.BatchNorm2d):
                module.running_mean = param_dict[name + '.running_mean']
                module.running_var = param_dict[name + '.running_var']

    @contextmanager
    def unflatten_weight(self, flat_w):
        ws = (t.view(s) for (t, s) in zip(flat_w.split(self._weights_numels), self._weights_shapes))
        for (m, n), w in zip(self._weights_module_names, ws):
            setattr(m, n, w.to(self.flat_w.device))
        yield
        for m, n in self._weights_module_names:
            setattr(m, n, None)

    def reshape_flat_weights(self, flat_w):

        reshaped_weights = {
            f"{m}.{n}": t.view(s)
            for (m, n), t, s in
            zip(self._weights_module_names, flat_w.split(self._weights_numels), self._weights_shapes)
        }
        return reshaped_weights

    def get_head_weights(self, flat_w):

        reshaped_weights = self.reshape_flat_weights(flat_w)
        linear_weights = [
            weight.view(-1)
            for name, weight in reshaped_weights.items()
            if 'Linear' in name
        ]
        if linear_weights:
            return torch.cat(linear_weights)
        else:
            return torch.tensor([])

    def get_body_weights(self, flat_w):

        reshaped_weights = self.reshape_flat_weights(flat_w)
        non_linear_weights = [
            weight.view(-1)
            for name, weight in reshaped_weights.items()
            if 'Linear' not in name
        ]
        if non_linear_weights:
            return torch.cat(non_linear_weights)
        else:
            return torch.tensor([])

    def forward_with_param(self, inp, new_w, *args, **kwargs):
        with self.unflatten_weight(new_w):
            return nn.Module.__call__(self, inp, *args, **kwargs)

    def load_state_dict(self, state_dict, strict=True):
        if 'flat_w' in state_dict:
            self.flat_w.data = state_dict['flat_w'].detach().clone().requires_grad_(True).to(self.flat_w.device)

            state_dict = {k: v for k, v in state_dict.items() if k != 'flat_w'}
            super(ReparamModule, self).load_state_dict(state_dict, strict)

            ws = (t.view(s) for (t, s) in zip(self.flat_w.split(self._weights_numels), self._weights_shapes))
            for (m, n), w in zip(self._weights_module_names, ws):
                setattr(m, n, w.to(self.flat_w.device))
        else:
            super(ReparamModule, self).load_state_dict(state_dict, strict)

    def __call__(self, inp, *args, **kwargs):
        return self.forward_with_param(inp, self.flat_w, *args, **kwargs)

# class ResNet18_domain(nn.Module):
#     def __init__(self, args=None, code_length=64, num_classes=10):
#         super(ResNet18_domain, self).__init__()
#         self.code_length = code_length
#         self.num_classes = num_classes
#         model = models.resnet18(weights='IMAGENET1K_V1')  # 域泛化用预训练模型
#         num_features = model.fc.in_features
#         model.fc = nn.Linear(num_features, num_classes)
#         modules = list(model.children())[:-1]  # 移除最后的全连接层
#         self.feature_extractor = nn.Sequential(*modules)
#         # self.feature_extractor = self.model
#         print(num_features)
#         self.classifier = nn.Sequential(
#             nn.Linear(num_features, self.num_classes))
#         # self.domain_classifier = DomainAdversarial(self.code_length, num_domains=num_classes)
#         self.proto_classifier = Proto_Classifier(num_features, num_classes=num_classes)
#
#         self.scaling = torch.nn.Parameter(torch.tensor([1.0]))
#         # FedBABU
#         temp = self.classifier[0].state_dict()['weight']
#         self.prototype = nn.Parameter(temp)
#
#
#         # print(self.classifier[0].weight.data)
#
#     def forward(self, x):
#         z = self.feature_extractor(x)
#         z = torch.flatten(z, 1)
#         p = self.classifier(z)
#         return z, p

# 域泛化ipm使用
class ResNet18_domain(nn.Module):
    def __init__(self, args=None, code_length=64, num_classes=10):
        super(ResNet18_domain, self).__init__()
        self.code_length = code_length
        self.num_classes = num_classes
        model = models.resnet18(weights='IMAGENET1K_V1')  # 域泛化用预训练模型
        # model = models.resnet18()
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

        self.scaling = torch.nn.Parameter(torch.tensor([1.0]))
        # FedBABU
        temp = self.classifier[0].state_dict()['weight']
        self.prototype = nn.Parameter(temp)


        # print(self.classifier[0].weight.data)

    def forward(self, x):
        z = self.feature_extractor(x)
        z = torch.flatten(z, 1)
        p = self.classifier(z)
        return z, p


class ResNet18_awa(ReparamModule):
    def __init__(self, args=None, code_length=64, num_classes=10):
        super(ResNet18_awa, self).__init__()
        self.code_length = code_length
        self.num_classes = num_classes
        model = models.resnet18(weights='IMAGENET1K_V1')  # 域泛化用预训练模型
        # model = models.resnet18()
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

        self.scaling = torch.nn.Parameter(torch.tensor([1.0]))
        # FedBABU
        temp = self.classifier[0].state_dict()['weight']
        self.prototype = nn.Parameter(temp)

        # print(self.classifier[0].weight.data)

    def forward(self, x):
        z = self.feature_extractor(x)
        z = torch.flatten(z, 1)
        p = self.classifier(z)
        return z, p


# 域倾斜tnnls使用
# class ResNet18_domain(nn.Module):
#     def __init__(self, args, code_length=64, num_classes=10):
#         super(ResNet18_domain, self).__init__()
#         self.code_length = code_length
#         self.num_classes = num_classes
#         self.feature_extractor = resnet18(num_classes=512)
#         self.classifier = nn.Sequential(
#             nn.Linear(512, self.num_classes))
#
#         self.domain_classifier = DomainAdversarial(self.code_length, num_domains=num_classes)
#         self.proto_classifier = Proto_Classifier(self.code_length, num_classes=num_classes)
#         self.scaling = torch.nn.Parameter(torch.tensor([1.0]))
#         # print(self.classifier[0].weight.data)
#
#     def forward(self, x):
#         z = self.feature_extractor(x)
#         p = self.classifier(z)
#         return z, p
#


class shufflenet(nn.Module):
    def __init__(self, args=None, code_length=64, num_classes=10):
        super(shufflenet, self).__init__()
        self.code_length = code_length
        self.num_classes = num_classes
        self.feature_extractor = shufflenet_v2_x0_5(num_classes=512)
        self.classifier = nn.Sequential(
            nn.Linear(512, self.num_classes))

        # self.domain_classifier = DomainAdversarial(self.code_length, num_domains=num_classes)
        self.proto_classifier = Proto_Classifier(self.code_length, num_classes=num_classes)
        self.scaling = torch.nn.Parameter(torch.tensor([1.0]))
        # print(self.classifier[0].weight.data)

    def forward(self, x):
        z = self.feature_extractor(x)
        p = self.classifier(z)
        return z, p


class efficinet(nn.Module):
    def __init__(self, args=None, code_length=64, num_classes=10):
        super(efficinet, self).__init__()
        self.code_length = code_length
        self.num_classes = num_classes
        self.feature_extractor = efficientnet_b0(num_classes=512)
        self.classifier = nn.Sequential(
            nn.Linear(512, self.num_classes))

        # self.domain_classifier = DomainAdversarial(self.code_length, num_domains=num_classes)
        self.proto_classifier = Proto_Classifier(self.code_length, num_classes=num_classes)
        self.scaling = torch.nn.Parameter(torch.tensor([1.0]))
        # print(self.classifier[0].weight.data)

    def forward(self, x):
        z = self.feature_extractor(x)
        p = self.classifier(z)
        return z, p



class mobilenet(nn.Module):
    def __init__(self, args=None, code_length=64, num_classes=10):
        super(mobilenet, self).__init__()
        self.code_length = code_length
        self.num_classes = num_classes
        self.feature_extractor = mobilenet_v3_small(num_classes=512)
        self.classifier = nn.Sequential(
            nn.Linear(512, self.num_classes))

        # self.domain_classifier = DomainAdversarial(self.code_length, num_domains=num_classes)
        self.proto_classifier = Proto_Classifier(self.code_length, num_classes=num_classes)
        self.scaling = torch.nn.Parameter(torch.tensor([1.0]))
        # print(self.classifier[0].weight.data)

    def forward(self, x):
        z = self.feature_extractor(x)
        p = self.classifier(z)
        return z, p


class vgg(nn.Module):
    def __init__(self, args=None, code_length=64, num_classes=10):
        super(vgg, self).__init__()
        self.code_length = code_length
        self.num_classes = num_classes
        self.feature_extractor = vgg11(num_classes=512)
        self.classifier = nn.Sequential(
            nn.Linear(512, self.num_classes))

        # self.domain_classifier = DomainAdversarial(self.code_length, num_domains=num_classes)
        self.proto_classifier = Proto_Classifier(self.code_length, num_classes=num_classes)
        self.scaling = torch.nn.Parameter(torch.tensor([1.0]))
        # print(self.classifier[0].weight.data)

    def forward(self, x):
        z = self.feature_extractor(x)
        p = self.classifier(z)
        return z, p


class densenet(nn.Module):
    def __init__(self, args=None, code_length=64, num_classes=10):
        super(densenet, self).__init__()
        self.code_length = code_length
        self.num_classes = num_classes
        self.feature_extractor = densenet121(num_classes=512)
        self.classifier = nn.Sequential(
            nn.Linear(512, self.num_classes))

        # self.domain_classifier = DomainAdversarial(self.code_length, num_domains=num_classes)
        self.proto_classifier = Proto_Classifier(self.code_length, num_classes=num_classes)
        self.scaling = torch.nn.Parameter(torch.tensor([1.0]))
        # print(self.classifier[0].weight.data)

    def forward(self, x):
        z = self.feature_extractor(x)
        p = self.classifier(z)
        return z, p


class ResNet181(nn.Module):
    def __init__(self, args, code_length=64, num_classes=10):
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
    def __init__(self, args, code_length=64, num_classes=10):
        super(ShuffLeNet, self).__init__()
        self.code_length = code_length
        self.num_classes = num_classes
        self.feature_extractor = models.shufflenet_v2_x1_0(num_classes=self.code_length)
        self.classifier = nn.Sequential(
            nn.Linear(self.code_length, self.num_classes))

    def forward(self, x):
        z = self.feature_extractor(x)
        p = self.classifier(z)
        return z, p




# StableFDG的注意力模块
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

    def forward(self, x, domain, label, supplemental_samples=None, tmp_=False, layer=2):

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

