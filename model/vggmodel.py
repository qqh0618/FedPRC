import math
import torch
import torch.nn as nn
import torch.nn.init as init
# from models.common import QuantOps as Q
__all__ = [
    'VGG', 'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn',
    'vgg19_bn', 'vgg19',
]


class CustomDropout(nn.Module):
    def __init__(self, p=0.5):
        super(CustomDropout, self).__init__()
        self.p = p

    def forward(self, x):
        if self.training:
            mask = (torch.rand(x.size()) > self.p).float().to(x.device)
            return x * mask / (1.0 - self.p)
        else:
            return x

class VGG(nn.Module):
    '''
    VGG model
    '''
    def __init__(self, features,num_classes):
        super(VGG, self).__init__()
        self.features = features
        # 'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
        # self.features = nn.Sequential(
        #     nn.Conv2d(3, 64, 3, padding=1),
        #     nn.MaxPool2d(2, 2),
        #     nn.Conv2d(64, 128, 3, padding=1),
        #     nn.MaxPool2d(2, 2),
        #     nn.Conv2d(128, 256, 3, padding=1),
        #     nn.Conv2d(256, 256, 3, padding=1),
        #     nn.MaxPool2d(2, 2),
        #     nn.Conv2d(256, 512, 3, padding=1),
        #     nn.Conv2d(512, 512, 3, padding=1),
        #     nn.MaxPool2d(2, 2),
        #     nn.Conv2d(512, 512, 3, padding=1),
        #     nn.Conv2d(512, 512, 3, padding=1),
        #     nn.MaxPool2d(2, 2)
        # )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(512, 512),
            nn.ReLU(),
            # nn.Linear(512, 100),
        )
        
        # print(dim_sample)
        # self.line_squeeze = nn.Linear(512, int(512/4*dim_sample))
        # self.v_head = nn.Linear(int(512/4*dim_sample), num_classes)
        self.scaling_train = torch.nn.Parameter(torch.tensor(10.0))
        
        self.head = nn.Linear(512, num_classes)
         # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                m.bias.data.zero_()

    def forward(self, x, NS = None):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        feats = self.classifier(x)
        # out_mid = self.line_squeeze(feats)
        # v_out = self.v_head(out_mid)
        out = self.head(feats)
        return out
    
    def head_vim(self, x):
        x = self.head(x)
        return x
    
    def project_head(self, feats):
        x = self.line_squeeze(feats)
        x = self.v_head(x)
        return x
def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU()]
            in_channels = v
    return nn.Sequential(*layers)


cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M',
          512, 512, 512, 512, 'M'],
}


def vgg11(num_classes, dim_sample):
    """VGG 11-layer model (configuration "A")"""
    return VGG(make_layers(cfg['A']),num_classes)


def vgg11_bn():
    """VGG 11-layer model (configuration "A") with batch normalization"""
    return VGG(make_layers(cfg['A'], batch_norm=True))


def vgg13():
    """VGG 13-layer model (configuration "B")"""
    return VGG(make_layers(cfg['B']))


def vgg13_bn():
    """VGG 13-layer model (configuration "B") with batch normalization"""
    return VGG(make_layers(cfg['B'], batch_norm=True))


def vgg16():
    """VGG 16-layer model (configuration "D")"""
    return VGG(make_layers(cfg['D']))


def vgg16_bn():
    """VGG 16-layer model (configuration "D") with batch normalization"""
    return VGG(make_layers(cfg['D'], batch_norm=True))


def vgg19():
    """VGG 19-layer model (configuration "E")"""
    return VGG(make_layers(cfg['E']))


def vgg19_bn():
    """VGG 19-layer model (configuration 'E') with batch normalization"""
    return VGG(make_layers(cfg['E'], batch_norm=True))
