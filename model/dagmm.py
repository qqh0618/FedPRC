"""
2025/10/26 16:19
while True:
    leanring
本文件由my_ywj首次创建编写
"""
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
#
#
# class DAGMM(nn.Module):
#     def __init__(self, n_gmm=2, code_len=1):
#         """Network for DAGMM (KDDCup99)"""
#         super(DAGMM, self).__init__()
#         # Encoder network
#         self.fc1 = nn.Linear(118, 60)
#         self.fc2 = nn.Linear(60, 30)
#         self.fc3 = nn.Linear(30, 10)
#         self.fc4 = nn.Linear(10, code_len)
#
#         # Decoder network
#         self.fc5 = nn.Linear(code_len, 10)
#         self.fc6 = nn.Linear(10, 30)
#         self.fc7 = nn.Linear(30, 60)
#         self.fc8 = nn.Linear(60, 118)
#
#         # Estimation network
#         self.fc9 = nn.Linear(code_len + 2, 10)
#         self.fc10 = nn.Linear(10, n_gmm)
#
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
#             elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
#                 nn.init.constant_(m.weight, 1)
#                 nn.init.constant_(m.bias, 0)
#
#     def encode(self, x):
#         h = torch.tanh(self.fc1(x))
#         h = torch.tanh(self.fc2(h))
#         h = torch.tanh(self.fc3(h))
#         return self.fc4(h)
#
#     def decode(self, x):
#         h = torch.tanh(self.fc5(x))
#         h = torch.tanh(self.fc6(h))
#         h = torch.tanh(self.fc7(h))
#         return self.fc8(h)
#
#     def estimate(self, z):
#         h = F.dropout(torch.tanh(self.fc9(z)), 0.5)
#         return F.softmax(self.fc10(h), dim=1)
#
#     def compute_reconstruction(self, x, x_hat):
#         relative_euclidean_distance = (x - x_hat).norm(2, dim=1) / x.norm(2, dim=1)
#         cosine_similarity = F.cosine_similarity(x, x_hat, dim=1)
#         return relative_euclidean_distance, cosine_similarity
#
#     def forward(self, x):
#         z_c = self.encode(x)
#         x_hat = self.decode(z_c)
#         rec_1, rec_2 = self.compute_reconstruction(x, x_hat)
#         z = torch.cat([z_c, rec_1.unsqueeze(-1), rec_2.unsqueeze(-1)], dim=1)
#         gamma = self.estimate(z)
#         return z_c, x_hat, z, gamma


"""
2025/10/26 16:19
while True:
    leanring
本文件由my_ywj首次创建编写
"""
import torch
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



class DAGMM(nn.Module):
    def __init__(self, args, n_gmm=2, code_length=10, img_size=32,num_classes=10):
        """Network for DAGMM (Image Data)"""
        super(DAGMM, self).__init__()
        self.img_size = img_size
        self.code_len = code_length

        # Encoder network (Convolutional)
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1, bias=False),
            nn.Tanh(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 3, padding=1, bias=False),
            nn.Tanh(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, padding=1, bias=False),
            nn.Tanh(),
            nn.MaxPool2d(2, 2)
        )

        # Bottleneck (Fully connected)
        self.en_fc = nn.Linear(128 * (img_size // 8) * (img_size // 8), code_length)

        self.de_fc = nn.Linear(code_length, 128 * (img_size // 8) * (img_size // 8))
        # Decoder network (Transposed Convolutional)
        self.decoder = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1, bias=False),
            nn.Tanh(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(64, 32, 3, padding=1, bias=False),
            nn.Tanh(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(32, 16, 3, padding=1, bias=False),
            nn.Tanh(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(16, 3, 3, padding=1, bias=False),
            nn.Sigmoid()
        )

        # Estimation network
        self.fc9 = nn.Linear(code_length + 2, 10)
        self.fc10 = nn.Linear(10, n_gmm)

        # Classifier
        self.proto_classifier = Proto_Classifier(code_length, num_classes)
        self.scaling = nn.Parameter(torch.tensor([1.0]))
        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='tanh')
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='tanh')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def encode(self, x):
        """Encode input image to latent code"""
        # x: (batch, 3, img_size, img_size)
        h = self.encoder(x)
        h = h.view(h.size(0), -1)  # Flatten
        return self.en_fc(h)

    def decode(self, z):
        """Decode latent code to reconstructed image"""
        # z: (batch, code_len)
        # h = self.fc_bottleneck.weight.t().mm(z.t()).t()  # Equivalent to linear layer
        h = self.de_fc(z)
        h = h.view(h.size(0), 128, self.img_size // 8, self.img_size // 8)
        return self.decoder(h)

    def compute_reconstruction(self, x, x_hat):
        """Compute reconstruction metrics for image data"""
        # Flatten images to compute metrics
        x_flat = x.view(x.size(0), -1)
        x_hat_flat = x_hat.view(x_hat.size(0), -1)

        # Relative Euclidean distance (normalized)
        rel_euclid = (x_flat - x_hat_flat).norm(2, dim=1) / x_flat.norm(2, dim=1)

        # Cosine similarity
        cos_sim = F.cosine_similarity(x_flat, x_hat_flat, dim=1)

        return rel_euclid, cos_sim

    def estimate(self, z):
        """Estimate GMM weights using latent code and metrics"""
        h = F.dropout(torch.tanh(self.fc9(z)), 0.5)
        return F.softmax(self.fc10(h), dim=1)

    def forward(self, x):
        """Forward pass for image data"""
        z_c = self.encode(x)  # Latent code
        x_hat = self.decode(z_c)  # Reconstructed image

        # Compute reconstruction metrics
        rec_1, rec_2 = self.compute_reconstruction(x, x_hat)

        # Concatenate with metrics for estimation
        z = torch.cat([z_c, rec_1.unsqueeze(-1), rec_2.unsqueeze(-1)], dim=1)

        # Estimate GMM weights
        gamma = self.estimate(z)

        return z_c, x_hat, z, gamma