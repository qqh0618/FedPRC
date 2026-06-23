import time

import torch
import torch.nn as nn
import torch.optim as optim
from core.Client.ClientBase import Client
from torch.autograd import Function
import torch.nn.functional as F
import random
from sklearn.cluster import KMeans
from torchvision.transforms import AugMix
from torchvision import transforms



class MixStyle(nn.Module):
    def __init__(self, p=1.0, alpha=0.1, eps=1e-6, n_clusters=4):
        super(MixStyle, self).__init__()
        self.p = p
        self.beta = torch.distributions.Beta(torch.tensor(alpha), torch.tensor(alpha))
        self.eps = eps
        self.alpha = alpha

    def forward(self, x, mu2, std2):
        if random.random() > self.p:
            return x
        B = x.size(0)
        mu = x.mean(dim=[2, 3], keepdim=True)
        var = x.var(dim=[2, 3], keepdim=True)
        std = (var + self.eps).sqrt()
        mu, std = mu.detach(), std.detach()
        x_normed = (x - mu) / std

        perm = torch.randperm(B)
        mu2 = mu[perm]
        std2 = std[perm]

        mu2 = mu2.to(x.device)
        std2 = std2.to(x.device)

        #         lmda = self.beta.sample((B, 1, 1, 1))
        #         lmda = lmda.to(x.device)

        difficulty_score = 1 - torch.softmax(x_normed.mean(dim=[2, 3]), dim=1).max(dim=1)[0].unsqueeze(-1).unsqueeze(
            -1).unsqueeze(-1)
        k = 15.0
        lmda = torch.sigmoid(k * difficulty_score) * self.beta.sample((B, 1, 1, 1)).to(x.device)

        mu_mix = mu * lmda + mu2 * (1 - lmda)
        sig_mix = std * lmda + std2 * (1 - lmda)
        return x_normed * sig_mix + mu_mix


class DomainDiscriminator(nn.Module):
    def __init__(self, feature_dim=1280):  # Updated default feature_dim to match input
        super(DomainDiscriminator, self).__init__()
        # Gradient reversal scaling factor
        self.alpha = 1.0

        # Adjusted architecture to handle 1280-dimensional input
        self.fc1 = nn.Linear(feature_dim, 1024)
        self.bn1 = nn.BatchNorm1d(1024)
        self.dropout1 = nn.Dropout(0.3)

        self.fc2 = nn.Linear(1024, 512)
        self.bn2 = nn.BatchNorm1d(512)
        self.dropout2 = nn.Dropout(0.3)

        self.ln = nn.LayerNorm(256)

        self.fc3 = nn.Linear(512, 1)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, alpha=None):
        if alpha is not None:
            self.alpha = alpha

        # Apply gradient reversal scaling
        if self.training:
            x = x * self.alpha

        # Main forward path with residual connection
        identity = x

        x = self.dropout1(self.bn1(F.relu(self.fc1(x))))
        x = self.dropout2(self.bn2(F.relu(self.fc2(x))))

        # Removed residual connection since dimensions don't match
        x = self.fc3(x)
        return x


class GradientReversal(Function):
    @staticmethod
    def forward(ctx, x):
        # Save any parameters if needed (e.g., scale factor)
        ctx.alpha = 1.0  # You can use a scaling factor if required
        return x

    @staticmethod
    def backward(ctx, grad_output):
        # Reverse the gradient during backpropagation
        return -ctx.alpha * grad_output


class ClientFedAlign(Client):
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
        self.statistic_pool = {}

        self.lambda1 = 0.1
        self.lambda2 = 0.2
        self.lambda3 = 0.3
        self.t = 0.5
        self.epsilon = 1e-6
        self.r = 0.1
        self.p = 1
        self.lambda_dann = 0.1

        self.MixStyle = MixStyle(self.p, 0.1, self.epsilon)
        self.n_clusters = getattr(self.args, 'n_clusters', 3)
        self.domain_discriminator = DomainDiscriminator(feature_dim=512).to(
            self.device)  # Update feature_dim as needed
        self.domain_optimizer = torch.optim.Adam(self.domain_discriminator.parameters(), lr=self.args.lr)
        self.domain_criterion = torch.nn.BCEWithLogitsLoss()



    @torch.no_grad()
    def compute_statistic(self):
        local_statistic_pool = {"mean": [], "std": []}
        num2upload = int(len(self.trainloader.dataset) * self.r)
        batches = int(num2upload / self.args.batch_size)
        left_num = num2upload % self.args.batch_size
        for enu, (data, target) in enumerate(self.trainloader):
            mean = torch.mean(data, dim=(2, 3), keepdim=True)
            var = torch.var(data, dim=(2, 3), keepdim=True)
            std: torch.Tensor = (var + self.epsilon).sqrt()
            if enu != batches:
                local_statistic_pool["mean"].append(mean)
                local_statistic_pool["std"].append(std)
            else:
                local_statistic_pool["mean"].append(mean[:left_num])
                local_statistic_pool["std"].append(std[:left_num])
                break
        local_statistic_pool["mean"] = torch.cat(
            local_statistic_pool["mean"], dim=0
        ).to(torch.device("cpu"))
        local_statistic_pool["std"] = torch.cat(local_statistic_pool["std"], dim=0).to(
            torch.device("cpu")
        )
        return local_statistic_pool

    def compute_global_statistics(self, local_statistics_pool):
        all_means = []
        all_stds = []
        for statistic_pool in local_statistics_pool:
            all_means.append(statistic_pool["mean"])
            all_stds.append(statistic_pool["std"])
        global_mean = torch.cat(all_means, dim=0)
        global_std = torch.cat(all_stds, dim=0)
        global_mean = global_mean.mean(dim=0)
        global_std = global_std.mean(dim=0)
        return global_mean, global_std


    def update_weights(self, global_round):   # 训练模型
        self.model.to(self.device)
        self.model.train()
        epoch_loss = []
        optimizer = self.optim
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=self.args.lr_sh_rate, gamma=0.5)

        self.fit_clusters()
        self.compute_weights()

        self.trainloader = self.get_trainloader()
        for iter in range(self.args.local_ep):
            batch_loss = []
            start_time = time.time()
            for batch_idx, (data, target) in enumerate(self.trainloader):
                data = data.to(self.device)
                target = target.to(self.device)
                # 转为半精度
                # data = data.half()
                # target = target.half()
                optimizer.zero_grad()
                feature_1, output = self.model(data)
                loss = self.ce(output,target)

                mix_output = []
                mix_feature = []

                for _ in range(1):
                    mu2, std2 = self.sample_statistic(len(data))
                    generated_data = self.MixStyle(data, mu2, std2)
                    generated_data = self.AugMixAugmentation(generated_data)
                    feature,pred = self.model(generated_data)
                    loss += self.ce(pred, target)
                    if self.lambda2 > 0:
                        mix_output.append(F.softmax(pred, dim=1))
                    if self.lambda1 > 0:
                        mix_feature.append(feature)
                for _ in range(1):
                    mu2, std2 = self.sample_statistic(len(generated_data))
                    generated_data1 = self.MixStyle(generated_data, mu2, std2)
                    generated_data1 = self.AugMixAugmentation(generated_data1)
                    feature,pred = self.model(generated_data1)
                    loss += self.ce(pred, target)
                    if self.lambda2 > 0:
                        mix_output.append(F.softmax(pred, dim=1))
                    if self.lambda1 > 0:
                        mix_feature.append(feature)
                # print(feature_1.device)
                domain_labels_original = torch.zeros(data.size(0), 1).to(self.device)
                domain_labels_augmented = torch.ones(2 * data.size(0), 1).to(self.device)
                domain_labels = torch.cat([domain_labels_original, domain_labels_augmented])
                domain_labels = domain_labels.squeeze(dim=-1).to(self.device)
                domain_features = torch.cat([feature_1, mix_feature[0], mix_feature[1]]).to(self.device)

                #                 reversed_features = GradientReversal.apply(domain_features)
                #                 domain_pred = self.domain_discriminator(reversed_features).to(self.device)
                domain_pred = self.domain_discriminator(domain_features.detach()).to(self.device)
                # domain_pred = self.domain_discriminator(domain_features).to(self.device)

                domain_pred = domain_pred.squeeze(dim=-1)
                domain_loss = self.domain_criterion(domain_pred, domain_labels).to(self.device)

                # domain_loss_scaled = self.lambda_dann*(domain_loss.item()/float(len(domain_labels_original)))y
                # loss = loss + domain_loss_scaled

                # Added representation consistency loss
                if self.lambda3 > 0:
                    consistency_loss = 0.0
                    for mix_feat in mix_feature:
                        consistency_loss += F.mse_loss(mix_feat, feature_1)  # ||h(X) - h(X_aug)||
                    loss += self.lambda3 * (consistency_loss / len(mix_feature))

                #                 original JS loss
                if self.lambda2 > 0:
                    M = torch.clamp(
                        (output + mix_output[0] + mix_output[1]) / 3, 1e-7, 1
                    ).log()
                    kl_1 = F.kl_div(M, output, reduction="batchmean")

                    if kl_1.isnan():
                        kl_1 = 0

                    kl_2 = F.kl_div(M, mix_output[0], reduction="batchmean")
                    kl_3 = F.kl_div(M, mix_output[1], reduction="batchmean")
                    JS_loss = (kl_1 + kl_2 + kl_3) / 3
                    loss += self.lambda2 * JS_loss

                if self.lambda1 > 0:
                    predication_alignment_loss = (
                                                         self.supervised_contrastive_loss(
                                                             mix_feature[0],
                                                             feature_1,
                                                             target,
                                                             temperature=self.t,
                                                         )
                                                         + self.supervised_contrastive_loss(
                                                     mix_feature[1],
                                                     feature_1,
                                                     target,
                                                     temperature=self.t,
                                                 )
                                                 ) / 2
                    loss += self.lambda1 * predication_alignment_loss

                self.domain_optimizer.zero_grad()
                domain_loss.backward(retain_graph=True)
                self.domain_optimizer.step()

                loss.backward()
                if self.args.clip_grad != None:
                    nn.utils.clip_grad_norm_(self.model.parameters(), max_norm = self.args.clip_grad)
                optimizer.step()
                batch_loss.append(loss.item())
            # print("一轮训练耗时:", time.time() - start_time, )
            total_loss = sum(batch_loss) / len(batch_loss)
            epoch_loss.append(total_loss)
            self.logger.info(f"Round {global_round} Client {self.idx} Epoch {iter} Loss {total_loss}")
        self.save_model(self.args.logdir + '/client_' + str(self.idx) + '.pth')
        return self.model.state_dict(), sum(epoch_loss) / len(epoch_loss)

    def set_statistic(self, statistic_pool):
        self.statistic_pool = {}
        statistic_pool["mean"].pop(self.idx-1)
        statistic_pool["std"].pop(self.idx-1)
        statistic_pool["mean"] = [x.to(self.device) for x in statistic_pool["mean"]]
        statistic_pool["std"] = [x.to(self.device) for x in statistic_pool["std"]]
        self.statistic_pool["mean"] = torch.cat(statistic_pool["mean"], dim=0)
        self.statistic_pool["std"] = torch.cat(statistic_pool["std"], dim=0)

    def fit_clusters(self):
        stats = torch.cat([self.statistic_pool["mean"], self.statistic_pool["std"]], dim=1)
        stats = stats.view(stats.size(0), -1)
        statistic_pool_np = stats.cpu().numpy()
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=42)
        kmeans.fit(statistic_pool_np)
        self.cluster_centers_ = torch.tensor(kmeans.cluster_centers_, dtype=torch.float32).to(self.device)
        self.cluster_labels_ = torch.tensor(kmeans.labels_, dtype=torch.long).to(self.device)

    def compute_weights(self):
        stats = torch.cat([self.statistic_pool["mean"], self.statistic_pool["std"]], dim=1)
        stats = stats.view(stats.size(0), -1)
        variance = torch.var(stats, dim=1).to(self.device)
        self.sampling_weights = torch.zeros_like(variance).to(self.device)
        for cluster_id in range(self.n_clusters):
            cluster_indices = (self.cluster_labels_ == cluster_id).to(self.device)
            cluster_variance = variance[cluster_indices]
            cluster_weights = cluster_variance / cluster_variance.sum()
            self.sampling_weights[cluster_indices] = cluster_weights.to(self.device)

    def sample_statistic(self, current_batch_size):
        if not hasattr(self, 'sampling_weights'):
            raise ValueError("Sampling weights not computed. Please call compute_weights first.")
        num = self.statistic_pool["mean"].shape[0]
        cluster_sizes = torch.bincount(self.cluster_labels_)
        cluster_sample_sizes = (cluster_sizes / cluster_sizes.sum() * current_batch_size).long()
        cluster_sample_sizes[-1] += current_batch_size - cluster_sample_sizes.sum()
        cluster_sample_sizes = torch.maximum(cluster_sample_sizes, torch.tensor(1, device=cluster_sample_sizes.device))
        sampled_indices = []
        for cluster_id, num_samples in enumerate(cluster_sample_sizes):
            cluster_indices = torch.where(self.cluster_labels_ == cluster_id)[0]
            cluster_indices = cluster_indices.to(self.device)
            cluster_weights = self.sampling_weights[cluster_indices]
            if num_samples > len(cluster_indices):
                sampled = cluster_indices[torch.multinomial(cluster_weights, num_samples, replacement=True)]
            else:
                sampled = cluster_indices[torch.multinomial(cluster_weights, num_samples, replacement=False)]
            sampled_indices.append(sampled)
        sampled_indices = torch.cat(sampled_indices).to(self.device)
        sampled_mean = self.statistic_pool["mean"][sampled_indices].to(self.device)
        sampled_std = self.statistic_pool["std"][sampled_indices].to(self.device)
        return sampled_mean, sampled_std

    def supervised_contrastive_loss(self, x, y, label, temperature):
        x_norm = torch.norm(x, dim=1, keepdim=True)
        y_norm = torch.norm(y, dim=1, keepdim=True)
        x = x / x_norm
        y = y / y_norm
        samples = torch.cat((x, y), dim=0)
        label = torch.cat((label, label), dim=0)
        same_label_matrix = torch.eq(label.unsqueeze(1), label.unsqueeze(0)).float()
        sim = torch.matmul(samples, samples.T) / temperature
        same_label_sim = sim * same_label_matrix
        same_label_num = torch.sum(same_label_matrix, dim=1)
        # diff_label_sim = torch.exp(sim) * diff_label_matrix
        negative_sim = torch.exp(sim)
        negative_sum = torch.log(torch.sum(negative_sim, dim=1) - negative_sim.diag())
        positive_sum = torch.sum(same_label_sim, dim=1) / same_label_num
        sum = torch.mean(-positive_sum + negative_sum)
        return sum

    def denormalize(self, tensor, mean, std):
        # Assuming mean and std are lists of channel means and stds
        mean = torch.as_tensor(mean).reshape(1, -1, 1, 1).to(tensor.device)
        std = torch.as_tensor(std).reshape(1, -1, 1, 1).to(tensor.device)
        return tensor * std + mean

    def AugMixAugmentation(self, input_images):
        mean = torch.tensor([0.485, 0.456, 0.406]).to(input_images.device)
        std = torch.tensor([0.229, 0.224, 0.225]).to(input_images.device)
        input_images = self.denormalize(input_images, mean, std)
        input_images = input_images * 255.0
        input_images = input_images.to(torch.uint8)
        augmix = AugMix()
        # augmixed_images = torch.stack([augmix(x) for x in input_images])
        augmixed_images = augmix(input_images)
        augmixed_images = augmixed_images.float().div(255.0)
        augmixed_images = transforms.Normalize(mean, std)(augmixed_images)
        return augmixed_images
