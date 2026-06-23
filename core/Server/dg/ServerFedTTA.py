import copy
import torch
import math
import time

import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
from torchvision import transforms
from PIL import Image, ImageFilter, ImageOps
import torchvision.models as models
import random

from core.Client.dg.ClientFedTTA import ClientFedTTA
from core.Server.ServerBase import Server
from mem_utils import MemReporter
from model.domain_model import Proto_Classifier


class SupConLoss(nn.Module):

    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf
        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # Compute logits
        mean_x = torch.mean(anchor_feature)
        mean_y = torch.mean(contrast_feature)

        # 每个向量减去均值
        anchor_feature = anchor_feature - mean_x
        contrast_feature = contrast_feature - mean_y
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        anchor_dot_contrast[anchor_dot_contrast == float('inf')] = 1

        # For numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()
        logits[logits == float('inf')] = 1

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask
        mask[mask == float('inf')] = 1

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
        log_prob[log_prob == float('inf')] = 1

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss


class NCropsTransform:
    def __init__(self, transform_list) -> None:
        self.transform_list = transform_list

    def __call__(self, x):
        data = [tsfm(x) for tsfm in self.transform_list]
        return data


class GaussianBlur(object):
    """Gaussian blur augmentation from SimCLR: https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x

class fedspl_Model(nn.Module):
    """
    Student and Teacher Models
    """

    def __init__(
            self,
            fed_model,
            ema_model,
            m=0.98,
            checkpoint_path=None,
    ):
        """
        dim: feature dimension (default: 128)
        m: EMA coefficient teacher model (default: 0.999)
        """
        super(fedspl_Model, self).__init__()

        self.m = m
        self.queue_ptr = 0

        # create the encoders
        self.fed_model = fed_model
        self.ema_model = ema_model

        # create the fc heads
        # feature_dim = fed_model.output_dim

        # freeze key model
        for name, param in self.ema_model.named_parameters():
            param.requires_grad_(False)
        if checkpoint_path:
            self.load_from_checkpoint(checkpoint_path)

    def load_from_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        state_dict = dict()
        for name, param in checkpoint["state_dict"].items():
            name = name[len("module."):] if name.startswith("module.") else name
            state_dict[name] = param

        msg = self.load_state_dict(state_dict, strict=False)


    @torch.no_grad()
    def _ema_model_update(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(
                self.fed_model.parameters(), self.ema_model.parameters()
        ):
            param_k.data = param_k.data * self.m + param_q.data * (1.0 - self.m)

    def forward(self, im_q, im_k=None, cls_only=False):
        """
        Input:
            im_q: 1st batch of augmented images
            im_k: 2nd batch of aumented images
        Output:
            Features and logits
        """

        ## Compute query features
        feats_q, logits_q = self.fed_model(im_q)

        if cls_only:
            return feats_q, logits_q

        ## EMA update of the Teacher Model
        with torch.no_grad():
            self._ema_model_update()

            # dequeue and enqueue will happen outside
        return feats_q, logits_q


def conv1x1(input_channel, output_channel,bias=False):
    return nn.Conv2d(input_channel, output_channel, kernel_size=1, bias=bias)


def conv3x3(in_channel, out_channel, stride=1, padding=1, bias=False):
    return nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=stride, padding=padding,bias=bias)


def random_sample(prob, sampling_num):
    batch_size, channels, h, w = prob.shape
    return torch.multinomial((prob.view(batch_size * channels, -1) + 1e-8), sampling_num, replacement=True)


class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, input_channel, output_channel, stride=1, downsample=None, track_running_stats=True):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(input_channel, output_channel, stride=stride)
        self.bn1 = nn.BatchNorm2d(output_channel, track_running_stats=track_running_stats)
        self.relu = nn.ReLU(inplace=False)
        self.conv2 = conv3x3(output_channel, output_channel)
        self.bn2 = nn.BatchNorm2d(output_channel, track_running_stats=track_running_stats)
        self.downsample = downsample
        self.stride = stride

    def forward(self, input):
        residual = input
        out = self.conv1(input)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(input)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, input_channel, channel, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = conv1x1(input_channel, channel)
        self.bn1 = nn.BatchNorm2d(channel)
        self.relu = nn.ReLU(inplace=False)
        self.conv2 = conv3x3(channel, channel, stride=stride)
        self.bn2 = nn.BatchNorm2d(channel)
        self.conv3 = conv1x1(channel, channel*4)
        self.bn3 = nn.BatchNorm2d(channel*4)
        self.downsample = downsample
        self.stride = stride

    def forward(self, input):
        residual = input # skip path
        out = self.conv1(input)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(input)

        out += residual

        return self.relu(out)


class ResNet18_domain(nn.Module):

    def __init__(self, args=None, code_length=64, num_classes=10):
        super(ResNet18_domain, self).__init__()
        self.code_length = code_length
        # model = models.resnet18(weights='IMAGENET1K_V1')  # 域泛化用预训练模型
        model = models.resnet18()
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, num_classes)
        modules = list(model.children())[:-1]  # 移除最后的全连接层
        self.feature_extractor = nn.Sequential(*modules)
        # self.feature_extractor = self.model
        print(num_features)
        self.classifier = nn.Sequential(
            nn.Linear(num_features, num_classes))
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
    # def get_params(self):
    #     """
    #     Backbone parameters use 1x lr; extra parameters use 10x lr.
    #     """
    #     backbone_params = []
    #     extra_params = []
    #     # case 1)
    #     # if not True:
    #     #     backbone_params.extend(self.parameters())
    #     # # case 2)
    #     # else:
    #     for module in self.baselayer[:-1]:  # Exclude the last fc_class
    #         backbone_params.extend(module.parameters())
    #
    #         # Add extra params
    #         extra_params.extend(self.fc_class.parameters())
    #     backbone_params = [param for param in backbone_params if param.requires_grad]
    #     extra_params = [param for param in extra_params if param.requires_grad]
    #
    #     return backbone_params, extra_params

    @property
    def num_classes(self):
        return self.fc.weight.shape[0]

    # @property
    def output_dim(self):
        return self._output_dim

    @property
    def use_bottleneck(self):
        return self.args.bottleneck_dim > 0

    @property
    def use_weight_norm(self):
        return self.args.weight_norm_dim >= 0


class ServerFedTTA  (Server):
    def __init__(self, args, global_model,Loader_train,Loaders_local_test,Loader_global_test,logger,device, net_idx_dataidx_map):
        super().__init__(args, global_model,Loader_train,Loaders_local_test,Loader_global_test,logger,device, net_idx_dataidx_map)

    def Create_Clints(self):
        for idx in range(self.args.num_clients):
            self.LocalModels.append(ClientFedTTA(self.args, copy.deepcopy(self.global_model),self.net_idx_dataidx_map[idx], idx=idx, logger=self.logger, code_length = self.args.code_len, num_classes = self.args.num_classes, device=self.device))

    def train(self):
        reporter = MemReporter()
        start_time = time.time()
        train_loss = []

        for epoch in tqdm(range(self.args.comm_round)):
            test_accuracy = 0
            local_weights, local_losses = [], []
            self.logger.info(f'Global Training Round: {epoch+1}')
            m = max(int(self.args.sampling_rate * self.args.num_clients), 1)
            idxs_users = np.random.choice(range(self.args.num_clients), m, replace=False)
            global_weights = self.global_model.state_dict()
            for idx in idxs_users:
                if self.args.upload_model == True:
                    self.LocalModels[idx].load_model(global_weights)
                w, loss = self.LocalModels[idx].update_weights(global_round=epoch)
                local_losses.append(copy.deepcopy(loss))
                local_weights.append(copy.deepcopy(w))
                acc = self.LocalModels[idx].test_accuracy()
                test_accuracy += acc

            # update global weights
            global_weights = self.average_weights(local_weights, idxs_users)
            self.global_model.load_state_dict(global_weights)

            # print loss
            loss_avg = sum(local_losses) / len(local_losses)
            train_loss.append(loss_avg)
            cur_g_acc = self.global_test_accuracy()
            if cur_g_acc > self.global_best_acc:
                self.global_best_acc = cur_g_acc
                self.Save_CheckPoint(self.args.logdir + '/best_model.pth')
            if test_accuracy / len(idxs_users)  > self.global_best_personal_acc:
                self.global_best_personal_acc = test_accuracy / len(idxs_users)
            self.logger.info(f'Global Training Loss: {loss_avg}')
            self.logger.info(
                f'Personal_Accuracy: {test_accuracy / len(idxs_users)} || Best_Personal_Accuracy: {self.global_best_personal_acc}')
            self.logger.info(f'Global_Accuracy: {cur_g_acc} || Best_Accuracy: {self.global_best_acc}')
            self.csv_log(
                {'DataName': f"{self.args.dataset}_{self.args.leave_domain}", 'Round': epoch + 1, 'Loss': loss_avg,
                 'Personal_Acc': test_accuracy / len(idxs_users), 'Global_Acc': cur_g_acc})

        self.TTA(local_weights)
        self.test_domain()
        self.logger.info('Training is completed.')
        end_time = time.time()
        self.logger.info(f'Total running time: {end_time - start_time} s || avg: {(end_time - start_time) / self.args.comm_round} s')
        reporter.report()

    def TTA(self, local_weights):
        # tta_model = copy.deepcopy(self.global_model)

        for lw in local_weights:  # 遍历源域模型
            # tta_model.load_state_dict(lw)
            self.train_target_model(lw)



    def train_target_model(self, source_model):
        fed_model = ResNet18_domain(self.args,512,self.args.num_classes)
        ema_model = ResNet18_domain(self.args,512,self.args.num_classes)  ## For contrastive loss
        fed_model.load_state_dict(copy.deepcopy(source_model))
        ema_model.load_state_dict(copy.deepcopy(source_model))


        model = fedspl_Model(
            fed_model,
            ema_model,
            m=0.999,
        ).cuda()

        val_loader = copy.deepcopy(self.global_testloader)
        train_loader = copy.deepcopy(self.global_testloader)

        val_transform = get_augmentation("test")  # 用的就是原始数据集

        train_transform = get_augmentation_versions()

        val_loader.dataset.transform = val_transform
        train_loader.dataset.transform = train_transform

        optimizer = get_target_optimizer(model)

        # torch.utils.data.DataLoader(one_test_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)

        self.train_tta(train_loader, val_loader, model, optimizer)

    def train_tta(self, train_loader, val_loader, model, optimizer):

        epoch = 1
        ## Make sure to switch to train mode
        model.train()

        num_class = 7
        N = 8
        accuracy_tot = 0
        accuracy_r = 0
        total_acc = 1
        end = time.time()
        zero_tensor = torch.tensor([0.0]).to("cuda")
        loss_coef = 1
        con_coeff = 0.5
        contrastive_criterion = SupConLoss()
        L2loss = torch.nn.MSELoss()

        mem_size = 2
        class_features = torch.zeros((mem_size, num_class, 256))
        probs_class = torch.zeros((mem_size, num_class, num_class))

        con_coeffs = np.zeros(20000)
        loss_classes = torch.zeros(20000)
        loss_coefs = torch.zeros(20000)
        con_losses = torch.zeros(20000)
        unsupervised_losses = torch.zeros(20000)
        # uncertainty_thresholds = torch.zeros(20000)
        conf_thress = torch.zeros(20000)
        acc_classes = []
        accuracies = []
        sel_Samples = []
        unsel_samples = []
        missed_images = {'img_path': [], 'labels': []}
        ind = 0

        for epoch in range(0, 15):
            # print(args.learn.start_epoch)
            # print(type(train_loader))
            for i, data in enumerate(train_loader):
                ## Unpack and move data
                images, labels_check = data
                labels_check = labels_check.to("cuda")

                ## Images for updating the model
                images_w, images_q, images_k = (
                    images[0].to("cuda"),
                    images[1].to("cuda"),
                    images[2].to("cuda"),
                )

                ## (N-3) number of Images for Calculating uncertainty
                images_un = torch.stack(images[3:N]).to("cuda")
                outputs_emas = []
                feats = []

                with torch.no_grad():
                    for jj in range(N - 3):
                        outputs_emas.append(model.ema_model(images_un[jj])[1])

                        ## Average the predictions for pseudo-labels
                    outputs_ema = torch.stack(outputs_emas).mean(0)
                    probs_w, pseudo_labels_w = torch.nn.functional.softmax(outputs_ema, dim=1).max(1)

                ## Per-step scheduler (Learning Rate Decay)
                step = i + epoch * len(train_loader)
                # adjust_learning_rate(optimizer, step, args)

                ## Get the logits
                feats_con, logits_q = model(images_q, images_k)

                softmax_outputs = torch.nn.functional.softmax(torch.squeeze(torch.stack(outputs_emas)), dim=2)
                top2_values, _ = torch.topk(softmax_outputs, 2, dim=2)
                pred_start = top2_values[:, :, 0]
                pred_start_2nd = top2_values[:, :, 1]

                pred_con = pred_start - pred_start_2nd
                conf_thres = pred_con.mean()
                confidence_sel = pred_con.mean(0) > conf_thres
                conf_th = pred_con.mean()

                ## Uncertainty Based Selection

                truth_array = confidence_sel
                ind_keep = truth_array.nonzero()
                ind_remove = (~truth_array).nonzero()

                try:
                    ind_total = torch.cat((torch.squeeze(ind_keep), torch.squeeze(ind_remove)), dim=0)
                except:
                    ind_total = ind_remove

                if ind_remove.numel():
                    threshold = torch.zeros(len(ind_remove))
                    num = 0
                    for kk in ind_remove:
                        out = torch.squeeze(outputs_ema[kk])
                        out, _ = out.sort(descending=True)
                        threshold[num] = out[0] - out[1]
                        num += 1

                    pre_threshold = threshold.mean(0)
                    truth_array1 = threshold > pre_threshold

                    ind_add = truth_array1.nonzero()

                    try:
                        ind_keep = torch.cat((torch.squeeze(ind_keep), torch.squeeze(ind_remove[ind_add])), dim=0)
                        ind_remove = torch.stack([kk for kk in ind_total if kk not in ind_keep])
                    except:
                        pass

                try:

                    unique_labels, counts = pseudo_labels_w[ind_keep].unique(return_counts=True)
                    min_count = torch.min(counts)

                    if len(counts) < num_class:
                        counts_new = torch.ones(num_class)
                        missing_classes = [ii for ii in range(num_class) if ii not in unique_labels]

                        for kk in missing_classes:
                            indices = (pseudo_labels_w == kk).nonzero(as_tuple=True)[0]

                            if indices.numel() > 0 and ind_keep.numel() > 0:
                                probs = probs_w[indices]
                                _, index_miss = probs.sort(descending=True)

                                try:
                                    ind_keep = torch.cat((ind_keep, indices[index_miss[
                                                                            0:min_count]]))  ## Taking all missing classes samples deteriorates the performance
                                except:
                                    pass
                                counts_new[kk] = 1
                            else:
                                counts_new[kk] = 1

                        num = 0
                        for nn in unique_labels:
                            counts_new[nn] = counts[num]
                            num += 1
                    else:
                        counts_new = counts

                    loss_cls, accuracy_psd = classification_loss(
                        torch.squeeze(outputs_ema[ind_keep]), torch.squeeze(logits_q[ind_keep]),
                        torch.squeeze(pseudo_labels_w[ind_keep]), torch.squeeze(outputs_ema[ind_keep]),
                        1 / counts_new.cuda()
                    )

                except:
                    # print("Oh No!!!, There is no confident samples::", ind_keep.numel(), ind_remove.numel())

                    loss_cls, accuracy_psd = propagation_loss(
                        torch.squeeze(outputs_ema), torch.squeeze(logits_q), torch.squeeze(pseudo_labels_w),
                        torch.squeeze(outputs_ema)
                    )

                accuracy = (pseudo_labels_w[ind_keep] == labels_check[ind_keep]).float().mean() * 100
                if not math.isnan(accuracy):
                    accuracy_tot += accuracy
                    total_acc += 1
                accuracies.append(accuracy_tot / total_acc)

                # print(f'Eopch:{epoch}, batch acc: {accuracy_tot / total_acc}')

                feats_k = model(images_k, cls_only=True)[0]
                feature_1 = torch.squeeze(feats_con[ind_remove])
                # print(feature_1.shape)
                # print(1)
                if len(feature_1.shape) == 1:
                    feature_1 = feature_1.unsqueeze(0)
                # print(feature_1.shape)
                f1 = F.normalize(feature_1, dim=1)
                feature_2 = torch.squeeze(feats_k[ind_remove])
                # print(2)
                # print(feature_2.shape)
                # print(3)
                if len(feature_2.shape) == 1:
                    feature_2 = feature_2.unsqueeze(0)
                # print(feature_2.shape)
                # print(4)
                f2 = F.normalize(feature_2, dim=1)
                features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
                loss_contrast = contrastive_criterion(features)

                cnm = torch.squeeze(logits_q[ind_remove])
                if len(cnm.shape) == 1:
                    cnm = cnm.unsqueeze(0)
                # print(cnm.shape)
                loss_cls_rem, accuracy_psd_meter = propagation_loss(
                    torch.squeeze(outputs_ema[ind_remove]), cnm, torch.squeeze(pseudo_labels_w[ind_remove]),
                    torch.squeeze(outputs_ema[ind_remove])
                )
                # print('cao')

                _, accuracy_psd_meter = propagation_loss(
                    torch.squeeze(outputs_ema), torch.squeeze(logits_q), torch.squeeze(pseudo_labels_w),
                    torch.squeeze(outputs_ema)
                )


                # difficulty_score = uncer_th/conf_th
                difficulty_score = 1 / conf_th
                loss_coef *= (1 - 0.001 * torch.exp(-1 / difficulty_score))
                con_coeff *= np.exp(-0.0001)

                ## At the beginning, we want to learn from more confident samples
                loss = loss_coef * loss_cls + (1 - loss_coef) * loss_cls_rem + con_coeff * loss_contrast
                # loss=  loss_coef * loss_cls

                ## Update the Parameters

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                con_coeffs[ind] = con_coeff
                loss_classes[ind] = loss_cls.item()
                loss_coefs[ind] = loss_coef
                con_losses[ind] = loss_contrast.item()
                # print(loss_cls_rem)
                unsupervised_losses[ind] = loss_cls_rem.item()
                # uncertainty_thresholds[ind] = uncer_th
                conf_thress[ind] = conf_th
                sel_Samples.append(len(ind_keep))
                unsel_samples.append(len(ind_remove))

                end = time.time()



                missed_images['labels'].append(labels_check[ind_remove])

                ind += 1
                ## Evaluate the model ##
            acc_per_class = eval_and_label_dataset(val_loader, model)
            model.train()
            acc_classes.append(acc_per_class.mean())

        print("一个准确率：", np.mean(accuracies))

def eval_and_label_dataset(dataloader, model):
    wandb_dict = dict()

    # Make sure to switch to eval mode
    model.eval()

    # Run inference
    logits, gt_labels, indices = [], [], []

    iterator = tqdm(dataloader)
    for imgs, labels in iterator:
        imgs = imgs.to("cuda")

        # (B, D) x (D, K) -> (B, K)
        _, logits_cls = model(imgs, cls_only=True)

        logits.append(logits_cls)
        gt_labels.append(labels)

    logits    = torch.cat(logits)
    gt_labels = torch.cat(gt_labels).to("cuda")



    assert len(logits) == len(dataloader.dataset)

    pred_labels = logits.argmax(dim=1)
    accuracy = (pred_labels+1 == gt_labels).float().mean() * 100



    return accuracy




def get_augmentation_versions():
    """
    Get a list of augmentations. "w" stands for weak, "s" stands for strong.
    E.g., "wss" stands for one weak, two strong.
    """
    transform_list = []
    for version in 'wsstwsws':  ## Change the value of augmented versions
        if version == "s":
            transform_list.append(get_augmentation('moco'))
        elif version == "w":
            transform_list.append(get_augmentation("plain"))
        elif version == "t":
            transform_list.append(get_augmentation("test"))
        else:
            raise NotImplementedError(f"{version} version not implemented.")
    # print(transform_list)
    transform = NCropsTransform(transform_list)

    return transform


def get_augmentation(aug_type, normalize=None):
    if not normalize:
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
    if aug_type == "moco-v2":
        return transforms.Compose(
            [
                transforms.RandomResizedCrop(224, scale=(0.2, 1.0)),
                transforms.RandomApply(
                    [transforms.ColorJitter(0.8, 0.8, 0.5, 0.2)],
                    p=0.8,
                ),
                transforms.RandomGrayscale(p=0.2),
                transforms.RandomRotation(degrees = [-2,2]),
                transforms.RandomPosterize(8, p=0.2),
                transforms.RandomEqualize(p=0.2),
                transforms.RandomApply([GaussianBlur([0.1, 2])], p=0.5),
                # transforms.AugMix(5,5),           ## While Applying Augmix, comment out the ColorJitter
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]
        )


    elif aug_type == "moco":
        return transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.RandomResizedCrop(224, scale=(0.2, 1.0)),
                transforms.RandomGrayscale(p=0.2),
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.4),
                # ImageNetPolicy(),
                transforms.RandomHorizontalFlip(),

                normalize,
            ]
        )
    elif aug_type == "plain":
        return transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize((256, 256)),
                transforms.RandomCrop(224),
                transforms.RandomGrayscale(p=0.05),           ## prob 0.1 works
                transforms.ColorJitter(0.1, 0.1, 0.1, 0.05),  ## all 0.1 works
                transforms.RandomHorizontalFlip(),
                normalize,
            ]
        )
    elif aug_type == "clip_inference":
        return transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize(224, interpolation=Image.BICUBIC),
                transforms.CenterCrop(224),
                normalize,
            ]
        )
    elif aug_type == "test":
        return transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize((256, 256)),
                transforms.CenterCrop(224),
                normalize,
            ]
        )
    return None


def get_target_optimizer(model):
    #
    # backbone_params, extra_params = (
    #     model.fed_model.get_params()
    # )


    optimizer = torch.optim.SGD(
        [
            {
                "params": model.fed_model.feature_extractor.parameters(),
                "lr": 0.001,
                "momentum": 0.9,
                "weight_decay": 1e-5,
                "nesterov": True,
            },
            {
                "params": model.fed_model.classifier.parameters(),
                "lr": 30 * 0.001,

                "weight_decay": 1e-5,
                "nesterov": True,
            },
        ]
    )

    for param_group in optimizer.param_groups:
        param_group["lr0"] = param_group["lr"]  # snapshot of the initial lr

    return optimizer

def cross_entropy_loss(logits, labels, class_weights=None):

    return F.cross_entropy(logits, labels, weight = class_weights)



def classification_loss(outputs_ema, logits_s, target_labels, targets_preds, class_weights=None):
    # if args.learn.ce_sup_type == "weak_weak":
    #     loss_cls = cross_entropy_loss(outputs_ema, target_labels, args, class_weights)
    #     accuracy = calculate_acc(outputs_ema, target_labels)
    # elif args.learn.ce_sup_type == "weak_strong":
    #     # loss_cls = torch.mean((logits_s - targets_preds)**2)
    #     loss_cls = cross_entropy_loss(logits_s, target_labels, args, class_weights)
    #     accuracy = calculate_acc(logits_s, target_labels)
    # else:
    #     raise NotImplementedError(
    #         f"{args.learn.ce_sup_type} CE supervision type not implemented."
    #     )

    # loss_cls = torch.mean((logits_s - targets_preds)**2)
    loss_cls = cross_entropy_loss(logits_s, target_labels, class_weights)
    accuracy = calculate_acc(logits_s, target_labels)
    return loss_cls, accuracy


def propagation_loss(outputs_ema, logits_s, target_labels, targets_preds):
    # if args.learn.ce_sup_type == "weak_weak":
    #     loss_cls = cross_entropy_loss(outputs_ema, target_labels, args)
    #     accuracy = calculate_acc(outputs_ema, target_labels)
    # elif args.learn.ce_sup_type == "weak_strong":
    #     loss_cls = torch.mean((targets_preds - logits_s) ** 2)
    #
    #     accuracy = calculate_acc(logits_s, target_labels)
    # else:
    #     raise NotImplementedError(
    #         f"{args.learn.ce_sup_type} CE supervision type not implemented."
    #     )

    loss_cls = torch.mean((targets_preds - logits_s) ** 2)

    accuracy = calculate_acc(logits_s, target_labels)
    return loss_cls, accuracy

@torch.no_grad()
def calculate_acc(logits, labels):
    preds = logits.argmax(dim=1)
    accuracy = (preds == labels).float().mean() * 100
    return accuracy