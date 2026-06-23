import copy
import random

import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F
from torchvision import transforms
from torch.autograd import grad
from core.Client.ClientBase import Client

class ClientFedGM(Client):
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
        # =============
        # FedGM
        self.pre_classifiers = None
        # =============
    def update_weights(self, global_round):   # 训练模型
        self.model.to(self.device)
        self.model.train()
        epoch_loss = []
        optimizer = self.optim
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=self.args.lr_sh_rate, gamma=0.5)
        part = 0.1
        self.trainloader = self.get_trainloader()
        for iter in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (X, y) in enumerate(self.trainloader):
                X = X.to(self.device)
                y = y.to(self.device)
                # ======================
                # FedGM数据增强
                X_aug = self.random_color_shift(X)
                # ======================
                optimizer.zero_grad()
                _, p = self.model(X)

                # ===================
                # FedGM
                _, p_aug = self.model(X_aug)
                src_loss1 = self.ce(p_aug, y)
                # ===================
                src_loss2 = self.ce(p,y)

                # ===================
                # FedGM
                task_loss = 0.5*src_loss1 + 0.5*src_loss2

                # intra-domain gradient matching
                grad_cossim11 = []
                # netE+C1
                for n, p in self.model.classifier.named_parameters():
                    # for n, p in model.named_parameters():
                    # if len(p.shape) == 1: continue

                    real_grad = grad([src_loss1],
                                     [p],
                                     create_graph=True,
                                     only_inputs=True,
                                     allow_unused=False)[0]
                    fake_grad = grad([src_loss2],
                                     [p],
                                     create_graph=True,
                                     only_inputs=True,
                                     allow_unused=False)[0]
                    if len(p.shape) > 1:
                        _cossim = F.cosine_similarity(fake_grad, real_grad, dim=1).mean()
                    else:
                        _cossim = F.cosine_similarity(fake_grad, real_grad, dim=0)

                    grad_cossim11.append(_cossim)
                grad_cossim1 = torch.stack(grad_cossim11)
                gm_intra_loss = (1.0 - grad_cossim1).mean()

                # # inter-domain gradient matching
                for i in range(1, len(self.pre_classifiers)):
                    grad_cossim_inter = []
                    feature,_ = self.model(X)
                    output_inter = self.pre_classifiers[i](feature)
                    inter_loss = self.ce(output_inter, y)

                    for g_p, p in zip(self.pre_classifiers[i].named_parameters(), self.model.classifier.named_parameters()):
                        g_p = g_p[1]
                        p = p[1]
                        inter_grad = grad([inter_loss],
                                          [g_p],
                                          create_graph=True,
                                          only_inputs=True,
                                          allow_unused=False)[0]
                        fake_grad = grad([src_loss1],
                                         [p],
                                         create_graph=True,
                                         only_inputs=True,
                                         allow_unused=False)[0]
                        if len(p.shape) > 1:
                            _cossim_inter = F.cosine_similarity(inter_grad, fake_grad, dim=1).mean()
                        else:
                            _cossim_inter = F.cosine_similarity(inter_grad, fake_grad, dim=0)

                        grad_cossim_inter.append(_cossim_inter)

                    grad_cossim_inter = torch.stack(grad_cossim_inter)
                    if i == 1:
                        gm_inter_loss = (1.0 - grad_cossim_inter).mean()
                    else:
                        gm_inter_loss += (1.0 - grad_cossim_inter).mean()

                gm_inter_loss /= (len(self.pre_classifiers) - 1)

                loss = task_loss + 0.5 * gm_inter_loss + 0.5 * gm_intra_loss
                # ===================

                loss.backward()
                if self.args.clip_grad != None:
                    nn.utils.clip_grad_norm_(self.model.parameters(), max_norm = self.args.clip_grad)
                optimizer.step()
                batch_loss.append(loss.item())
            total_loss = sum(batch_loss) / len(batch_loss)
            epoch_loss.append(total_loss)
            self.logger.info(f"Round {global_round} Client {self.idx} Epoch {iter} Loss {total_loss}")
        # self.save_model(self.args.logdir + '/client_' + str(self.idx) + '.pth')
        return self.model.state_dict(),sum(epoch_loss) / len(epoch_loss)

    def random_color_shift(self, image_tensor):
        """
        随机调整 RGB 通道的亮度和对比度
        Args:
            image_tensor: [C, H, W] 范围的 Tensor（通常 [0, 1] 或 [-1, 1]）
        Returns:
            增强后的图像
        """
        # 随机亮度调整（整体加减一个值）
        brightness = random.uniform(-0.1, 0.1)
        image_tensor = image_tensor + brightness

        # 随机对比度调整（乘以一个值）
        contrast = random.uniform(0.9, 1.1)
        image_tensor = image_tensor * contrast
        return image_tensor

    def load_pre_classifiers(self, pre_classifiers):
        self.pre_classifiers = copy.deepcopy(pre_classifiers)