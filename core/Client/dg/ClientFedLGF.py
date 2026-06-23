import time

import torch
import torch.nn as nn
import torch.optim as optim
from core.Client.ClientBase import Client
import torch.nn.functional as F

class ClientFedLGF(Client):
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
        # '{"perturb_type": "singular", "perturb_dist": "normal", "local_smooth": 0.1, "global_smooth": 0.1, "perturb_init_scale": 10, "perturb_grad_scale": 0.1}'

        hparams = {}
        hparams['local_smooth'] = 0.1
        hparams['global_smooth'] = 0.1
        hparams['perturb_init_scale'] = 10
        hparams['perturb_grad_scale'] = 0.1
        hparams['perturb_type'] = "singular"
        hparams['perturb_dist'] = "normal"


        self.hparams = hparams
        self.perturb_type = "singular"

    def update_weights(self, global_round):   # 训练模型
        self.model.to(self.device)
        self.model.train()
        epoch_loss = []
        optimizer = self.optim
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=self.args.lr_sh_rate, gamma=0.5)
        self.trainloader = self.get_trainloader()
        for iter in range(self.args.local_ep):
            batch_loss = []
            start_time = time.time()
            for batch_idx, (X, y) in enumerate(self.trainloader):
                X = X.to(self.device)
                y = y.to(self.device)
                # 转为半精度
                # X = X.half()
                # y = y.half()
                optimizer.zero_grad()
                feature,logits = self.model(X)
                loss = self.ce(logits,y)

                # 2. local smooth
                logits_local_perturb = self.get_local_perturbed_output(feature.detach(), logits.detach())
                loss_local_smooth = F.kl_div(F.log_softmax(logits, dim=1), F.softmax(logits_local_perturb, dim=1),
                                             reduction='batchmean')
                loss += self.hparams['local_smooth'] * loss_local_smooth

                # 3. global smoothness
                if self.hparams['global_smooth'] > 0:
                    logits_global_perturb = self.get_global_perturbed_output(X, logits.detach())
                    loss_global_smooth = F.kl_div(F.log_softmax(logits, dim=1), F.softmax(logits_global_perturb, dim=1),
                                                  reduction='batchmean')
                    loss += self.hparams['global_smooth'] * loss_global_smooth

                loss.backward()
                if self.args.clip_grad != None:
                    nn.utils.clip_grad_norm_(self.model.parameters(), max_norm = self.args.clip_grad)
                optimizer.step()
                batch_loss.append(loss.item())
            # print("一轮训练耗时:", time.time() - start_time, )
            total_loss = sum(batch_loss) / len(batch_loss)
            epoch_loss.append(total_loss)
            self.logger.info(f"Round {global_round} Client {self.idx} Epoch {iter} Loss {total_loss}")
        # self.save_model(self.args.logdir + '/client_' + str(self.idx) + '.pth')
        return self.model.state_dict(), sum(epoch_loss) / len(epoch_loss)

    def get_local_perturbed_output(self, feature, logits):
        adv_perturb = AdversarialPertubation(self.model.classifier, self.device, self.hparams)
        if self.perturb_type == 'weight':
            logits_perturb = adv_perturb.weight_perturb_predict(feature, logits)
        # else:
        #     raise NotImplementedError
        elif self.perturb_type == 'singular':
            logits_perturb = adv_perturb.singular_perturb_predict(feature, logits)
        return logits_perturb

    def get_global_perturbed_output(self, x, logits):
        with torch.no_grad():
            feature,_ = self.global_model(x)

        adv_perturb = AdversarialPertubation(self.global_model.classifier, self.device, self.hparams)
        if self.perturb_type == 'weight':
            logits_perturb = adv_perturb.weight_perturb_predict(feature, logits)
        # else:
        #     raise NotImplementedError
        elif self.perturb_type == 'singular':
            logits_perturb = adv_perturb.singular_perturb_predict(feature, logits)
        return logits_perturb


class AdversarialPertubation:
    def __init__(self, classifier, device, hparams):
        # self.featurizer = featurizer
        self.classifier = classifier

        # self.perturb_init_scale = 0.1
        # self.perturb_grad_scale = 0.01
        self.device = device
        self.hparams = hparams

    def weight_perturb_predict(self, feature, logits):
        # add perturbation to weight, and then forward
        # ----- step 1: generate random perturbation -----#
        pertub_layers = []
        for layer in self.classifier:
            if isinstance(layer, nn.Linear):

                weight = layer.weight.data

                # generate random perturbation
                if self.hparams['perturb_dist'] == 'uniform':
                    delta = torch.rand(weight.shape).sub(0.5).to(self.device)
                elif self.hparams['perturb_dist'] == 'normal':
                    delta = torch.randn(weight.shape).to(self.device)
                # normalize to unit ball
                delta = delta.div(torch.norm(delta, p=2, dim=1, keepdim=True) + 1e-8)
                # require grad
                delta.requires_grad = True

                # not perturb bias
                bias = layer.bias.data

                pertub_layers.append((weight, delta, bias))
            else:
                pertub_layers.append(layer)

        # ----- step 2: forward with perturbation -----#
        # z = self.featurizer(x)
        z = feature
        for layer in pertub_layers:
            if isinstance(layer, tuple):
                weight, delta, bias = layer
                z = F.linear(z, weight + self.hparams['perturb_init_scale'] * delta, bias)
            else:
                z = layer(z)
        logits_perturb = z

        # calculate KL div loss
        loss_kl = F.kl_div(F.log_softmax(logits_perturb, dim=1), F.softmax(logits, dim=1), reduction='batchmean')
        loss_kl.backward()

        # ----- step 3: forward with new perturbation -----#
        z = feature
        for layer in pertub_layers:
            if isinstance(layer, tuple):
                weight, delta, bias = layer
                grad = delta.grad
                grad = grad.div(torch.norm(grad, p=2, dim=1, keepdim=True) + 1e-8)
                z = F.linear(z, weight + self.hparams['perturb_grad_scale'] * grad, bias)
                # import pdb; pdb.set_trace()
            else:
                z = layer(z)
        logits_perturb = z
        # loss_kl = - F.kl_div(F.log_softmax(logits_perturb, dim=1), F.softmax(logits, dim=1), reduction='batchmean')
        return logits_perturb.detach()

    def singular_perturb_predict(self, feature, logits):
        # add perturbation to singular value of weight, and then forward
        # ----- step 1: generate random perturbation -----#
        pertub_layers = []
        for layer in self.classifier:
            if isinstance(layer, nn.Linear):

                weight = layer.weight.data

                # svd of weight
                U, S, Vh = torch.linalg.svd(weight, full_matrices=False)

                # generate random perturbation
                if self.hparams['perturb_dist'] == 'uniform':
                    delta = torch.rand(S.shape).sub(0.5).to(self.device)
                elif self.hparams['perturb_dist'] == 'normal':
                    delta = torch.randn(S.shape).to(self.device)
                # normalize to unit ball
                delta = delta.div(torch.norm(delta, p=2) + 1e-8)
                # require grad
                delta.requires_grad = True

                # not perturb bias
                bias = layer.bias.data

                pertub_layers.append((U, S, Vh, delta, bias))
            else:
                pertub_layers.append(layer)

        # ----- step 2: forward with perturbation -----#
        # z = self.featurizer(x)
        z = feature
        for layer in pertub_layers:
            if isinstance(layer, tuple):
                U, S, Vh, delta, bias = layer
                S_ = F.relu(S + self.hparams['perturb_init_scale'] * delta)
                weight = U @ torch.diag(S_) @ Vh
                z = F.linear(z, weight, bias)
            else:
                z = layer(z)
        logits_perturb = z

        # calculate KL div loss
        loss_kl = F.kl_div(F.log_softmax(logits_perturb, dim=1), F.softmax(logits, dim=1), reduction='batchmean')
        loss_kl.backward()

        # ----- step 3: forward with new perturbation -----#
        z = feature
        for layer in pertub_layers:
            if isinstance(layer, tuple):
                U, S, Vh, delta, bias = layer
                grad = delta.grad
                grad = grad.div(torch.norm(grad, p=2) + 1e-8)
                S_ = F.relu(S + self.hparams['perturb_grad_scale'] * grad)
                weight = U @ torch.diag(S_) @ Vh
                z = F.linear(z, weight, bias)
                # import pdb; pdb.set_trace()
            else:
                z = layer(z)
        logits_perturb = z
        # loss_kl = - F.kl_div(F.log_softmax(logits_perturb, dim=1), F.softmax(logits, dim=1), reduction='batchmean')
        return logits_perturb.detach()

