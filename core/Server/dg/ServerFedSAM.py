import copy

import torch
from collections import OrderedDict

from torch import optim

from core.Client.dg.ClientFedSAM import ClientFedSAM
from core.Server.ServerBase import Server
from core.Client.ClientFedAvg import ClientFedAvg
from tqdm import tqdm
import numpy as np
from utils import average_weights
from mem_utils import MemReporter
import time

class ServerFedSAM(Server):
    def __init__(self, args, global_model,Loader_train,Loaders_local_test,Loader_global_test,logger,device, net_idx_dataidx_map):
        super().__init__(args, global_model,Loader_train,Loaders_local_test,Loader_global_test,logger,device, net_idx_dataidx_map)
        # =======================
        self.server_lr = self.args.lr
        self.server_momentum = 0.9
        self.server_opt = optim.SGD(params=self.global_model.parameters(), lr=self.server_lr, momentum=0.9)
        # if opt_ckpt is not None:
        #     self.load_optimizer_checkpoint(opt_ckpt)
        # =======================


    def Create_Clints(self):
        for idx in range(self.args.num_clients):
            self.LocalModels.append(ClientFedSAM(self.args, copy.deepcopy(self.global_model),self.net_idx_dataidx_map[idx], idx=idx, logger=self.logger, code_length = self.args.code_len, num_classes = self.args.num_classes, device=self.device))
            
            
    def train(self):
        reporter = MemReporter()
        start_time = time.time()
        train_loss = []
        global_weights = self.global_model.state_dict()
        for epoch in tqdm(range(self.args.comm_round)):
            test_accuracy = 0
            local_weights, local_losses = [], []
            self.logger.info(f'Global Training Round: {epoch+1}')
            m = max(int(self.args.sampling_rate * self.args.num_clients), 1)
            idxs_users = np.random.choice(range(self.args.num_clients), m, replace=False)
            for idx in idxs_users:
                if self.args.upload_model == True:
                    self.LocalModels[idx].load_model(global_weights)
                w, loss = self.LocalModels[idx].update_weights(global_round=epoch)
                local_losses.append(copy.deepcopy(loss))
                local_weights.append(copy.deepcopy(w))
                acc = self.LocalModels[idx].test_accuracy()
                test_accuracy += acc

            # =================
            # FedSAM
            self._save_updates_as_pseudogradients(local_weights)
            # =================
            # update global weights
            global_weights = self.average_weights(local_weights, idxs_users)

            # =========================
            # FedSAM
            self._update_global_model_gradient(global_weights)
            # =========================

            self.global_model.load_state_dict(global_weights)

            # ================
            # FedSAM
            self.total_grad = self._get_model_total_grad()
            # ================
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

        self.test_domain()
        self.logger.info('Training is completed.')
        end_time = time.time()
        self.logger.info(f'Total running time: {end_time - start_time} s || avg: {(end_time - start_time) / self.args.comm_round} s')
        reporter.report()

    def _save_updates_as_pseudogradients(self, weights):
        self.updates = weights
        clients_models = copy.deepcopy(self.updates)
        self.updates = []
        for i, update in enumerate(clients_models):
            delta = self._compute_client_delta(update)
            self.updates.append(delta)

    def _compute_client_delta(self, cmodel):
        """Args:
            cmodel: client update, i.e. state dict of client's update.
        Returns:
            delta: delta between client update and global model. """
        delta = OrderedDict.fromkeys(cmodel.keys())  # (delta x_i)^t
        for k, x, y in zip(self.global_model.state_dict().keys(), self.global_model.state_dict().values(), cmodel.values()):
            delta[k] = y - x if "running" not in k and "num_batches_tracked" not in k else y
        return delta

    def _update_global_model_gradient(self, pseudo_gradient):
        """Args:
            pseudo_gradient: global pseudo gradient, i.e. weighted average of the trained clients' deltas.

        Updates the global model gradient as -1.0 * pseudo_gradient
        """
        for n, p in self.global_model.named_parameters():
            p.grad = -1.0 * pseudo_gradient[n]

        self.server_opt.step()

        bn_layers = OrderedDict(
            {k: v for k, v in pseudo_gradient.items() if "running" in k or "num_batches_tracked" in k})
        self.global_model.load_state_dict(bn_layers, strict=False)

    def _get_model_total_grad(self):
        """Returns:
            total_grad: sum of the L2-norm of the gradient of each trainable parameter"""
        total_norm = 0
        for name, p in self.global_model.named_parameters():
            if p.requires_grad:
                try:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
                except Exception:
                    # this param had no grad
                    pass
        total_grad = total_norm ** 0.5
        # print("total grad norm:", total_grad)
        return total_grad