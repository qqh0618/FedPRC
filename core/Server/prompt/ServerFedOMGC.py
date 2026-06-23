import copy

import torch
from core.Client.prompt.ClientFedOMGC import ClientFedOMGC
from torch.nn.utils import vector_to_parameters
from core.Server.ServerBase import Server
from tqdm import tqdm
import numpy as np
from utils import average_weights, Accuracy
from mem_utils import MemReporter
import time
from numbers import Number
from collections import OrderedDict
import operator

class ParamDict(OrderedDict):
    """Code adapted from https://github.com/Alok/rl_implementations/tree/master/reptile.
    A dictionary where the values are Tensors, meant to represent weights of
    a model. This subclass lets you perform arithmetic on weights directly."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, *kwargs)

    def _prototype(self, other, op):
        if isinstance(other, Number):
            return ParamDict({k: op(v, other) for k, v in self.items()})
        elif isinstance(other, dict):
            return ParamDict({k: op(self[k], other[k]) for k in self})
        else:
            raise NotImplementedError

    def __add__(self, other):
        return self._prototype(other, operator.add)

    def __rmul__(self, other):
        return self._prototype(other, operator.mul)

    __mul__ = __rmul__

    def __neg__(self):
        return ParamDict({k: -v for k, v in self.items()})

    def __rsub__(self, other):
        # a- b := a + (-b)
        return self.__add__(other.__neg__())

    __sub__ = __rsub__

    def __truediv__(self, other):
        return self._prototype(other, operator.truediv)


class ServerFedOMGC(Server):
    def __init__(self, args, global_model, Loader_train, Loaders_local_test, Loader_global_test, logger, device,
                 net_idx_dataidx_map):
        super().__init__(args, global_model, Loader_train, Loaders_local_test, Loader_global_test, logger, device,
                         net_idx_dataidx_map)

        # 冻结模型参数
        for name, param in self.global_model.named_parameters():
            # print(name,":",param.size())
            if "prompt_learner" not in name:
                param.requires_grad_(False)
            else:
                print(name)

        self.global_model_lr = 0.5
        self.parameter_c = 0.5


    def Create_Clints(self):
        for idx in range(self.args.num_clients):
            self.LocalModels.append(
                ClientFedOMGC(self.args, copy.deepcopy(self.global_model), self.net_idx_dataidx_map[idx], idx=idx,
                                logger=self.logger, code_length=self.args.code_len, num_classes=self.args.num_classes,
                                device=self.device))

    def train(self):
        reporter = MemReporter()
        start_time = time.time()
        train_loss = []

        for epoch in tqdm(range(self.args.comm_round)):
            test_accuracy = 0
            local_weights, local_losses = [], []
            self.logger.info(f'Global Training Round: {epoch + 1}')
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
            if test_accuracy / len(idxs_users) > self.global_best_personal_acc:
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
        self.logger.info(
            f'Total running time: {end_time - start_time} s || avg: {(end_time - start_time) / self.args.comm_round} s')
        reporter.report()

    def average_weights(self, w, idxs_users):
        global_model = self.FedOMG(self.global_model,self.global_model_lr,self.parameter_c,idxs_users)
        return global_model
    def FedOMG(self,global_model, global_lr, cagrad_c, idxs_users):
        """

        :param model_dict:
        :param weight_dict:
        :param global_model:
        :param dataobj:
        :param global_lr:
        :param cagrad_c:
        :return:
        """
        print("\nFedOMG")

        total_data_points = sum([len(self.LocalModels[r].train_idx) for r in idxs_users])

        fed_avg_freqs = [len(self.LocalModels[r].train_idx) / total_data_points for r in idxs_users]

        all_domain_grads = []
        flatten_global_weights = torch.cat([param.view(-1) for param in global_model.parameters()])

        for i, r in enumerate(idxs_users):
            # domain_grad_diff = [torch.flatten(grad_param - global_param) * fed_avg_freqs[i] for
            #                     grad_param, global_param in
            #                     zip(self.LocalModels[r].model.parameters(), global_model.parameters())]

            domain_grad_diff = []

            for (grad_name, grad_param), (global_name,global_param) in zip(self.LocalModels[r].model.named_parameters(), global_model.named_parameters()):
                if 'prompt_learner' not in grad_name:
                    continue
                domain_grad_diff.append(torch.flatten(grad_param - global_param) * fed_avg_freqs[i])

            domain_grad_vector = torch.cat(domain_grad_diff)
            all_domain_grads.append(domain_grad_vector)

        all_domain_grads_tensor = torch.stack(all_domain_grads)

        omg_grads = self.OMG(all_domain_grads_tensor, 3, cagrad_c)
        flatten_global_weights += omg_grads * global_lr

        vector_to_parameters(flatten_global_weights, global_model.parameters())

        global_model = ParamDict(global_model.state_dict())

        return global_model

    def OMG(self, grad_vec, num_tasks, cagrad_c):
        """
        grad_vec: [num_tasks, dim]
        """
        # if self.args.dataset == 'office31':
        #     num_tasks = 2*self.args.meta_num
        # elif self.args.dataset == "domainnet":
        #     num_tasks = 5*self.args.meta_num
        # else:
        #     num_tasks = 3*self.args.meta_num

        num_tasks = self.args.num_clients

        grads = grad_vec
        # print(grads.size())
        GG = grads.mm(grads.t()).cpu()
        scale = (torch.diag(GG) + 1e-4).sqrt().mean()
        GG = GG / scale.pow(2)
        Gg = GG.mean(1, keepdims=True)
        gg = Gg.mean(0, keepdims=True)

        # print(GG.size())
        # print(Gg.size())
        # print(gg.size())

        w = torch.zeros(num_tasks, 1, requires_grad=True).half()
        if num_tasks == 50:
            w_opt = torch.optim.SGD([w], lr=50, momentum=0.5)
        else:
            w_opt = torch.optim.SGD([w], lr=25, momentum=0.5)

        c = (gg + 1e-4).sqrt() * cagrad_c

        w_best = None
        obj_best = np.inf
        for i in range(21):
            w_opt.zero_grad()
            ww = torch.softmax(w, 0)
            obj = ww.t().mm(Gg) + c * (ww.t().mm(GG).mm(ww) + 1e-4).sqrt()
            if obj.item() < obj_best:
                obj_best = obj.item()
                w_best = w.clone()
            if i < 20:
                obj.backward(retain_graph=True)
                w_opt.step()

        ww = torch.softmax(w_best, 0)
        gw_norm = (ww.t().mm(GG).mm(ww) + 1e-4).sqrt()

        lmbda = c.view(-1) / (gw_norm + 1e-4)
        g = ((1 / num_tasks + ww * lmbda).view(
            -1, 1).to(grads.device) * grads).sum(0) / (1 + cagrad_c ** 2)
        return g
