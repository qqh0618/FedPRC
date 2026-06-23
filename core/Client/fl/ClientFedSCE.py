import copy
import time

import torch
import torch.nn as nn
import torch.optim as optim
from core.Client.ClientBase import Client
import torch.nn.functional as F


class ClientFedSCE(Client):
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
        # ==============================
        # FedSCE, 脱裤子放屁一样
        self.lamda = 0.1
        self.layer_idx = 30
        self.rank = 3
        self.gap = 5
        self.KL = nn.KLDivLoss()


        # ==============================
    def update_weights(self, global_round):   # 训练模型
        self.model.to(self.device)
        self.model.train()
        self.global_model.train()
        epoch_loss = []
        optimizer = self.optim

        optimizer_g = torch.optim.SGD(self.global_model.parameters(), lr=self.args.lr)

        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=self.args.lr_sh_rate, gamma=0.5)
        self.trainloader = self.get_trainloader()
        # self.freeze_model()
        # ==============================
        # FedSCE
        global_model_before = copy.deepcopy(self.global_model)
        projection = self.initialize_projection_mat(self.model)
        prev_params = [p.clone() for p in self.model.parameters()]
        # ==============================

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
                optimizer_g.zero_grad()

                _,output = self.model(X)

                CE_loss = self.ce(output,y)
                # ==================================
                _, output_g = self.global_model(X)
                CE_loss_g = self.ce(output_g, y)

                L_d = self.KL(F.log_softmax(output, dim=1), F.softmax(output_g, dim=1)) / (CE_loss + CE_loss_g)
                L_d_g = self.KL(F.log_softmax(output_g, dim=1), F.softmax(output, dim=1)) / (CE_loss + CE_loss_g)

                loss = CE_loss + L_d
                loss_g = CE_loss_g + L_d_g
                reg_loss = self.projection_mat_loss(self.model, prev_params, projection)
                loss += reg_loss * self.lamda
                # ==================================
                loss.backward(retain_graph=True)
                loss_g.backward()
                if self.args.clip_grad != None:
                    nn.utils.clip_grad_norm_(self.model.parameters(), max_norm = self.args.clip_grad)
                optimizer.step()
                optimizer_g.step()

                # optimizer_g.zero_grad()

                batch_loss.append(loss.item())
            # print("一轮训练耗时:", time.time() - start_time, )
            total_loss = sum(batch_loss) / len(batch_loss)
            epoch_loss.append(total_loss)
            self.logger.info(f"Round {global_round} Client {self.idx} Epoch {iter} Loss {total_loss}")

        self.calculate_F1_F2(global_model_before, self.global_model, self.trainloader)

        self.global_model_before = copy.deepcopy(self.global_model)
        self.prev_params = [p.clone() for p in self.model.parameters()]

        self.save_model(self.args.logdir + '/client_' + str(self.idx) + '.pth')
        return self.model.state_dict(), sum(epoch_loss) / len(epoch_loss)

    # def freeze_model(self):
    #     for name, param in self.model.named_parameters():
    #         if "bn" in name:
    #             param.requires_grad = False

    def initialize_projection_mat(self, model):

        params = list(model.parameters())
        total_dim = 0

        for param in params[-self.layer_idx:]:
            total_dim += param.numel()

        random_matrix = torch.randn(total_dim, self.rank, device=self.device)
        projection = random_matrix.to(self.device)

        return projection

    def projection_mat_loss(self, model, prev_params, projection):

        delta_w_combined = []
        model_params = list(model.parameters())

        for param, prev_param in zip(model_params[-self.layer_idx:], prev_params[-self.layer_idx:]):
            if param.requires_grad:
                delta_w = (param - prev_param).reshape(-1)
                delta_w_combined.append(delta_w)

        delta_w_combined = torch.cat(delta_w_combined, dim=0)

        delta_w_proj = projection @ (projection.T @ delta_w_combined)

        reg_loss = torch.norm(delta_w_combined - delta_w_proj) ** 2

        return reg_loss

    def calculate_F1_F2(self, global_model_before, global_model_after, trainloader):
        param_differences = [(p_after - p_before).norm(2) for p_before, p_after in
                             zip(global_model_before.parameters(), global_model_after.parameters())]
        self.F1 = torch.sqrt(sum([diff ** 2 for diff in param_differences]))

        global_model_before.eval()
        global_model_after.eval()
        feature_differences = []
        with torch.no_grad():
            for x, _ in trainloader:
                x = x.to(self.device)

                feature_before,_ = global_model_before(x)
                feature_after,_ = global_model_after(x)

                feature_differences.append((feature_after - feature_before).norm(2))

        self.F2 = torch.sqrt(sum([diff ** 2 for diff in feature_differences]))