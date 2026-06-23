"""
2025/4/30 18:57
while True:
    leanring
本文件由my_ywj首次创建编写
"""
import torch
import torch.nn as nn
import torch.optim as optim
from core.Client.ClientBase import Client


class ClientFedGA(Client):
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
        # ==================
        # FedDG-GA
        self.metric = None
        # ==================
    def update_weights(self, global_round):  # 训练模型
        self.model.to(self.device)
        self.model.train()
        epoch_loss = []
        optimizer = self.optim
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=self.args.lr_sh_rate, gamma=0.5)
        self.trainloader = self.get_trainloader()
        for iter in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (X, y) in enumerate(self.trainloader):
                X = X.to(self.device)
                y = y.to(self.device)
                optimizer.zero_grad()
                _, p = self.model(X)
                loss = self.ce(p, y)
                loss.backward()
                if self.args.clip_grad != None:
                    nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.args.clip_grad)
                optimizer.step()
                # ======================
                # FedDG-GA
                self.metric.update(p, y)
                # ======================
                batch_loss.append(loss.item())
            total_loss = sum(batch_loss) / len(batch_loss)
            epoch_loss.append(total_loss)
            self.logger.info(f"Round {global_round} Client {self.idx} Epoch {iter} Loss {total_loss}")
        # self.save_model(self.args.logdir + '/client_' + str(self.idx) + '.pth')
        return self.model.state_dict(), sum(epoch_loss) / len(epoch_loss)

    def load_metrics(self, metric):
        self.metric = metric

    def site_evaluation(self):
        # 客户端测试
        self.model.eval()
        with torch.no_grad():
            for imgs, labels, in self.testloader:
                imgs = imgs.cuda()
                _,output = self.model(imgs)
                self.metric.update(output, labels)
        results_dict = self.metric.results()

        return results_dict