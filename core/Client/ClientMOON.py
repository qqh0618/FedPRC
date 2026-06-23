import copy
import time
import torch
import torch.nn as nn
import torch.optim as optim
from core.Client.ClientBase import Client

class ClientMOON(Client):
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
        self.old_model = copy.deepcopy(self.model)
        self.cos = torch.nn.CosineSimilarity(dim=-1)

    def update_weights(self, global_round):   # 训练模型
        self.model.to(self.device)
        self.model.train()
        epoch_loss = []
        optimizer = self.sgd
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=self.args.lr_sh_rate, gamma=0.5)

        # ======================================
        self.global_model = copy.deepcopy(self.model)
        self.frezze_model(self.global_model)
        self.frezze_model(self.old_model)
        # ======================================


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
                f, p = self.model(X)
                g_f, g_p = self.global_model(X)

                posi = self.cos(f, g_f)
                logits = posi.reshape(-1, 1)

                for previous_net in [self.old_model]:
                    previous_net.cuda()
                    pro3, _ = previous_net(X)
                    nega = self.cos(f, pro3)
                    logits = torch.cat((logits, nega.reshape(-1, 1)), dim=1)

                    previous_net.to('cpu')

                logits /= 0.5
                labels = torch.zeros(X.size(0)).cuda().long()

                loss2 = self.ce(logits, labels)

                loss = self.ce(p,y) + loss2
                loss.backward()
                if self.args.clip_grad != None:
                    nn.utils.clip_grad_norm_(self.model.parameters(), max_norm = self.args.clip_grad)
                optimizer.step()
                batch_loss.append(loss.item())
            # print("一轮训练耗时:", time.time() - start_time, )
            total_loss = sum(batch_loss) / len(batch_loss)
            epoch_loss.append(total_loss)
            self.logger.info(f"Round {global_round} Client {self.idx} Epoch {iter} Loss {total_loss}")

        self.old_model = copy.deepcopy(self.model)
        self.save_model(self.args.logdir + '/client_' + str(self.idx) + '.pth')
        return self.model.state_dict(), sum(epoch_loss) / len(epoch_loss)

    def frezze_model(self, model):
        model.eval()
        for param in model.parameters():
            param.requires_grad = False