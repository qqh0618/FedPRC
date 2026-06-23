import torch
import torch.nn as nn
import torch.optim as optim
from utils import soft_predict
from core.Client.ClientBase import Client
import gc
class ClientFedMD(Client):
    """
    This class is for train the local model with input global model(copied) and output the updated weight
    args: argument 
    Loader_train,Loader_val,Loaders_test: input for training and inference
    user: the index of local model
    idxs: the index for data of this local model
    logger: log the loss and the process
    """
    def __init__(self, args, model, local_idx_dataidx_map, loader_pub,idx, logger, code_length, num_classes, device):
        super().__init__(args, model, local_idx_dataidx_map,idx, logger, code_length, num_classes, device)
        self.loader_pub = loader_pub
        
    def update_weights(self,global_round):
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
                _,p = self.model(X)
                loss = self.ce(p,y)               
                loss.backward()
                if self.args.clip_grad != None:
                    nn.utils.clip_grad_norm_(self.model.parameters(), max_norm = self.args.clip_grad)
                optimizer.step()
                batch_loss.append(loss.item())
            total_loss = sum(batch_loss) / len(batch_loss)
            epoch_loss.append(total_loss)
            self.logger.info(f"Round {global_round} Client {self.idx} Epoch {iter} Loss {total_loss}")
        self.save_model(self.args.logdir + '/client_' + str(self.idx) + '.pth')
        return self.model.state_dict(),sum(epoch_loss) / len(epoch_loss)

    def update_weights_MD(self,knowledges, lam, temp, global_round):
        self.model.to(self.device)
        self.model.train()
        epoch_loss = []
        global_soft_prediction =  torch.stack(knowledges)
        optimizer = optim.Adam(self.model.parameters(),lr=self.args.lr)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=self.args.lr_sh_rate, gamma=0.5)
        for iter in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (X, y) in enumerate(self.trainloader):
                X = X.to(self.device)
                y = y.to(self.device)
                optimizer.zero_grad()
                _,Z = self.model(X)
                loss1 = self.ce(Z,y)
                loss2 = torch.tensor(0.0).to(self.device)
                for idx, (X_pub,y_pub) in enumerate(self.loader_pub):
                    if idx == batch_idx:
                        X_pub = X_pub.to(self.device)
                        y_pub = y_pub.to(self.device)
                        _,Z_pub = self.model(X_pub)
                        Q_pub = soft_predict(Z_pub,temp).to(self.device)
                        loss2 -= self.kld(Q_pub,global_soft_prediction[idx].to(self.device))
                
                loss = loss1 + lam*loss2
                loss.backward()
                optimizer.step()
                batch_loss.append(loss.item())
            total_loss = sum(batch_loss) / len(batch_loss)
            epoch_loss.append(total_loss)
            self.logger.info(f"Round {global_round} Client {self.idx} Epoch {iter} Loss {total_loss}")
        self.save_model(self.args.logdir + '/client_' + str(self.idx) + '.pth')
        return self.model.state_dict(), sum(epoch_loss) / len(epoch_loss)
    
    def generate_knowledge(self, temp):
        self.model.to(self.device)
        self.model.eval()
        num_classes = self.model.num_classes
        soft_predictions = []
        for batch_idx, (X, y) in enumerate(self.loader_pub):
            X = X.to(self.device)
            y = y
            _,Z = self.model(X) 
            Q = soft_predict(Z,temp).to(self.device).detach().cpu()
            soft_predictions.append(Q)
            del X
            del y
            del Z
            del Q
            gc.collect()
         
        return soft_predictions