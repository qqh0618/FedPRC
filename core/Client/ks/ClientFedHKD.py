import torch
import copy
import torch.nn as nn
import torch.optim as optim
from utils import soft_predict
from core.Client.ClientBase import Client
import gc
class ClientFedHKD(Client):
    """
    This class is for train the local model with input global model(copied) and output the updated weight
    args: argument 
    Loader_train,Loader_val,Loaders_test: input for training and inference
    user: the index of local model
    idxs: the index for data of this local model
    logger: log the loss and the process
    """
    def __init__(self, args, model, local_idx_dataidx_map,idx, logger, code_length, num_classes, device):
        super().__init__(args, model, local_idx_dataidx_map,idx, logger, code_length, num_classes, device)
    
    
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

    
    def update_weights_HKD(self,global_features, global_soft_prediction, lam, gamma, temp, global_round):
        self.model.to(self.device)
        self.model.train()
        epoch_loss = []
        optimizer = optim.Adam(self.model.parameters(),lr=self.args.lr)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=self.args.lr_sh_rate, gamma=0.5)
        tensor_global_features = self.dict_to_tensor(global_features).to(self.device)
        tensor_global_soft_prediction = self.dict_to_tensor(global_soft_prediction).to(self.device)
        for iter in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (X, y) in enumerate(self.trainloader):
                X = X.to(self.device)
                y = y.to(self.device)
                optimizer.zero_grad()
                F,Z = self.model(X)
                Z_help = self.model.classifier(tensor_global_features)
                Q_help = soft_predict(Z_help,temp).to(self.device)
                loss1 = self.ce(Z,y)
                target_features = copy.deepcopy(F.data)

                
                for i in range(y.shape[0]):
                    if int(y[i]) in global_features.keys():
                        target_features[i] = global_features[int(y[i])][0].data
    
                        
                target_features = target_features.to(self.device)
                if len(global_features) == 0:
                    loss2 = 0*loss1
                    loss3 = 0*loss1
                else:
                    loss2 = self.kld(Q_help.log(),tensor_global_soft_prediction)
                    loss3 = self.mse(F,target_features)
                loss = loss1 + lam*loss2 + gamma*loss3
                loss.backward()
                if self.args.clip_grad != None:
                    nn.utils.clip_grad_norm_(self.model.parameters(), max_norm = self.args.clip_grad)
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm =1.1)
                optimizer.step()
                batch_loss.append(loss.item())
            total_loss = sum(batch_loss) / len(batch_loss)
            epoch_loss.append(total_loss)
            self.logger.info(f"Round {global_round} Client {self.idx} Epoch {iter} Loss {total_loss}")
        # self.save_model(self.args.logdir + '/client_hkd_' + str(self.idx) + '.pth')
                        
        return self.model.state_dict(), sum(epoch_loss) / len(epoch_loss)
    
    # generate knowledge for FedDFKD
    def generate_knowledge(self, temp):
        self.model.to(self.device)
        self.model.eval()        
        local_features = {}
        local_soft_prediction = {}
        num_classes = self.model.num_classes
        features = [torch.zeros(self.code_length).to(self.device)]*num_classes
        soft_predictions = [torch.zeros(num_classes).to(self.device)]*num_classes
        count = [0]*num_classes
        for batch_idx, (X, y) in enumerate(self.trainloader):
            X = X.to(self.device)
            y = y
            F,Z = self.model(X)
            Q = soft_predict(Z,temp).to(self.device)
            m = y.shape[0]
            for i in range(len(y)):
                if y[i].item() in local_features:
                    local_features[y[i].item()].append(F[i,:])
                    local_soft_prediction[y[i].item()].append(Q[i,:])
                else:
                    local_features[y[i].item()] = [F[i,:]]
                    local_soft_prediction[y[i].item()]  = [Q[i,:]] 
            del X
            del y
            del F
            del Z
            del Q
            gc.collect()
            
        features,soft_predictions = self.local_knowledge_aggregation(local_features,local_soft_prediction, std = self.args.std)
        
        return (features,soft_predictions)
    
    def local_knowledge_aggregation(self,local_features,local_soft_prediction, std):
        agg_local_features = dict()
        agg_local_soft_prediction = dict()
        feature_noise = std*torch.randn(self.args.code_len).to(self.device)
        for [label, features] in local_features.items():
            if len(features) > 1:
                feature = 0 * features[0].data
                for i in features:
                    feature += i.data   
                agg_local_features[label] = [feature / len(features) + feature_noise]
            else:
                agg_local_features[label] = [features[0].data + feature_noise]
                
        for [label, soft_prediction] in local_soft_prediction.items():
            if len(soft_prediction) > 1:
                soft = 0 * soft_prediction[0].data
                for i in soft_prediction:
                    soft += i.data

                agg_local_soft_prediction[label] = [soft / len(soft_prediction) ]
            else:
                agg_local_soft_prediction[label] = [soft_prediction[0].data]
                
        return agg_local_features,agg_local_soft_prediction
    
    def dict_to_tensor(self, dic):
        lit = []
        for key,tensor in dic.items():
            lit.append(tensor[0])
        lit = torch.stack(lit)
        return lit