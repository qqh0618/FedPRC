import copy
from core.Server.ServerBase import Server
from core.Client.ClientFedProto import ClientFedProto
from tqdm import tqdm
import numpy as np
from utils import average_weights
from mem_utils import MemReporter
import time
import gc

class ServerFedProto(Server):
    def __init__(self, args, global_model,Loader_train,Loaders_local_test,Loader_global_test,logger,device, net_idx_dataidx_map):
        super().__init__(args, global_model,Loader_train,Loaders_local_test,Loader_global_test,logger,device, net_idx_dataidx_map)

    
    def Create_Clints(self):
        for idx in range(self.args.num_clients):
            self.LocalModels.append(ClientFedProto(self.args, copy.deepcopy(self.global_model), self.net_idx_dataidx_map[idx], idx=idx, logger=self.logger, code_length = self.args.code_len, num_classes = self.args.num_classes, device=self.device))
            
    def global_knowledge_aggregation(self, features):
        global_local_features = dict()
        for [label, features] in features.items():
            if len(features) > 1:
                feature = 0 * features[0].data
                for i in features:
                    feature += i.data
                global_local_features[label] = [feature / len(features)]
            else:
                global_local_features[label] = [features[0].data]

 
        return global_local_features

    def proto_aggregation(self, local_protos_list):
        agg_protos_label = dict()
        for idx in range(len(local_protos_list)):
            local_protos = local_protos_list[idx]
            for label in local_protos.keys():
                if label in agg_protos_label:
                    agg_protos_label[label].append(local_protos[label])
                else:
                    agg_protos_label[label] = [local_protos[label]]

        for [label, proto_list] in agg_protos_label.items():
            if len(proto_list) > 1:
                proto = 0 * proto_list[0][0]
                for i in proto_list:
                    proto += i[0]
                agg_protos_label[label] = [proto / len(proto_list)]
            else:
                agg_protos_label[label] = proto_list[0]

        return agg_protos_label

    def train(self):
        global_features = {}
        reporter = MemReporter()
        start_time = time.time()
        train_loss = []
        global_weights = self.global_model.state_dict()
        for epoch in tqdm(range(self.args.comm_round)):
            Knowledges = []
            test_accuracy = 0
            local_weights, local_losses = [], []
            self.logger.info(f'\n | Global Training Round : {epoch+1} |\n')
            m = max(int(self.args.sampling_rate * self.args.num_clients), 1)
            idxs_users = np.random.choice(range(self.args.num_clients), m, replace=False)
            for idx in idxs_users:
                if self.args.upload_model == True:
                    self.LocalModels[idx].load_model(global_weights)
                if epoch < 1:        
                    w, loss = self.LocalModels[idx].update_weights(global_round=epoch)
                    local_losses.append(copy.deepcopy(loss))
                    local_weights.append(copy.deepcopy(w))
                    acc = self.LocalModels[idx].test_accuracy()
                    test_accuracy += acc
                    
                else:
                    w, loss = self.LocalModels[idx].update_weights_Proto(global_round=epoch, global_features=copy.deepcopy(global_features), gamma = self.args.gamma)
                    local_losses.append(copy.deepcopy(loss))
                    local_weights.append(copy.deepcopy(w))
                    acc = self.LocalModels[idx].test_accuracy()
                    test_accuracy += acc
                    
                local_features  = self.LocalModels[idx].generate_knowledge()
                Knowledges.append(local_features)


             # update global weights
            global_weights = self.average_weights(local_weights, idxs_users)
            global_features = self.proto_aggregation(Knowledges)
            self.global_model.load_state_dict(global_weights)
            loss_avg = sum(local_losses) / len(local_losses)
            train_loss.append(loss_avg)
            cur_g_acc = self.global_test_accuracy()
            if cur_g_acc > self.global_best_acc:
                self.global_best_acc = cur_g_acc
                self.Save_CheckPoint(self.args.logdir + '/best_model.pth')
            if test_accuracy / len(idxs_users)  > self.global_best_personal_acc:
                self.global_best_personal_acc = test_accuracy / len(idxs_users)
            self.logger.info(f'Global Training Loss: {loss_avg}')
            self.logger.info(f'Personal_Accuracy: {test_accuracy / len(idxs_users) } || Best_Personal_Accuracy: {self.global_best_personal_acc}')
            self.logger.info(f'Global_Accuracy: {cur_g_acc} || Best_Accuracy: {self.global_best_acc}')
        self.test_domain()
        self.logger.info('Training is completed.')
        end_time = time.time()
        self.logger.info(f'Total running time: {end_time - start_time} s || avg: {(end_time - start_time) / self.args.comm_round} s')
        reporter.report()