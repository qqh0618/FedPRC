
from torch.utils.data import Dataset
import torch
import copy
from utils import Accuracy
import pandas as pd

class Server(object):
    def __init__(self,args, global_model,Loaders_train, Loaders_local_test, Loader_global_test, logger, device, net_idx_dataidx_map):
        self.global_model = global_model
        self.args = args
        self.Loaders_train = Loaders_train
        self.Loaders_local_test = Loaders_local_test
        self.global_testloader = Loader_global_test
        self.logger = logger
        self.device = device
        self.LocalModels = []
        self.global_best_acc = 0
        self.global_best_personal_acc = 0
        self.num_classes = self.args.num_classes
        self.net_idx_dataidx_map = net_idx_dataidx_map
        # 定义要记录的指标（例如：损失值、准确率等）
        columns = ['DataName', 'Round', 'Loss', 'Personal_Acc', 'Global_Acc']
        # 创建空的 DataFrame
        self.df = pd.DataFrame(columns=columns)
        # =================
        # 建一个csv记录全局测试准确率


        
    def global_test_accuracy(self):

        if self.args.model=='clip':
            all_domain_acc = []
            for domain, dataloader in self.global_testloader.items():
                accuracy = 0
                cnt = 0
                domain_acc = []

                for batch_idx, (X, y) in enumerate(dataloader):
                    X = X.to(self.device)
                    y = y.to(self.device)
                    # X = X.half()
                    p,_,_ = self.global_model(X)
                    y_pred = p.argmax(1)

                    accuracy += Accuracy(y, y_pred)
                    cnt += 1
                domain_acc.append(accuracy / cnt)
                self.logger.info(f'|| Server Domain:: {domain} Accuracy: {accuracy / cnt} ||')
                # self.logger.info(f'|| Server Domain Avg: {sum(domain_acc) / len(domain_acc)}')
                all_domain_acc.append(accuracy / cnt)

            self.logger.info(f'|| Server Domain Avg: {sum(all_domain_acc) / len(all_domain_acc)}')
            return sum(all_domain_acc) / len(all_domain_acc)

        if not self.args.domain:
            # 传统联邦学习
            self.global_model.eval()
            accuracy = 0
            cnt = 0
            for batch_idx, (X, y) in enumerate(self.global_testloader):
                X = X.to(self.device)
                # X = X.half()
                y = y.to(self.device)
                _, p = self.global_model(X)
                y_pred = p.argmax(1)
                accuracy += Accuracy(y,y_pred)
                cnt += 1
            return accuracy/cnt
        elif self.args.alg =='StableFDG':
            self.global_model.eval()
            accuracy = 0
            cnt = 0
            domain_acc = []
            for domain, dataloader in self.global_testloader.items():
                for batch_idx, (X, y) in enumerate(dataloader):
                    X = X.to(self.device)
                    y = y.to(self.device)
                    p = self.global_model(X,y)
                    # y_pred = p.argmax(1)
                    if isinstance(p, tuple):
                        pred = p[0].max(1)[1]
                    else:
                        pred = p.max(1)[1]
                    accuracy += Accuracy(y,pred)
                    cnt += 1
                domain_acc.append(accuracy / cnt)
                self.logger.info(f'|| Server Domain: {domain} Accuracy: {accuracy / cnt} ||')
            self.logger.info(f'|| Server Domain Avg: {sum(domain_acc) / len(domain_acc)}')
            return sum(domain_acc)/len(domain_acc)
        else:
            # 域适应
            self.global_model.eval()
            accuracy = 0
            cnt = 0
            domain_acc = []
            if self.args.leave_domain:
                if ('VDDA' in self.args.alg) or ('FedETF'==self.args.alg):
                    # 域泛化
                    # for domain, dataloader in self.global_testloader.items():
                    for batch_idx, (X, y) in enumerate(self.global_testloader):
                        X = X.to(self.device)
                        y = y.to(self.device)
                        z, p = self.global_model(X)
                        # y_pred = p.argmax(1)
                        z_norm = torch.div(z, torch.norm(z, p=2, dim=1, keepdim=True))
                        output_local = torch.matmul(z_norm, self.global_model.proto_classifier.proto.to(self.device))
                        p = self.global_model.scaling * output_local
                        y_pred = p.argmax(1)

                        accuracy += Accuracy(y, y_pred)
                        cnt += 1
                    domain_acc.append(accuracy / cnt)
                    self.logger.info(f'|| Server Domain: {self.args.test_domain} Accuracy: {accuracy / cnt} ||')
                    # self.logger.info(f'|| Server Domain Avg: {sum(domain_acc) / len(domain_acc)}')
                    return sum(domain_acc) / len(domain_acc)
                else:
                    for batch_idx, (X, y) in enumerate(self.global_testloader):
                        X = X.to(self.device)
                        y = y.to(self.device)
                        # X = X.half()
                        z, p = self.global_model(X)
                        # y_pred = p.argmax(1)
                        # z_norm = torch.div(z, torch.norm(z, p=2, dim=1, keepdim=True))
                        # output_local = torch.matmul(z_norm, self.global_model.proto_classifier.proto.to(self.device))
                        # p = self.global_model.scaling * output_local
                        y_pred = p.argmax(1)

                        accuracy += Accuracy(y, y_pred)
                        cnt += 1
                    domain_acc.append(accuracy / cnt)
                    self.logger.info(f'|| Server Domain: {self.args.test_domain} Accuracy: {accuracy / cnt} ||')
                    # self.logger.info(f'|| Server Domain Avg: {sum(domain_acc) / len(domain_acc)}')
                    return sum(domain_acc) / len(domain_acc)

            elif self.args.train_one:
                all_domain_acc = []
                for domain, dataloader in self.global_testloader.items():
                    accuracy = 0
                    cnt = 0
                    domain_acc = []
                    if 'VDDA' in self.args.alg:
                        # 域泛化
                        # for domain, dataloader in self.global_testloader.items():
                        for batch_idx, (X, y) in enumerate(dataloader):
                            X = X.to(self.device)
                            y = y.to(self.device)
                            z, p = self.global_model(X)
                            # y_pred = p.argmax(1)
                            z_norm = torch.div(z, torch.norm(z, p=2, dim=1, keepdim=True))
                            output_local = torch.matmul(z_norm, self.global_model.proto_classifier.proto.to(self.device))
                            p = self.global_model.scaling * output_local
                            y_pred = p.argmax(1)

                            accuracy += Accuracy(y, y_pred)
                            cnt += 1
                        domain_acc.append(accuracy / cnt)
                        self.logger.info(f'|| Server Domain:: {domain} Accuracy: {accuracy / cnt} ||')
                        # self.logger.info(f'|| Server Domain Avg: {sum(domain_acc) / len(domain_acc)}')
                        all_domain_acc.append(accuracy / cnt)
                    else:
                        for batch_idx, (X, y) in enumerate(dataloader):
                            X = X.to(self.device)
                            y = y.to(self.device)
                            # X = X.half()
                            z, p = self.global_model(X)
                            # y_pred = p.argmax(1)
                            # z_norm = torch.div(z, torch.norm(z, p=2, dim=1, keepdim=True))
                            # output_local = torch.matmul(z_norm, self.global_model.proto_classifier.proto.to(self.device))
                            # p = self.global_model.scaling * output_local
                            y_pred = p.argmax(1)

                            accuracy += Accuracy(y, y_pred)
                            cnt += 1
                        domain_acc.append(accuracy / cnt)
                        self.logger.info(f'|| Server Domain:: {domain} Accuracy: {accuracy / cnt} ||')
                        # self.logger.info(f'|| Server Domain Avg: {sum(domain_acc) / len(domain_acc)}')
                        all_domain_acc.append(accuracy / cnt)

                self.logger.info(f'|| Server Domain Avg: {sum(all_domain_acc) / len(all_domain_acc)}')
                return sum(all_domain_acc) / len(all_domain_acc)

            else:
                for domain, dataloader in self.global_testloader.items():
                    for batch_idx, (X, y) in enumerate(dataloader):
                        X = X.to(self.device)
                        y = y.to(self.device)
                        # X = X.half()
                        _, p = self.global_model(X)
                        y_pred = p.argmax(1)
                        accuracy += Accuracy(y,y_pred)
                        cnt += 1
                    domain_acc.append(accuracy/cnt)
                    self.logger.info(f'|| Server Domain: {domain} Accuracy: {accuracy/cnt} ||')
                self.logger.info(f'|| Server Domain Avg: {sum(domain_acc)/len(domain_acc)}')
                return sum(domain_acc)/len(domain_acc)

    def Save_CheckPoint(self, save_path):
        torch.save(self.global_model.state_dict(), save_path)

    def save_all(self,save_path):
        pass

    def average_weights(self, w, idxs_users):
        """
        average the weights from all local models
        """
        # train_idx
        # 获取全部数据量

        total_data_points = sum([len(self.LocalModels[r].train_idx) for r in idxs_users])

        fed_avg_freqs = [len(self.LocalModels[r].train_idx) / total_data_points for r in idxs_users]
        #
        # # ==================================
        # global_para= copy.deepcopy(w[0])
        # # 加权平均
        # for idx in range(len(idxs_users)):  # 可以包含GCE聚合，GCE按照客户端数据量进行加权平均
        #     net_para = w[idx]
        #     if idx == 0:
        #         for key in net_para:
        #             global_para[key] = net_para[key] * fed_avg_freqs[idx]
        #     else:
        #         for key in net_para:
        #             global_para[key] += net_para[key] * fed_avg_freqs[idx]

        #  ==================================
        fedavg_global_params = copy.deepcopy(w[0])
        # d=[]
        for name_param in w[0]:
            list_values_param = []
            for dict_local_params, num_local_data in zip(w, fed_avg_freqs):
                # print(dict_local_params[name_param])
                list_values_param.append(dict_local_params[name_param] * num_local_data)
            # print("list_values_param:",list_values_param)
            value_global_param = sum(list_values_param) / sum(fed_avg_freqs)
            # print("value_global_param:",value_global_param)

            # print("name_param:"+name_param+':',fedavg_global_params[name_param]-value_global_param)

            # print("name_param:"+name_param+':',torch.mean(torch.abs(fedavg_global_params[name_param]-value_global_param)))
            # if name_param[-6:]=="weight":
            # a=1-torch.mean(torch.abs(fedavg_global_params[name_param]-value_global_param))
            # d.append(a.item())
            # d=0.999
            fedavg_global_params[name_param] = value_global_param
        global_para = fedavg_global_params
        return global_para


    def test_domain(self):
        if self.args.leave_domain:
            return
        print('------------------------------------------')
        print('test on client domain')
        self.global_model.eval()
        doamin_client_dict = self.args.domain_client_dict  # 列表
        client_acc = []
        root_path = self.args.logdir
        # 拷贝一个空模型
        # self.global_model.load_state_dict(torch.load(root_path + '/global_model.pth'))
        for idx in range(self.args.num_clients):
            try:
                self.LocalModels[idx].load_model(torch.load(root_path + '/client_' + str(idx) + '.pth'))
            except:
                # 有的客户端可能从来没被选择过
                self.LocalModels[idx].load_model(torch.load(root_path + '/best_model' + '.pth', map_location='cpu'))
            # w, loss = self.LocalModels[idx].update_weights(global_round=self.args.comm_round)
            acc = self.LocalModels[idx].test_accuracy()
            client_acc.append(acc)

        # doamin_client_dict = ['m','m','n','n','q','q']
        meta_num = self.args.meta_num  # 每个域的客户端数量
        domain_acc = {}
        for i in range(0, len(doamin_client_dict), meta_num):
            one_domain_acc = []
            for j in range(meta_num):
                one_domain_acc.append(client_acc[i+j])
            domain_acc[doamin_client_dict[i]] = sum(one_domain_acc)/meta_num

        for domain, acc in domain_acc.items():
            self.logger.info(f'|| Client Domain: {domain} Accuracy: {acc} ||')

        print('---------------------------------------------')

        # 删除模型权重文件
        # import os
        # for idx in range(self.args.num_clients):
        #     try:
        #         os.remove(root_path + '/client_' + str(idx) + '.pth')
        #     except:
        #         pass
        # try:
        #     os.remove(root_path + '/best_model.pth')
        # except:
        #     pass

    def csv_log(self, new_row):
        # 追加新行
        # self.df = self.df.append(new_row, ignore_index=True)
        # # 保存到 CSV 文件
        # print(self.args.logdir + '/result.csv')
        # self.df.to_csv(self.args.logdir + '/result.csv', index=False)
        #
        # new_row_df = pd.DataFrame([new_row])
        self.df.loc[len(self.df)] = new_row
        self.df.to_csv(self.args.logdir + '/result.csv', index=False)