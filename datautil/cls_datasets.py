"""
2025/3/12 20:30
本文件由my_ywj首次创建编写
"""
import torch
import numpy as np

from datautil.domainsampling import domain_gen_util, domain_shift_util
from datautil.sampling import partition_data, LocalDataloaders


def cls_util(args, logger):
    logger.info('Dataset: {}'.format(str(args.dataset)))
    if args.domain:
        # 域泛化
        if args.leave_domain:  # 有的留一域通常是做域泛化
            print("留一域：域泛化任务")
            Loaders_train, Loaders_test, global_loader_test, net_idx_dataidx_map = domain_gen_util(args, logger)
            return Loaders_train, Loaders_test, global_loader_test, net_idx_dataidx_map
        
        else:   # 不留一域的是研究域倾斜的
            print("不留一域：研究域倾斜任务")
            Loaders_train, Loaders_test, global_loader_test, net_idx_dataidx_map = domain_shift_util(args, logger)
            return Loaders_train, Loaders_test, global_loader_test, net_idx_dataidx_map
    else:
        # 数据加载, 传统联邦学习
        #
        train_dataset, testset, dict_users, dict_users_test = partition_data(n_users=args.num_clients, alpha=args.beta,
                                                                             rand_seed=args.seed,
                                                                             dataset=str(args.dataset),args=args)
        Loaders_train = LocalDataloaders(train_dataset, dict_users, args.batch_size, ShuffleorNot=True, frac=args.part)
        Loaders_test = LocalDataloaders(testset, dict_users_test, args.batch_size, ShuffleorNot=True)
        global_loader_test = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=True, num_workers=0)
        for idx in range(args.num_clients):
            counts = [0] * args.num_classes
            for batch_idx, (X, y) in enumerate(Loaders_train[idx]):
                batch = len(y)
                y = np.array(y)
                for i in range(batch):
                    counts[int(y[i])] += 1
            logger.info('Client {} data distribution:'.format(idx))
            logger.info(counts)

        net_idx_dataidx_map = {}
        for i in range(args.num_clients):
            net_idx_dataidx_map[i] = {args.dataset: {'train':Loaders_train[i].dataset.idxs, 'test':Loaders_test[i].dataset.idxs}}
        return Loaders_train, Loaders_test, global_loader_test, net_idx_dataidx_map
