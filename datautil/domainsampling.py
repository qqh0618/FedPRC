"""
2025/3/30 0:16
while True:
    leanring
本文件由my_ywj首次创建编写
"""
import os

import numpy as np
import torch
from torch.utils import data
from torchvision import datasets, transforms

from datautil.sampling import partition_data, LocalDataloaders, H5_custom

pacs_name_dict = {
    'p': 'photo',
    'a': 'art_painting',
    'c': 'cartoon',
    's': 'sketch',
}  # 10类

office_home_name_dict = {
    'a': 'art',
    'c': 'clipart',
    'p': 'product',
    'r': 'realworld',
}  # 65类

office_caltech_name_dict = {
    'c': 'caltech',
    'a': 'amazon',
    'w': 'webcam',
    'd': 'dslr'
} # 10类

digits_name_dict = {
    'm': 'mnist',
    'u': 'usps',
    'y': 'syn',
    's': 'svhn'
}  # 10类

office31_name_dict = {
    'a': 'amazon',
    'd': 'dslr',
    'w': 'webcam',
}  # 31类

domainnet_name_dict = {
    'c': 'clipart',
    'p': 'painting',
    'r': 'real',
    's': 'sketch',
    'i': 'infograph',
    'q': 'quickdraw',
}


def domain_gen_util(args, logger):
    # 域泛化
    logger.info('Dataset: {}'.format(str(args.dataset)))
    # 留一域作为测试域
    if args.dataset == 'pacs':
        domain_name_dict = pacs_name_dict
        args.num_classes = 7
    elif args.dataset == 'office_home':
        domain_name_dict = office_home_name_dict
        args.num_classes = 65
    elif args.dataset == 'caltech10':
        domain_name_dict = office_caltech_name_dict
        args.num_classes = 10
    elif args.dataset == 'digits':
        domain_name_dict = digits_name_dict
        args.num_classes = 10
    elif args.dataset == 'office31':
        domain_name_dict = office31_name_dict
        args.num_classes = 31
    elif args.dataset == 'domainnet':
        domain_name_dict = domainnet_name_dict
        args.num_classes = 345
    else:
        raise NotImplementedError
    # 随机选一个留一域作为测试域
    if args.leave_domain:
        # 选取第一个域为测试域
        args.test_domain = list(domain_name_dict.keys())[int(args.leave_domain)-1]

        logger.info(f'Leave one domain: {domain_name_dict[args.test_domain]} ')
    # 其他域作为训练域
    args.train_domain = [k for k in domain_name_dict.keys() if k != args.test_domain]


    all_train_dataset = []
    all_test_dataset = []
    global_loader_test = None
    args.num_clients = (int(args.meta_num))*(len(args.train_domain))

    net_idx_dataidx_map = {}

    for i, k in enumerate(args.train_domain):
        # 利用现有的数据划分方法，划分出每个域的客户端数量
        # train_dataset, testset, dict_users, dict_users_test = partition_data(n_users=args.num_clients, alpha=args.beta,
        #                                                                      rand_seed=args.seed,
        #                                                                      dataset=str(args.dataset))
        logger.info(f'Domain: {domain_name_dict[k]} || Client number: {args.meta_num}')
        train_dataset, testset, dict_users, dict_users_test = partition_data(n_users=args.meta_num, alpha=args.beta,  # 根据代码设计规则，后N-1组为客户端域的数量
                                                                             rand_seed=args.seed,
                                                                             dataset=domain_name_dict[k], args=args)   # 这里的dataset子域是数据集的名称
        Loaders_train = LocalDataloaders(train_dataset, dict_users, args.batch_size, ShuffleorNot=True, frac=args.part)  # 其实没啥用
        Loaders_test = LocalDataloaders(testset, dict_users_test, args.batch_size, ShuffleorNot=True,frac=
                                        2*args.part)  # 其实没啥用

        # 把每个客户端的索引和数据索引对应起来
        for j in range(args.meta_num):
            net_idx_dataidx_map[i*args.meta_num+j] = {domain_name_dict[k]: {'train':dict_users[j], 'test':dict_users_test[j]}}


        # Loaders_train是list,要把他们加到一个list中

        for lt in Loaders_train:
            all_train_dataset.append(lt)
        for lt in Loaders_test:
            all_test_dataset.append(lt)

    _, global_test_dataset,_, _ = partition_data(n_users=1, alpha=args.beta,  # 根据代码设计规则，后N-1组为客户端域的数量
                   rand_seed=args.seed,
                   dataset=domain_name_dict[args.test_domain], args=args)
    global_test_loader= torch.utils.data.DataLoader(global_test_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)

    distribution(all_train_dataset, args, logger)

    return all_train_dataset, all_test_dataset, global_test_loader, net_idx_dataidx_map





def domain_shift_util(args, logger):
    # 做域适应
    logger.info('Dataset: {}'.format(str(args.dataset)))
    # 域泛化
    if args.dataset == 'pacs':
        domain_name_dict = pacs_name_dict
        args.num_classes = 7
    elif args.dataset == 'office_home':
        domain_name_dict = office_home_name_dict
        args.num_classes = 65
    elif args.dataset == 'caltech10':
        domain_name_dict = office_caltech_name_dict
        args.num_classes = 10
    elif args.dataset == 'digits':
        domain_name_dict = digits_name_dict
        args.num_classes = 10
    elif args.dataset == 'office31':
        domain_name_dict = office31_name_dict
        args.num_classes = 31
    elif args.dataset == 'domainnet':
        domain_name_dict = domainnet_name_dict
        args.num_classes = 345
    else:
        print(args.dataset)
        raise NotImplementedError

    # 其他域作为训练域
    args.train_domain = [k for k in domain_name_dict.keys()]

    all_train_dataset = []
    all_test_dataset = []
    all_test = {}
    args.num_clients = int(args.meta_num) * (len(args.train_domain))
    domain_client_dict = [] # 记录客户端所在域

    net_idx_dataidx_map = {}

    for i, k in enumerate(args.train_domain):
        # 利用现有的数据划分方法，划分出每个域的客户端数量
        logger.info('Domain: {}'.format(domain_name_dict[k]))
        # train_dataset, testset, dict_users, dict_users_test = partition_data(n_users=args.num_clients, alpha=args.beta,
        #                                                                      rand_seed=args.seed,
        #                                                                      dataset=str(args.dataset))

        train_dataset, testset, dict_users, dict_users_test = partition_data(n_users=args.meta_num,
                                                                             alpha=args.beta,
                                                                             rand_seed=args.seed,
                                                                             dataset=domain_name_dict[k],
                                                                             args=args)  # 这里的dataset子域是数据集的名称
        Loaders_train = LocalDataloaders(train_dataset, dict_users, args.batch_size, ShuffleorNot=True, frac=args.part)
        Loaders_test = LocalDataloaders(testset, dict_users_test, args.batch_size, ShuffleorNot=True,
                                        frac=2 * args.part)

        # 把每个客户端的索引和数据索引对应起来
        for j in range(args.meta_num):
            net_idx_dataidx_map[i*args.meta_num+j] = {domain_name_dict[k]: {'train':dict_users[j], 'test':dict_users_test[j]}}

        # 按顺序添加客户端的域
        for j in range(args.meta_num):
            domain_client_dict.append(domain_name_dict[k])

        for lt in Loaders_train:
            all_train_dataset.append(lt)
        for lt in Loaders_test:
            all_test_dataset.append(lt)

        all_test[domain_name_dict[k]] = testset  # 只是为了方便分域测试，所以进行了一个字典的存储

    args.domain_client_dict = domain_client_dict  # 记录客户端所在域
    # 把所有的测试集合并成一个
    # all_test = torch.utils.data.ConcatDataset(all_test)
    global_loader_test = {}
    for k in all_test.keys():
        global_loader_test[k] = torch.utils.data.DataLoader(all_test[k], batch_size=args.batch_size, shuffle=True, num_workers=0)

    distribution(all_train_dataset, args, logger)

    return all_train_dataset, all_test_dataset, global_loader_test, net_idx_dataidx_map



def distribution(Loaders_train, args, logger):
    for idx in range(len(Loaders_train)):
        counts = [0] * args.num_classes
        for batch_idx, (X, y) in enumerate(Loaders_train[idx]):
            batch = len(y)
            y = np.array(y)
            for i in range(batch):
                counts[int(y[i])] += 1
        logger.info('Client {} data distribution:'.format(idx))
        logger.info(counts)

def get_loader(args, dataset, data_idx, train=True, args_test=None):
    # 单个客户端加载数据集
    if dataset == 'cifar10':
        K = 10
        data_dir = './data/cifar10/'
        apply_transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        # train_dataset = datasets.CIFAR10(data_dir, train=True, download=True,
        #                                transform=apply_transform)
        # test_dataset = datasets.CIFAR10(data_dir, train=False, download=True,
        #                                   transform=apply_transform)
        if train:
            train_dataset = H5_custom(root=os.path.join(data_dir, 'train.h5'), dataidxs=data_idx, train=True,
                                      transform=apply_transform)
            # train
            dl = data.DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False)
        else:
            test_dataset = H5_custom(root=os.path.join(data_dir, 'test.h5'), dataidxs=data_idx, train=False,
                                     transform=apply_transform)
            # test
            dl = data.DataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False)
        # y_train = np.array(train_dataset.targets)
        # y_test = np.array(test_dataset.targets)


    if dataset == 'cifar100':
        K = 100
        data_dir = './data/cifar100/'
        apply_transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        train_dataset = datasets.CIFAR100(data_dir, train=True, download=True,
                                          transform=apply_transform)
        test_dataset = datasets.CIFAR100(data_dir, train=False, download=True,
                                         transform=apply_transform)
        y_train = np.array(train_dataset.targets)
        y_test = np.array(test_dataset.targets)

    if dataset == 'cars':
        K = 196
        data_dir = './data/StanfordCars/'
        apply_transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        if train:
            train_dataset = H5_custom(root=os.path.join(data_dir, 'train.h5'), dataidxs=data_idx, train=True,
                                      transform=apply_transform)
            # train
            dl = data.DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False)
        else:
            test_dataset = H5_custom(root=os.path.join(data_dir, 'test.h5'), dataidxs=data_idx, train=False,
                                     transform=apply_transform)
            # test
            dl = data.DataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False)


    if dataset == 'cubs':
        K = 200
        data_dir = './data/cubs/'
        apply_transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        if train:
            train_dataset = H5_custom(root=os.path.join(data_dir, 'train.h5'), dataidxs=data_idx, train=True,
                                      transform=apply_transform)
            # train
            dl = data.DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False)
        else:
            test_dataset = H5_custom(root=os.path.join(data_dir, 'test.h5'), dataidxs=data_idx, train=False,
                                     transform=apply_transform)
            # test
            dl = data.DataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False)




    if dataset == 'EMNIST':
        K = 62
        data_dir = './data/EMNIST/'
        apply_transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5), (0.5))])
        if train:
            train_dataset = H5_custom(root=os.path.join(data_dir, 'train.h5'), dataidxs=data_idx, train=True,
                                      transform=apply_transform)
            # train
            dl = data.DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False)
        else:
            test_dataset = H5_custom(root=os.path.join(data_dir, 'test.h5'), dataidxs=data_idx, train=False,
                                     transform=apply_transform)
            # test
            dl = data.DataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False)

    if dataset.lower() in ['svhn', 'mnist', 'usps', 'syn',  # digits
                           'amazon', 'caltech', 'dslr', 'webcam',  # caltech
                           'amazon', 'dslr', 'webcam',  # office31
                           'art', 'clipart', 'product', 'realworld',  # office_home
                           'art_painting', 'cartoon', 'photo', 'sketch',  # pacs
                            'clipart', 'infograph', 'quickdraw', 'real', 'sketch','painting'  # domainnet
                           ]:  # h5大家族
        if dataset.lower() in ['svhn', 'mnist', 'usps', 'syn']:
            K = 10
        elif args.domain_dataset == 'caltech10' and dataset.lower() in ['amazon', 'caltech', 'dslr', 'webcam']:
            K = 10
        elif args.domain_dataset == 'office31' and dataset.lower() in ['amazon', 'dslr', 'webcam']:
            K = 31
        elif args.domain_dataset == 'office_home' and dataset.lower() in ['art', 'clipart', 'product', 'realworld']:
            K = 65
        elif args.domain_dataset == 'pacs' and dataset.lower() in ['art_painting', 'cartoon', 'photo', 'sketch']:
            K = 7
        elif args.domain_dataset == 'domainnet' and dataset.lower() in ['clipart', 'infograph', 'quickdraw', 'real', 'sketch','painting']:
            K = 345
        else:
            print(f"{args.domain_dataset} 或者{dataset}没有实现")
            raise NotImplementedError
        data_dir = os.path.join('./data', dataset.lower())

        data_dir = os.path.join('./data', args.domain_dataset, dataset) if args.domain_dataset else data_dir

        apply_transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Resize((244, 244))
             ]
        )
        if train:
            train_dataset = H5_custom(root=os.path.join(data_dir, 'train.h5'), dataidxs=data_idx, train=True, transform=apply_transform)
            # train
            dl = data.DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False)
        else:
            test_dataset = H5_custom(root=os.path.join(data_dir, 'test.h5'), dataidxs=data_idx, train=False, transform=apply_transform)
            # test
            dl = data.DataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False)

    else:
        if args.domain:
            print(f"数据集{dataset}没有实现")
            raise NotImplementedError


    return dl
