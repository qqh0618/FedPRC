import os

import h5py
import numpy as np
from torch.utils.data import Dataset
import torch
from torchvision import datasets, transforms


class LocalDataset(Dataset):
    """
    because torch.dataloader need override __getitem__() to iterate by index
    this class is map the index to local dataloader into the whole dataloader
    """

    def __init__(self, dataset, Dict):
        self.dataset = dataset
        self.idxs = [int(i) for i in Dict]

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        X, y = self.dataset[self.idxs[item]]
        return X, y


def LocalDataloaders(dataset, dict_users, batch_size, ShuffleorNot=True, BatchorNot=True, frac=1):
    """
    dataset: the same dataset object
    dict_users: dictionary of index of each local model
    batch_size: batch size for each dataloader
    ShuffleorNot: Shuffle or Not
    BatchorNot: if False, the dataloader will give the full length of data instead of a batch, for testing
    """
    num_users = len(dict_users)
    loaders = []
    for i in range(num_users):
        num_data = len(dict_users[i])
        frac_num_data = int(frac * num_data)
        if frac_num_data < 10:  # 整个客户端不能少于10个
            frac_num_data = num_data
        if num_data < 100 and frac < 1.0:  # 总数少于100
            print(f"用户: {[i]} 总数为{num_data}, 不建议进行部分采样")
        whole_range = range(num_data)
        frac_range = np.random.choice(whole_range, frac_num_data)
        frac_dict_users = [dict_users[i][j] for j in frac_range]
        if BatchorNot == True:
            loader = torch.utils.data.DataLoader(
                LocalDataset(dataset, frac_dict_users),
                batch_size=batch_size,
                shuffle=ShuffleorNot,
                num_workers=0,
                drop_last=False)
        else:
            loader = torch.utils.data.DataLoader(
                LocalDataset(dataset, frac_dict_users),
                batch_size=len(LocalDataset(dataset, dict_users[i])),
                shuffle=ShuffleorNot,
                num_workers=0,
                drop_last=False)
        loaders.append(loader)
    return loaders


class H5_custom(Dataset):
    # def __init__(self, root, dataidxs=None, train=True, transform=None, target_transform=None, test_ratio=0.2, seed=42):
    def __init__(self, root, dataidxs=None, train=True, transform=None, target_transform=None, download=False,
                 test_ratio=0.2, seed=42):
        self.root = root
        self.dataidxs = dataidxs
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.download = download
        self.datas, self.targets = self.__build_truncated_dataset__()

    def __build_truncated_dataset__(self):
        with h5py.File(self.root, 'r') as hf:
            if self.train:
                data = hf['images'][:]
                target = hf['labels'][:]
            else:
                data = hf['images'][:]
                target = hf['labels'][:]
        if self.dataidxs is not None:
            data = data[self.dataidxs]
            target = target[self.dataidxs]
        return data, target

    def __getitem__(self, index):
        img, target = self.datas[index], self.targets[index]
        # img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target

    def __len__(self):
        return len(self.datas)


def partition_data(n_users, alpha=0.5, rand_seed=0, dataset='cifar10', doamin=False, args=None):
    if dataset.lower() == 'cifar10':
        K = 10
        data_dir = './data/cifar10/'
        apply_transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        # train_dataset = datasets.CIFAR10(data_dir, train=True, download=True,
        #                                transform=apply_transform)
        # test_dataset = datasets.CIFAR10(data_dir, train=False, download=True,
        #                                   transform=apply_transform)
        train_dataset = H5_custom(root=os.path.join(r"E:\wj4all\federate\WJFed\data\cifar10\train.h5"), train=True,
                                  test_ratio=0, seed=rand_seed, transform=apply_transform)
        test_dataset = H5_custom(root=os.path.join(r"E:\wj4all\federate\WJFed\data\cifar10\test.h5"), train=False,
                                 test_ratio=0, seed=rand_seed, transform=apply_transform)
        y_train = np.array(train_dataset.targets)
        y_test = np.array(test_dataset.targets)

    if dataset.lower() == 'cifar100':
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

    if dataset == 'EMNIST':
        K = 62
        data_dir = './data/EMNIST/'
        apply_transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5), (0.5))])
        train_dataset = datasets.EMNIST(data_dir, train=True, split='byclass', download=True,
                                        transform=apply_transform)
        test_dataset = datasets.EMNIST(data_dir, train=False, split='byclass', download=True,
                                       transform=apply_transform)
        y_train = np.array(train_dataset.targets)
        y_test = np.array(test_dataset.targets)

    if dataset=='cars':
        K = 196
        data_dir = './data/StanfordCars/'
        apply_transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        train_dataset = H5_custom(root=os.path.join(r"E:\wj4all\federate\WJFed\data\StanfordCars\train.h5"), train=True,
                                  test_ratio=0, seed=rand_seed, transform=apply_transform)
        test_dataset = H5_custom(root=os.path.join(r"E:\wj4all\federate\WJFed\data\StanfordCars\test.h5"), train=False,
                                 test_ratio=0, seed=rand_seed, transform=apply_transform)
        y_train = np.array(train_dataset.targets)
        y_test = np.array(test_dataset.targets)

    if dataset == 'cubs':
        K = 200
        data_dir = './data/cubs/'
        apply_transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        train_dataset = H5_custom(root=os.path.join(r"E:\wj4all\federate\WJFed\data\cubs\train.h5"), train=True,
                                  test_ratio=0, seed=rand_seed, transform=apply_transform)
        test_dataset = H5_custom(root=os.path.join(r"E:\wj4all\federate\WJFed\data\cubs\test.h5"), train=False,
                                 test_ratio=0, seed=rand_seed, transform=apply_transform)
        y_train = np.array(train_dataset.targets)
        y_test = np.array(test_dataset.targets)

    if args.domain:
        apply_transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Resize((224, 224)),
             transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
             ])
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
            elif args.domain_dataset == 'domainnet' and dataset.lower() in ['clipart', 'infograph', 'quickdraw', 'real', 'sketch','painting' ]:
                K = 345
            else:
                print(f"{args.domain_dataset} 或者{dataset}没有实现")
                raise NotImplementedError
            data_dir = os.path.join('./data', dataset.lower())

            data_dir = os.path.join('./data', args.domain_dataset, dataset) if args.domain_dataset else data_dir



            train_dataset = H5_custom(root=os.path.join(data_dir, 'train.h5'), train=True, transform=apply_transform)
            test_dataset = H5_custom(root=os.path.join(data_dir, 'test.h5'), train=False, transform=apply_transform)
            y_train = np.array(train_dataset.targets)
            y_test = np.array(test_dataset.targets)

        else:
            print(f"数据集{dataset}没有实现")
            raise NotImplementedError

    min_size = 0
    N = len(train_dataset)
    N_test = len(test_dataset)
    net_dataidx_map = {}
    net_dataidx_map_test = {}
    np.random.seed(rand_seed)

    while min_size < 10:
        idx_batch = [[] for _ in range(n_users)]
        idx_batch_test = [[] for _ in range(n_users)]
        for k in range(K):
            idx_k = np.where(y_train == k)[0]
            idx_k_test = np.where(y_test == k)[0]
            np.random.shuffle(idx_k)
            proportions = np.random.dirichlet(np.repeat(alpha, n_users))
            ## Balance
            proportions_train = np.array([p * (len(idx_j) < N / n_users) for p, idx_j in zip(proportions, idx_batch)])
            proportions_test = np.array(
                [p * (len(idx_j) < N_test / n_users) for p, idx_j in zip(proportions, idx_batch_test)])
            proportions_train = proportions_train / proportions_train.sum()
            proportions_test = proportions_test / proportions_test.sum()
            proportions_train = (np.cumsum(proportions_train) * len(idx_k)).astype(int)[:-1]
            proportions_test = (np.cumsum(proportions_test) * len(idx_k_test)).astype(int)[:-1]
            idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions_train))]
            idx_batch_test = [idx_j + idx.tolist() for idx_j, idx in
                              zip(idx_batch_test, np.split(idx_k_test, proportions_test))]
            min_size = min([len(idx_j) for idx_j in idx_batch])

    for j in range(n_users):
        np.random.shuffle(idx_batch[j])
        net_dataidx_map[j] = idx_batch[j]
        net_dataidx_map_test[j] = idx_batch_test[j]

    return (train_dataset, test_dataset, net_dataidx_map, net_dataidx_map_test)


def record_net_data_stats(y_train, net_dataidx_map):
    net_cls_counts = {}
    for net_i, dataidx in net_dataidx_map.items():
        unq, unq_cnt = np.unique(y_train[dataidx], return_counts=True)
        tmp = {unq[i]: unq_cnt[i] for i in range(len(unq))}
        net_cls_counts[net_i] = tmp
    return net_cls_counts

