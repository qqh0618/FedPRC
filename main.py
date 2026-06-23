"""
2025/3/12 20:28
本文件由my_ywj首次创建编写
"""
import datetime
import json
import os.path
import pickle
import logging
import random
from pathlib import Path

import torch
from tensorboardX import SummaryWriter

from core.base import init_server
from general import increment_path, mkdirs
from model.base import init_model
from option import args_parser

from datautil.cls_datasets import cls_util
import numpy as np
def init_logger(args):
    cur_config_dir = f"{args.alg}_{args.beta}_{args.seed}_{args.part}_{args.dataset}_{args.leave_domain}_{args.sampling_rate}_{args.meta_num}_{args.uv}_{args.coral}_{args.style}"
    save_dir = str(increment_path(Path(args.project) / cur_config_dir / args.name))
    mkdirs(os.path.join(save_dir, args.logdir))
    args.logdir = os.path.join(save_dir, args.logdir)

    if args.log_file_name is None:
        argument_path = 'experiment_arguments-%s.json' % datetime.datetime.now().strftime("%Y-%m-%d-%H:%M-%S")
    else:
        argument_path = args.log_file_name + '.json'
    with open(os.path.join(args.logdir, argument_path), 'w') as f:
        json.dump(str(args), f)
    log_path = args.log_file_name + '.log'
    logging.basicConfig(

        filename=os.path.join(args.logdir, log_path),
        # filename='/home/qinbin/test.log',
        format='%(asctime)s %(levelname)-8s %(message)s',
        datefmt='%m-%d %H:%M', level=logging.DEBUG, filemode='w')
    # 提示存储位置
    print('Results Save in: {}'.format(save_dir))
    return logging.getLogger()


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 参数解析
    args = args_parser()
    logger = init_logger(args)
    # 数据划分

    # seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    train_loader, test_loader, global_test, net_idx_dataidx_map = cls_util(args, logger)

    # args.code_len = args.num_classes*4

    global_model = init_model(args)
    global_model.to(device)

    server = init_server(args, global_model, train_loader, test_loader, global_test, logger, device, net_idx_dataidx_map)

    server.Create_Clints()
    server.train()






