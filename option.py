import argparse


def args_parser():
    parser = argparse.ArgumentParser()

    #Data specifc paremeters
    parser.add_argument('--dataset', default='office31', help='CIFAR10, CIFAR100, SVHN, EMNIST')
    #Training specifc parameters
    parser.add_argument('--log_frq', type=int, default=5, help='frequency of logging')
    parser.add_argument('--logdir', default='logs',)
    parser.add_argument('--batch_size', type=int, default=32, help='minibatch size')  # clip下影响很大2的时候是60，8的时候是80
    parser.add_argument('--comm_round', type=int, default=200, help='number of epochs')
    parser.add_argument('--clip_grad', type=float, default=None, help='gadient clipping')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--lr_sh_rate', type=int, default=10, help='number of steps to drop the lr')
    parser.add_argument('--use_lrschd', action="store_true", default=False, help='Use lr rate scheduler')
    parser.add_argument('--num_clients',  type=int, default=10, help='number of local models')
    parser.add_argument('--num_classes', type=int, default=7, help='number of classes')
    parser.add_argument('--sampling_rate', type=float, default=1.0, help='frac of local models to update')
    parser.add_argument('--local_ep',type=int, default=3, help='iterations of local updating')
    parser.add_argument('--beta', type=float,default=0.1, help='beta for non-iid distribution')
    parser.add_argument('--seed', type=int,default=10, help='random seed for generating datasets')
    parser.add_argument('--code_len', type=int,default=64, help='length of code')
    parser.add_argument('--alg', default='FedSCE', help='FedAvg, FedProx, Moon, FedMD, FedProto, FedDFKD|| FedHEAL, FedFPL, FedIIR')
    parser.add_argument('--lam', type=float, default=0.05, help='hyper-parameter for loss2')
    parser.add_argument('--gamma', type=float, default=0.05, help='hyper-parameter for loss3')
    parser.add_argument('--std', type=float, default=2, help='std of gaussian noise ')
    parser.add_argument('--part', type=float,default=0.1, help='percentage of each local data')
    parser.add_argument('--temp', type=float,default=0.5, help='temperture for soft prediction')
    parser.add_argument('--model', default= 'resnet18', help='CNN resnet18 shufflenet')
    parser.add_argument('--save_model', action="store_true", default= False, help='saved model parameters')
    parser.add_argument('--upload_model', action="store_true", default= True, help='upload parameters')
    parser.add_argument('--eval_only', action="store_true", default=False,help='evaluate the model')
    parser.add_argument('--name', default='exp',help='存放日志累增文件夹名')
    parser.add_argument('--project', type=str, default='da_runs_result',help='存放目录')
    parser.add_argument('--log_file_name', type=str, default='main', help='存放目录')

    # =====================================================
    parser.add_argument('--k', type=int, default=110, help='存放目录')
    # =====================================================

    # =====================================================
    # domain域泛化参数
    # 主要说明用哪些域的数据，每个域的客户端数量，留一域
    parser.add_argument('--domain_path', default=r"E:\wj4all\data", help='存放数据的路径')
    parser.add_argument('--domain', default=True, help='是否使用域泛化，True是域泛化，否则为传统联邦学习')
    parser.add_argument('--domain_dataset', default='office31', help='office, office-home, visda')
    parser.add_argument('--leave_domain', default=0, help='如果为0为域适应,否则使用k个域作为测试域进行域泛化')  # 留一域, 如果设置为None，那么就不使用留一域
    parser.add_argument('--meta_num', type=int, default=3, help='每个域的客户端数量')  # 或者剩下域的每个域客户端数量, 默认客户端域的数量
    # 单域泛化额外参数
    parser.add_argument('--single',type=bool,default=True, help='是否使用单域泛化')
    parser.add_argument('--train_one', type=int, default=1, help='保留哪个域作为训练域,0为不保存')
    # =====================================================
    # VDDA 参数
    parser.add_argument('--ug', type=float, default=1.0, help='每个域的客户端数量')  # 或者剩下域的每个域客户端数量, 默认客户端域的数量
    parser.add_argument('--coral', type=float, default=0.1, help='每个域的客户端数量')  # 或者剩下域的每个域客户端数量, 默认客户端域的数量
    parser.add_argument('--uv', type=float, default=0.1, help='每个域的客户端数量')  # 或者剩下域的每个域客户端数量, 默认客户端域的数量
    parser.add_argument('--soft', type=float, default=0.5, help='每个域的客户端数量')  # 或者剩下域的每个域客户端数量, 默认客户端域的数量
    parser.add_argument('--style', type=float, default=0.1, help='每个域的客户端数量')  # 或者剩下域的每个域客户端数量, 默认客户端域的数量
    parser.add_argument('--noise', type=float, default=1.0, help='噪声客户端比例')

    ## PromptFL参数
    parser.add_argument('--clip_name', default='ViT-B/16', help='clip模型名称')
    parser.add_argument('--n_ctx', type=int, default=7, help='number of text encoder of text prompts')
    parser.add_argument('--ctx_init', type=bool, default=False, help='is using the ctx init, set True for CLIP')
    parser.add_argument('--INPUT_SIZE', type=int, default=224, help='输入图像尺寸')
    parser.add_argument('--csc', type=bool, default=False, help='is using the class-specific context')
    parser.add_argument('--class_token_position', type=str, default='end', help='class token position')
    parser.add_argument('--num_prompt', type=int, default=2, help="number of prompts")
    parser.add_argument('--avg_prompt', type=int, default=1, help="number of prompts to aggregate")
    # FetOTP参数

    parser.add_argument('--glp_otp_n', type=int, default=1, help='number of prompts')
    parser.add_argument('--use_uniform', type=bool, default=True, help='use uniform distribution')
    parser.add_argument('--eps', type=float, default=1, help='epsilon')
    parser.add_argument('--thresh', type=float, default=1e-2, help='threshold')
    parser.add_argument('--OT', type=str, default='sinkhorn', help='OT algorithm')
    parser.add_argument('--top_percent', type=float, default=0.8, help='top percent')
    parser.add_argument('--max_iter', type=int, default=10, help='max iteration')

    # TTA参数
    parser.add_argument('--model_path', type=int, default=10, help='max iteration')

    args = parser.parse_args()
    return args