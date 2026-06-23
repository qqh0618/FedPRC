"""
2025/3/12 20:36
本文件由my_ywj首次创建编写
"""
import clip

from model.domain_model import ResNet18_domain, shufflenet, mobilenet, vgg, efficinet, densenet, ResNet18_awa
from model.models import CNNFemnist, ResNet18, ShuffLeNet, AGGCNN
from model.stable_model import SimpleNet as ResNet18_stable
from model.wjclip import WJCLIP
def init_model(args):
    if args.model == 'CNN':
        # for EMNIST 62 classes
        global_model = CNNFemnist(args, code_length=args.code_len, num_classes=args.num_classes)

    if args.model == "AGGCNN":
        global_model = AGGCNN(args, code_length=args.code_len, num_classes=args.num_classes)

    if args.model == 'resnet18':
        global_model = ResNet18(args, code_length=args.code_len, num_classes=args.num_classes)

    if args.model == 'shufflenet':
        # global_model = ShuffLeNet(args, code_length=args.code_len, num_classes=args.num_classes)
        global_model = shufflenet(args, code_length=args.code_len, num_classes=args.num_classes)

    if args.model == 'mobilenet':
        global_model = mobilenet(args, code_length=args.code_len, num_classes=args.num_classes)

    if args.model == 'vgg':
        global_model = vgg(args, code_length=args.code_len, num_classes=args.num_classes)

    if args.model == 'efficinet':
        global_model = efficinet(args, code_length=args.code_len, num_classes=args.num_classes)

    if args.model == 'densenet':
        global_model = densenet(args, code_length=args.code_len, num_classes=args.num_classes)

    if args.model == 'resnet18' and args.domain:
        if args.alg=='FedAWA':
            print('***使用awa***')
            global_model = ResNet18_awa(args, code_length=args.code_len, num_classes=args.num_classes)
        else:
            print('***使用域模型***')
            global_model = ResNet18_domain(args, code_length=args.code_len, num_classes=args.num_classes)

    if args.model == 'resnet18' and args.alg == 'StableFDG':
        global_model = ResNet18_stable(num_classes=args.num_classes)

    if args.model == 'clip':

        global_model = WJCLIP(args)
        # global_model = clip.load()

    return global_model