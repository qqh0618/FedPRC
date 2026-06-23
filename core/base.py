"""
2025/3/12 20:40
本文件由my_ywj首次创建编写
"""
from core.Server.ServerFedAWA import ServerFedAWA
from core.Server.ServerFedAvg import ServerFedAvg
from core.Server.ServerFedBN import ServerFedBN
from core.Server.ServerFedProto import ServerFedProto
from core.Server.ServerFedProx import ServerFedProx
from core.Server.ServerMOON import ServerMOON
from core.Server.dg.ServerCCST import ServerCCST
from core.Server.dg.ServerFedAlign import ServerFedAlign
from core.Server.dg.ServerFedGA import ServerFedGA
from core.Server.dg.ServerFedGM import ServerFedGM
from core.Server.dg.ServerFedGS import ServerFedGS
from core.Server.dg.ServerFedIIR import ServerFedIIR
from core.Server.dg.ServerFedLGF import ServerFedLGF
from core.Server.dg.ServerFedOMG import ServerFedOMG
from core.Server.dg.ServerFedSAM import ServerFedSAM
from core.Server.dg.ServerFedSR import ServerFedSR
from core.Server.dg.ServerFedTTA import ServerFedTTA
from core.Server.dg.ServerFedPRC import ServerFedVDDG

from core.Server.dg.ServerStableFDG import ServerStableFDG
from core.Server.fl.ServerFedSCE import ServerFedSCE

from core.Server.ks.ServerFedHKD import ServerFedHKD
from core.Server.ks.ServerFedMD import ServerFedMD

from core.Server.pfl.ServerFedALA import ServerFedALA
from core.Server.pfl.ServerFedAS import ServerFedAS
from core.Server.pfl.ServerFedBABU import ServerFedBABU
from core.Server.pfl.ServerFedDYN import ServerFedDYN
from core.Server.pfl.ServerFedETF import ServerFedETF
from core.Server.pfl.ServerFedFDA import ServerFedFDA
from core.Server.pfl.ServerFedGPFL import ServerFedGPFL
from core.Server.pfl.ServerFedNH import ServerFedNH
from core.Server.pfl.ServerFedROD import ServerFedROD
from core.Server.pfl.ServerFedUV import ServerFedUV
from core.Server.semi.ServerFedSemi import ServerFedSemi


from core.Server.prompt.ServerFedOMGC import ServerFedOMGC
from core.Server.prompt.ServerFedOTP import ServerFedOTP
from core.Server.prompt.ServerFedProxC import ServerFedProxC
from core.Server.prompt.ServerPFedPrompt import ServerPFedPrompt

from core.Server.single.ServerFedSingle import ServerFedSingle
from core.Server.prompt.ServerFedPrompt import ServerFedPrompt
from core.Server.prompt.ServerFedAvgC import ServerFedAvgC
from core.Server.tta.ServerFedATP import ServerFedATP


def init_server(args, global_model, Loaders_train, Loaders_test, global_loader_test, logger, device, net_idx_dataidx_map):
    if args.alg == 'FedAvg':
        server = ServerFedAvg(args, global_model, Loaders_train, Loaders_test, global_loader_test, logger, device, net_idx_dataidx_map)
    elif args.alg == "MOON":
        server = ServerMOON(args, global_model, Loaders_train, Loaders_test, global_loader_test, logger, device,
                              net_idx_dataidx_map)

    elif args.alg == 'FedProx':
        server = ServerFedProx(args, global_model, Loaders_train, Loaders_test, global_loader_test, logger, device, net_idx_dataidx_map)
    elif args.alg == 'FedBN':
        server = ServerFedBN(args, global_model, Loaders_train, Loaders_test, global_loader_test, logger, device,
                               net_idx_dataidx_map)
    elif args.alg =='FedAWA':
        server = ServerFedAWA(args, global_model, Loaders_train, Loaders_test, global_loader_test, logger, device,
                             net_idx_dataidx_map)
    elif args.alg =='FedSCE':
        server = ServerFedSCE(args, global_model, Loaders_train, Loaders_test, global_loader_test, logger, device,
                             net_idx_dataidx_map)
    elif args.alg == 'FedMD':
        server = ServerFedMD(args, global_model, Loaders_train, Loaders_test, global_loader_test, global_loader_test.dataset, logger,device, net_idx_dataidx_map)
    elif args.alg == 'FedProto':
        server = ServerFedProto(args, global_model, Loaders_train, Loaders_test, global_loader_test, logger, device, net_idx_dataidx_map)
    elif args.alg == 'FedHKD':
        server = ServerFedHKD(args, global_model, Loaders_train, Loaders_test, global_loader_test, logger, device,net_idx_dataidx_map)
    elif args.alg == 'FedALA':
        server = ServerFedALA(args, global_model, Loaders_train, Loaders_test, global_loader_test, logger, device, net_idx_dataidx_map)
    elif args.alg == 'FedDyn':
        server = ServerFedDYN(args, global_model, Loaders_train, Loaders_test, global_loader_test, logger, device, net_idx_dataidx_map)
    elif args.alg == 'FedBABU':
        server = ServerFedBABU(args, global_model, Loaders_train, Loaders_test, global_loader_test, logger, device, net_idx_dataidx_map)
    elif args.alg == 'FedETF':
        server = ServerFedETF(args, global_model, Loaders_train, Loaders_test, global_loader_test, logger, device, net_idx_dataidx_map)
    elif args.alg == 'FedAS':
        server = ServerFedAS(args, global_model, Loaders_train, Loaders_test, global_loader_test, logger, device, net_idx_dataidx_map)
    elif args.alg == 'FedNH':
        server = ServerFedNH(args, global_model, Loaders_train, Loaders_test, global_loader_test, logger, device, net_idx_dataidx_map)
    elif args.alg == 'FedPAC':
        pass
    elif args.alg == 'FedROD':
        server = ServerFedROD(args, global_model, Loaders_train, Loaders_test, global_loader_test, logger, device, net_idx_dataidx_map)
    elif args.alg == 'pFedFDA':
        pass
    elif args.alg == 'GPFL':
        server = ServerFedGPFL(args, global_model, Loaders_train, Loaders_test, global_loader_test, logger, device, net_idx_dataidx_map)
    
    # =================================
    # DG
    elif args.alg == 'FedGA':
        print("FedGA")
        server = ServerFedGA(args, global_model, Loaders_train, Loaders_test, global_loader_test, logger, device, net_idx_dataidx_map)
    elif args.alg == 'FedIIR':
        print("FedIIR")
        server = ServerFedIIR(args, global_model, Loaders_train, Loaders_test, global_loader_test, logger, device, net_idx_dataidx_map)
    elif args.alg == 'StableFDG':
        print("StableFDG")
        server = ServerStableFDG(args, global_model, Loaders_train, Loaders_test, global_loader_test, logger, device, net_idx_dataidx_map)
    elif args.alg == 'FedGM':
        print("FedGM")
        server = ServerFedGM(args, global_model, Loaders_train, Loaders_test, global_loader_test, logger, device, net_idx_dataidx_map)
    elif args.alg == 'FedSAM':
        print("FedSAM")
        server = ServerFedSAM(args, global_model, Loaders_train, Loaders_test, global_loader_test, logger, device, net_idx_dataidx_map)
    elif args.alg == 'CCST':
        print("CCST")
        server = ServerCCST(args, global_model, Loaders_train, Loaders_test, global_loader_test, logger, device, net_idx_dataidx_map)
    elif args.alg == 'FedOMG':
        print("FedOMG")
        server = ServerFedOMG(args, global_model, Loaders_train, Loaders_test, global_loader_test, logger, device, net_idx_dataidx_map)
    elif args.alg == 'FedSR':
        print("FedSR")
        server = ServerFedSR(args, global_model, Loaders_train, Loaders_test, global_loader_test, logger, device,
                              net_idx_dataidx_map)
    elif args.alg == 'FedLGF':
        server = ServerFedLGF(args, global_model, Loaders_train, Loaders_test, global_loader_test, logger, device, net_idx_dataidx_map)
    elif args.alg == 'FedTTA': # tta
        server = ServerFedTTA(args, global_model, Loaders_train, Loaders_test, global_loader_test, logger, device, net_idx_dataidx_map)


    elif args.alg == 'FedAlign':
        server = ServerFedAlign(args, global_model, Loaders_train, Loaders_test, global_loader_test, logger, device, net_idx_dataidx_map)
    # =================================

    # =================================
    # PFL 个性化联邦
    elif args.alg == 'FedUV':
        server = ServerFedUV(args, global_model, Loaders_train, Loaders_test, global_loader_test, logger, device, net_idx_dataidx_map)
    elif args.alg == 'FedFDA':
        server = ServerFedFDA(args, global_model, Loaders_train, Loaders_test, global_loader_test, logger, device, net_idx_dataidx_map)
    elif args.alg == 'FedALA':
        pass
    # =================================

    # =================================
    # single 单域泛化
    elif args.alg == 'FedSingle':
        server = ServerFedSingle(args, global_model, Loaders_train, Loaders_test, global_loader_test, logger, device, net_idx_dataidx_map)
    elif args.alg == 'FedAvgC':
        server = ServerFedAvgC(args, global_model, Loaders_train, Loaders_test, global_loader_test, logger, device, net_idx_dataidx_map)
    elif args.alg == 'FedPrompt':
        server = ServerFedPrompt(args, global_model, Loaders_train, Loaders_test, global_loader_test, logger, device, net_idx_dataidx_map)
    elif args.alg == 'FedProxC':
        server = ServerFedProxC(args, global_model, Loaders_train, Loaders_test, global_loader_test, logger, device, net_idx_dataidx_map)
    elif args.alg == 'PFedPrompt':
        server = ServerPFedPrompt(args, global_model, Loaders_train, Loaders_test, global_loader_test, logger, device, net_idx_dataidx_map)
    elif args.alg == 'FedOTP':
        server = ServerFedOTP(args, global_model, Loaders_train, Loaders_test, global_loader_test, logger, device, net_idx_dataidx_map)
    elif args.alg == 'FedOMGC':
        server = ServerFedOMGC(args, global_model, Loaders_train, Loaders_test, global_loader_test, logger, device, net_idx_dataidx_map)
    # WJDG
    elif args.alg == 'FedGS':
        server = ServerFedGS(args, global_model, Loaders_train, Loaders_test, global_loader_test, logger, device, net_idx_dataidx_map)

    # =================================
    # 联邦半监督学习
    elif args.alg == 'FedSemi':  # todo
        server = ServerFedSemi(args, global_model, Loaders_train, Loaders_test, global_loader_test, logger, device, net_idx_dataidx_map)


    # ==================================
    # 测试时适应
    elif args.alg == 'FedATP':
        server = ServerFedATP(args, global_model, Loaders_train, Loaders_test, global_loader_test, logger, device,
                               net_idx_dataidx_map)

    elif args.alg == 'FedPRC':
        server = ServerFedPRC(args, global_model, Loaders_train, Loaders_test, global_loader_test, logger, device,
                               net_idx_dataidx_map)
    # ==================================

    else:
        raise NotImplementedError(f"Algorithm {args.alg} is not implemented.")

    return server