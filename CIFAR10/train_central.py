import os
import numpy as np
import argparse
import yaml
import shutil 
from pathlib import Path

import torch
torch.backends.cudnn.benchmark = True 
from torch.utils.tensorboard import SummaryWriter

from utils.misc import update_config, deterministic
from utils.logger import Logger, print_write, write_summary
from data import dataloader
from models.utils import *
from utils.train_helper import validate_one_model


"""
Script for centralized training (train only a global model). 
"""


if __name__ == '__main__':
    
    data_root_dict = {
        "CIFAR10": "../dataset/cifar_10",
        }

    """
    Parameters
    """
    # important parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', default="config/central.yaml", type=str)
    parser.add_argument("--exp_name", default="central", type=str, help="exp name")
    parser.add_argument("--work_dir", default="./runs_exp", type=str, help="output dir")
    # optional params
    parser.add_argument("--non_iidness", default=1, type=int, help="non-iid degree of distributed data")
    parser.add_argument('--imb_ratio', type=float, default=1, choices=[0.01, 0.05, 0.1, 1])
    parser.add_argument('--seed', default=1, type=int, help="using fixed random seed")  
    # unused params
    parser.add_argument('--test', default=False, action='store_true')
    parser.add_argument('--gpu_idx', default=0, type=int)
    args = parser.parse_args()

    # random seed
    if args.seed is not None:
        deterministic(args.seed)

    # config
    log_dir = f'{args.work_dir}/{args.exp_name}'  
    if os.path.exists(log_dir):
        shutil.rmtree(log_dir)  
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    with open(args.cfg) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    config = update_config(config, args, log_dir)    # let args overwite YAML config
    config.update({"device": torch.device(f'cuda:{args.gpu_idx}' if torch.cuda.is_available() else 'cpu')})
    device = config["device"]

    # logger
    logger = Logger(log_dir)
    logger.log_cfg(config)  # save cfg
    log_file = f"{log_dir}/log.txt"
    if os.path.exists(log_file):
        os.remove(log_file)

    # tensorboard
    tensorboard = SummaryWriter(log_dir=f"{log_dir}/tensorboard")
    

    """
    Prepare models and distributed dataset 
    """
    # model, criterion, optimizers
    network = init_models(config)
    # criterion = torch.nn.CrossEntropyLoss()
    criterion = init_criterion(config)
    optimizer = init_optimizers(network, config)

    # dataset
    dataset = config['dataset']['name']
    data_root = data_root_dict[dataset]
    assert dataset in ["CIFAR100", "CIFAR10"]
    if config['dataset']['shot_few'] > 0:
        per_client_data, per_client_label, test_data, test_label, cls_per_client = dataloader.CIFAR_FL_mixed(data_root, config)
    else:   
        # per_client_data, per_client_label, test_data, test_label, cls_per_client = dataloader.CIFAR_FL(data_root, config)
        per_client_data, per_client_label, test_data, test_label, cls_per_client, num_per_cls_per_client, train_num_per_cls = dataloader.CIFAR_FL(data_root, config)


    training_num_per_cls = np.array([len(i) for i in per_client_label])
    print_write([cls_per_client], log_file)

    # combined distributed dataset into centralized sets
    # train
    train_data, train_label = [], []
    for client_i in range(10):  # 10 client
        train_data.extend(per_client_data[client_i])
        train_label.extend(per_client_label[client_i])
    print(len(train_label), len(train_data), len(test_data), len(test_label))
    train_dataset = dataloader.local_client_dataset(train_data, train_label, config, aug=False)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size = 64, shuffle = True, 
        num_workers = 16, pin_memory=True)

    # test
    test_dataset = dataloader.local_client_dataset(test_data, test_label, config)
    # test_loader = torch.utils.data.DataLoader(
    #     test_dataset, batch_size = 64, shuffle = True, 
    #     num_workers = 16, pin_memory=True)


    # training mode
    if not args.test:
        best_acc = 0

        # FL rounds
        for round_i in range(config["fl_opt"]["rounds"]):
            
            for key in network.keys():
                network[key].train()     

            for (imgs, labels, indexs) in train_loader:

                # to device
                imgs = imgs.to(device)
                labels = labels.to(device)
                # forward
                feat = network['feat_model'](imgs)
                logits = network['classifier'](feat)
                # loss    
                loss, loss_cls, loss_kd = criterion(logits, labels)
                # backward
                for opt in optimizer.values():
                    opt.zero_grad()
                loss.backward()
                for opt in optimizer.values():
                    opt.step()
                # classifier L2-norm
                if network['classifier'].l2_norm:
                    network['classifier'].weight_norm()

            # classes per client
            # for client_i in selected_idx:
            #     print(np.unique(per_client_label[client_i]))

            # evaluate
            for key in network.keys():
                network[key].eval()     
            train_loss_per_cls, train_acc_per_cls = validate_one_model(
                network, train_dataset, device, per_cls_acc=True) 
            test_loss_per_cls, test_acc_per_cls = validate_one_model(
                network, test_dataset, device, per_cls_acc=True) 

            # testset (e.g., Cifar-100) is equally distributed among classes
            test_loss_mean = test_loss_per_cls.mean()
            test_acc_mean = test_acc_per_cls.mean()
            train_loss_mean = train_loss_per_cls.mean()
            train_acc_mean = train_acc_per_cls.mean()

            # logging
            print_write(["\n", "train_loss, train_acc_per_cls, test_loss, per_cls_acc, test_acc_mean: ",\
                train_loss_mean, train_acc_per_cls, test_loss_mean, test_acc_per_cls, test_acc_mean], log_file)
            write_summary(
                tensorboard, split='train', step=round_i, loss=train_loss_mean, acc=train_acc_mean)
            write_summary(
                tensorboard, split='val', step=round_i, loss=test_loss_mean, acc=test_acc_mean, 
                cls0_acc=test_acc_per_cls[0], cls2_acc=test_acc_per_cls[1], cls3_acc=test_acc_per_cls[2],
                cls4_acc=test_acc_per_cls[3], cls5_acc=test_acc_per_cls[4], cls6_acc=test_acc_per_cls[5]
                )

            # save ckpts
            if test_acc_mean > best_acc:
                ckpt = {'round_i': round_i, 'model': network}
                ckpt_name = f"{log_dir}/best.pth"
                torch.save(ckpt, ckpt_name)
                best_acc = test_acc_mean
                print_write([f"best round: {round_i}, accuracy: {test_acc_mean*100}"], log_file)
                # del ckpt

            # scheduler_feat.step()
            # scheduler_classifier.step()

            # print(optimizer['classifier'].state_dict()['param_groups'][0]['lr'])

    # Currently do not support test mode. Use `evaluate.py` instead.
    else:
        pass


