import argparse
import os
import shutil
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.tensorboard import SummaryWriter

from data import dataloader
from fed import Fed_client, Fed_server
from models.utils import *
from utils.logger import Logger, print_write, write_summary
from utils.misc import deterministic, update_config

torch.backends.cudnn.benchmark = True

"""
Script for original federated training (train only a global model). 
Support different datasets/losses/samplings. 
"""


if __name__ == "__main__":

    data_root_dict = {
        "IMU": "../dataset/IMU",
    }

    """
    Parameters
    """
    # important parameters
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", default="config/fedlt.yaml", type=str)
    parser.add_argument("--exp_name", default="ours", type=str, help="exp name")
    parser.add_argument(
        "--non_iidness", default=1, type=int, help="non-iid degree of distributed data"
    )
    # parser.add_argument("--tao_ratio", type=float, default=2, choices=[0.5, 1, 2, 4])
    # optional params
    parser.add_argument("--seed", default=1, type=int, help="using fixed random seed")
    parser.add_argument("--work_dir", default="./runs_exp", type=str, help="output dir")
    # unused params
    parser.add_argument("--test", default=False, action="store_true")
    args = parser.parse_args()

    # random seed
    if args.seed is not None:
        deterministic(args.seed)

    # config
    log_dir = f"{args.work_dir}/{args.exp_name}"
    if os.path.exists(log_dir):
        shutil.rmtree(log_dir)
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    with open(args.cfg) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    config = update_config(config, args, log_dir)  # let args overwite YAML config

    # logger
    logger = Logger(log_dir)
    logger.log_cfg(config)  # save cfg
    log_file = f"{log_dir}/log.txt"
    if os.path.exists(log_file):
        os.remove(log_file)

    # tensorboard
    tensorboard = SummaryWriter(log_dir=f"{log_dir}/tensorboard")

    # prepare dataset
    dataset = config["dataset"]["name"]
    data_root = data_root_dict[dataset]
    if dataset in ["CIFAR100", "CIFAR10"]:
        if config["dataset"]["shot_few"] > 0:
            (
                per_client_data,
                per_client_label,
                test_data,
                test_label,
                cls_per_client,
            ) = dataloader.CIFAR_FL_mixed(data_root, config)
        else:
            (
                per_client_data,
                per_client_label,
                test_data,
                test_label,
                cls_per_client,
                num_per_cls_per_client,
                train_num_per_cls,
            ) = dataloader.CIFAR_FL(data_root, config)
    elif dataset == "IMU":
        (
            per_client_data,
            per_client_label,
            test_data,
            test_label,
            cls_per_client,
            num_per_cls_per_client,
            train_num_per_cls,
        ) = dataloader.IMU_FL(data_root, config)
    training_num_per_cls = np.array([len(i) for i in per_client_label])

    tmp = []
    for i, (cls, num_per_cls) in enumerate(zip(cls_per_client, num_per_cls_per_client)):
        tmp.append(f"Client {i}: {cls}" + "\n" + f"{num_per_cls}" + "\n\n")
    print_write(tmp, log_file)

    # print_write([cls_per_client], log_file)
    # print_write([train_num_per_cls], log_file)
    # print_write([num_per_cls_per_client], log_file)
    # print(len(per_client_label[0]))

    """"
    FL config setup
    """
    # assign devices
    num_client = config["fl_opt"]["num_clients"]
    num_cls = config["dataset"]["num_classes"]

    gpu_list = [0,1,2,3][: torch.cuda.device_count()]
    gpu_idx = [i % len(gpu_list) for i in range(num_client)]
    config.update(
        {"device": torch.device("cpu" if torch.cuda.is_available() else "cpu")}
    )  # init_network
    config.update(
        {
            "device_client": [
                torch.device(f"cuda:{i}" if torch.cuda.is_available() else "cpu")
                for i in gpu_idx
            ]
        }
    )  # client device\
    print(f"gpu of clients: {gpu_idx}")

    # init model and criterion on cpu
    network = init_models(config)  # print(network["feat_model"])
    criterion = init_criterion(config)

    # multi-process setup
    import multiprocessing as mp

    if torch.cuda.is_available():
        mp.set_start_method("spawn")
    process_list = []
    state_list = mp.Manager().list([0 for i in range(num_client)])
    model_list = mp.Manager().list([None for i in range(num_client)])

    # FL class for servrer
    fed = Fed_server(
        network,
        criterion,
        config,
        per_client_data,
        per_client_label,
        training_num_per_cls,
        test_data,
        test_label,
        state_list,
        model_list,
    )
    aggre_mode = config["fl_opt"]["aggregation"]
    frac = config["fl_opt"]["frac"]
    client_per_round = max(int(frac * fed.num_clients), 1)

    # FL class for clients
    for i in range(num_client):
        p = Fed_client(
            network,
            criterion,
            config,
            per_client_data,
            per_client_label,
            training_num_per_cls,
            test_data,
            test_label,
            state_list,
            model_list,
            idx=i,
        )
        p.daemon = True
        p.start()
        process_list.append(p)

    """"
    FL starts
    """
    # training mode
    if not args.test:
        best_acc = 0

        # FL rounds
        for round_i in range(config["fl_opt"]["rounds"]):

            # select users
            selected_idx = np.random.choice(
                range(fed.num_clients), client_per_round, replace=False
            )
            # select classes and num_of_samples per class
            selected_cls = []
            for i in selected_idx:
                selected_cls += list(cls_per_client[i])
            print_write(
                [f"\n Round: {round_i}, selected clients: {selected_idx}"], log_file
            )
            # print_write([f'selected cls: {set(selected_cls)}'], log_file)

            if round_i in config["metainfo"]["lr_step"]:
                for i, p in enumerate(process_list):
                    for m in p.optimizers[i].values():
                        for pg in m.param_groups:
                            pg["lr"] *= 0.1
                    for pg in p.optimizers_stage2[i].param_groups:
                        pg["lr"] *= 0.1

            # train and aggregate
            fed.local_train(selected_idx)
            fed.aggregation(selected_idx, aggre_mode)

            # evaluate
            (
                train_loss_per_cls,
                train_acc_per_cls,
                test_loss_per_cls,
                test_acc_per_cls,
            ) = fed.evaluate_global()
            train_loss = train_loss_per_cls.mean()
            train_acc = train_acc_per_cls.mean()

            # testset (e.g., Cifar-100) is equally distributed among classes
            test_loss_mean = test_loss_per_cls.mean()
            test_acc_mean = test_acc_per_cls.mean()

            # logging
            kd_loss = np.array(fed.losses_kd)[selected_idx].mean()
            cls_loss = np.array(fed.losses_cls)[selected_idx].mean()
            np.set_printoptions(precision=3)
            print_write(
                [
                    "cls_loss, kd_loss, train_loss, test_loss, train_acc, test_acc: ",
                    [
                        cls_loss,
                        kd_loss,
                        train_loss,
                        test_loss_mean,
                        train_acc,
                        test_acc_mean,
                    ],
                ],
                log_file,
            )
            print_write(["per_cls_acc (train): ", train_acc_per_cls], log_file)
            print_write(["per_cls_acc: ", test_acc_per_cls], log_file)

            write_summary(
                tensorboard,
                split="train",
                step=round_i,
                kd_loss=kd_loss,
                cls_loss=cls_loss,
                loss=train_loss,
                acc=train_acc,
                **{
                    f"cls{i}_acc": train_acc_per_cls[i]
                    for i in range(config["dataset"]["num_classes"])
                },
            )
            write_summary(
                tensorboard,
                split="val",
                step=round_i,
                loss=test_loss_mean,
                acc=test_acc_mean,
                **{
                    f"cls{i}_acc": test_acc_per_cls[i]
                    for i in range(config["dataset"]["num_classes"])
                },
            )
            # torch.save(
            #     f"{log_dir}/{round_i}.pth",
            #     {
            #         "round_i": round_i,
            #         "server_network": fed.server_network,
            #         "client_network": fed.networks,
            #         "train_acc_per_cls": train_acc_per_cls,
            #         "test_acc_per_cls": test_acc_per_cls,
            #     },
            # )
            # save ckpts
            if test_acc_mean > best_acc:
                ckpt = {
                    "round_i": round_i,
                    "model": fed.server_network,
                    "train_acc_per_cls": train_acc_per_cls,
                    "test_acc_per_cls": test_acc_per_cls,
                }
                ckpt_name = f"{log_dir}/best.pth"
                torch.save(ckpt, ckpt_name)
                best_acc = test_acc_mean
                print_write(
                    [f"best round: {round_i}, accuracy: {test_acc_mean*100}"], log_file
                )
                # del ckpt

    # Currently do not support test mode. Use `evaluate.py` instead.
    else:
        pass

    for i in process_list:
        p.join()
