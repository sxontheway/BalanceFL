import copy
import time
from collections import OrderedDict

import torch
from data.dataloader import local_client_dataset, test_dataset
from models.utils import *
from utils.train_helper import validate_one_model
from utils.sampling import *

import numpy as np
from multiprocessing import Process
import time


def return_state_dict(network):
    """
    save model to state_dict
    """
    feat_model = {k: v.cpu() for k, v in network["feat_model"].state_dict().items()}
    classifier = {k: v.cpu() for k, v in network["classifier"].state_dict().items()}
    return {"feat_model": feat_model, "classifier": classifier}


def load_state_dict(network, state_dict):
    """
    restore model from state_dict
    """
    network["feat_model"].load_state_dict(state_dict["feat_model"])
    network["classifier"].load_state_dict(state_dict["classifier"])
    
    # for name, param in state_dict["feat_model"].items():
    #     print(name, "\t",  param.size())
    return network


def check_status(status_list, selected_idx, target_status):
    """
    0. original status (1st FL round)
    1. server finished sending: server_network --> mp_list
    2. client received, and returned the model: mp_list --> networks[i] --> local_update --> mp_list
    3. server received: mp_list --> networks[i]
    --> 1. aggregation finished. networks[i] --> aggregate --> server_network --> mp_list, the status change to 1
    ---
    Return True: when all clients meet conditions, else False
    """
    tmp = np.array(status_list)
    if (tmp[selected_idx] == target_status).all() == True:
        return True
    else:
        return False


def set_status(status_list, selected_idx, target_status):
    """
    see function: check_status
    """
    if type(selected_idx) is int:
        selected_idx = [selected_idx]
    for i in selected_idx:
        status_list[i] = target_status
    # print(f"set_status {target_status}")


def difference_models_norm_2(model_1, model_2):
    """
    Return the norm 2 difference between the two model parameters. Used in FedProx. 
    """
    tensor_1_backbone = list(model_1["feat_model"].parameters())
    tensor_1_classifier = list(model_1["classifier"].parameters())
    tensor_2_backbone = list(model_2["feat_model"].parameters())
    tensor_2_classifier = list(model_2["classifier"].parameters())
    
    diff_list = [torch.sum((tensor_1_backbone[i] - tensor_2_backbone[i])**2) for i in range(len(tensor_1_backbone))]
    diff_list.extend([torch.sum((tensor_1_classifier[i] - tensor_2_classifier[i])**2) for i in range(len(tensor_1_classifier))])

    norm = sum(diff_list)
    return norm


class Fed_server(Process):
    """
    Class for client updating and model aggregation
    """
    def __init__(
        self, init_network, criterion, config, per_client_data, 
        per_client_label, training_num_per_cls,
        test_data, test_label, state_list=None, state_dict_list=None, idx=None
        ):

        super(Fed_server, self).__init__()

        self.local_bs = config["fl_opt"]["local_bs"]
        self.local_ep = config["fl_opt"]["local_ep"]
        self.num_clients = config["fl_opt"]["num_clients"]
        self.criterion = criterion
        self.networks, self.optimizers, self.optimizers_stage2, self.schedulers = [], [], [], []
        self.train_loaders = []     # include dataloader or pre-loaded dataset
        self.train_loader_balanced = [] # balanced-sampling dataloader
        self.local_num_per_cls = []   # list to store local data number per class
        self.test_loaders = []
        self.status_list = state_list
        self.state_dict_list = state_dict_list
        self.client_idx = idx   # physical idx of clients (hardcoded)

        self.config = config
        self.prefetch = False 
        self.feat_aug = config["fl_opt"]["feat_aug"]
        self.crt = config["fl_opt"]["crt"]

        self.client_weights = np.array([i for i in training_num_per_cls])
        self.client_weights = self.client_weights/self.client_weights.sum()

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.server_network = copy.deepcopy(init_network)
        self.server_network["feat_model"].to(self.device)
        self.server_network["classifier"].to(self.device)

        # per-client accuracy and loss
        self.acc = [0 for i in range(self.num_clients)]
        self.losses_cls = [-1 for i in range(self.num_clients)]
        self.losses_kd = [-1 for i in range(self.num_clients)]

        print(f'=====> {config["metainfo"]["optimizer"]}, Server (fed.py)\n ')

        ######## init backbone, classifier, optimizer and dataloader ########
        for client_i in range(self.num_clients):
            
            backbone = copy.deepcopy(self.server_network["feat_model"])
            classifier = copy.deepcopy(self.server_network["classifier"])
            self.networks.append({"feat_model": backbone, "classifier": classifier})

            """ Server does not need
            # list of optimizer_dict. One optimizer for one network
            self.optimizers.append(init_optimizers(self.networks[client_i], config))   
            optim_params_dict = {'params': self.networks[client_i]["classifier"].parameters(), 'lr': 0.001, 'momentum': 0.9, 'weight_decay': 0} 
            self.optimizers_stage2.append(torch.optim.SGD([optim_params_dict],))

            # dataloader
            num_workers = 0
            local_dataset = \
                local_client_dataset(per_client_data[client_i], per_client_label[client_i], config)
            self.train_loaders.append(
                torch.utils.data.DataLoader(
                    local_dataset, batch_size=self.local_bs, shuffle=True, 
                    num_workers=num_workers, pin_memory=False)
            )
            self.train_loader_balanced.append(
                torch.utils.data.DataLoader(
                    local_dataset, batch_size=self.local_bs, sampler=local_dataset.get_balanced_sampler(), 
                    num_workers=num_workers, pin_memory=False)
            )
            self.local_num_per_cls.append(local_dataset.class_sample_count)
            """

        # centralized train dataset
        train_data_all, train_label_all = [], []
        for client_i in range(len(per_client_label)):
            train_data_all = train_data_all + per_client_data[client_i]
            train_label_all = train_label_all + per_client_label[client_i]
        self.train_dataset = local_client_dataset(train_data_all, train_label_all, config)
        self.test_dataset = test_dataset(test_data, test_label, config)


    def local_train(self, selected_idx):
        """
        server-side code
        """
        # self.server_network --> mp_list
        for i in selected_idx:
            self.state_dict_list[i] = return_state_dict(self.server_network)  # model transfer
        set_status(self.status_list, selected_idx, 1)
        if self.local_ep > 10:  # is local training
            print("Waiting")

        # wait until all clients returning the model
        while check_status(self.status_list, selected_idx, 2) is False:
            time.sleep(0.1)

        # mp_list --> self.networks (copys of client models on the server). Prepare for aggregation.
        for i in selected_idx:
            load_state_dict(self.networks[i], self.state_dict_list[i])  # model transfer
        print("===> Local training finished")


    def aggregation(self, selected_idx, mode):
        """
        server-side code: aggregation
        """
        if mode in ["fedavg", "fedavgm", "fedbn", "fedprox"]:
            self.aggregate_layers(selected_idx, mode, backbone_only=False)
        elif mode == "fedavg_fs":
            opt = self.config["fl_opt"]
            backbone_only, imprint, spread_out = opt["backbone_only"], opt["imprint"], opt["spread_out"] 
            self.aggregate_layers(selected_idx, "fedavg", backbone_only=backbone_only)
            if imprint:
                self.imprint(selected_idx)
            if spread_out:
                self.spread_out()

        # model: self.server_network --> mp_list
        for i in selected_idx:
            self.state_dict_list[i] = return_state_dict(self.server_network)  # model transfer
        set_status(self.status_list, selected_idx, 0)    # back to original 
        
        print("===> Aggregation finished")


    def aggregate_layers(self, selected_idx, mode, backbone_only):
        """
        backbone_only: choose to only aggregate backbone
        """
        weights_sum = self.client_weights[selected_idx].sum()
        with torch.no_grad():
            if mode in ["fedavg", "fedprox"]:
                for net_name, net in self.server_network.items():
                    if net_name == "classifier" and backbone_only:
                        pass
                    else:
                        for key, layer in net.state_dict().items():
                            if 'num_batches_tracked' in key:
                                # num_batches_tracked is a non trainable LongTensor 
                                # and num_batches_tracked are the same for 
                                # all clients for the given datasets
                                layer.data.copy_(self.networks[0][net_name].state_dict()[key])
                            else: 
                                temp = torch.zeros_like(layer)
                                # Fedavg
                                for idx in selected_idx:
                                    weight = self.client_weights[idx]/weights_sum
                                    temp += weight * self.networks[idx][net_name].state_dict()[key]
                                layer.data.copy_(temp)
                                # update client models
                                # for idx in selected_idx:
                                #     self.networks[idx][net_name].state_dict()[key].data.copy_(layer)                
            
            elif mode == "fedbn":   # https://openreview.net/pdf?id=6YEQUn0QICG
                for net_name, net in self.server_network.items():
                    if net_name == "classifier" and backbone_only:
                        pass
                    else:
                        for key, layer in net.state_dict().items():
                            if 'bn' not in key:  
                                temp = torch.zeros_like(layer)
                                # Fedavg
                                for idx in selected_idx:
                                    weight = self.client_weights[idx]/weights_sum
                                    temp += weight * self.networks[idx][net_name].state_dict()[key]
                                layer.data.copy_(temp)
                                # update client models
                                # for idx in selected_idx:
                                #     self.networks[idx][net_name].state_dict()[key].data.copy_(layer)
            elif mode == "fedavgm":
                raise NotImplementedError


    def evaluate_global(self, train_dataset=None, test_dataset=None):
        """
        Accuracy of the global model and all classes
        """
        # evaluate on training set
        if train_dataset is None:
            train_dataset = self.train_dataset
        if test_dataset is None:
            test_dataset = self.test_dataset
        train_loss_per_cls, train_acc_per_cls = validate_one_model(
            self.server_network, train_dataset, self.device, per_cls_acc=True) 

        # evaluate on test set: per-class loss/acc
        test_loss_per_cls, test_acc_per_cls = validate_one_model(
            self.server_network, test_dataset, self.device, per_cls_acc=True) 
        print("===> Evaluation finished\n")
        
        return train_loss_per_cls, train_acc_per_cls, test_loss_per_cls, test_acc_per_cls


    def evaluate_global_all(self, train_dataset=None, test_dataset=None):
        """
        Accuracy of models of all nodes and all classes

        Return: all_results
        shape: (4, num_client, num_cls), 4 for (train_loss, train_acc, test_loss, test_acc)
        """
        # evaluate on training set
        if train_dataset is None:
            train_dataset = self.train_dataset
        if test_dataset is None:
            test_dataset = self.test_dataset

        all_results = [None for i in range(self.num_clients)]
        for idx in range(self.num_clients):
            # evaluate on test set: per-class loss/acc
            train_loss_per_cls, train_acc_per_cls = validate_one_model(
                self.networks[idx], train_dataset, self.device, per_cls_acc=True) 
            # evaluate on test set: per-class loss/acc
            test_loss_per_cls, test_acc_per_cls = validate_one_model(
                self.networks[idx], test_dataset, self.device, per_cls_acc=True) 
            all_results[idx] = train_loss_per_cls, train_acc_per_cls, test_loss_per_cls, test_acc_per_cls

        print(f"===> Evaluation finished{idx}\n")        

        all_results = np.array(all_results).transpose(1,0,2)    
        return all_results



class Fed_client(Process):
    """
    Class for client updating and model aggregation
    """
    def __init__(
        self, init_network, criterion, config, per_client_data, 
        per_client_label, training_num_per_cls,
        test_data, test_label, 
        state_list=None, state_dict_list=None, idx=None
        ):

        super(Fed_client, self).__init__()

        self.local_bs = config["fl_opt"]["local_bs"]
        self.local_ep = config["fl_opt"]["local_ep"]
        self.num_clients = config["fl_opt"]["num_clients"]
        self.criterion = criterion
        self.networks, self.optimizers, self.optimizers_stage2, self.schedulers = [], [], [], []
        self.train_loaders = []     # include dataloader or pre-loaded dataset
        self.train_loader_balanced = [] # balanced-sampling dataloader
        self.local_num_per_cls = []   # list to store local data number per class
        self.test_loaders = []
        self.status_list = state_list
        self.state_dict_list = state_dict_list
        self.client_idx = idx   # physical idx of clients (hardcoded)

        self.config = config
        self.device = config["device_client"][idx]
        self.server_network = copy.deepcopy(init_network)
        self.balanced_loader = config["fl_opt"]["balanced_loader"]
        
        self.prefetch = False
        self.feat_aug = config["fl_opt"]["feat_aug"]
        self.crt = config["fl_opt"]["crt"]

        if config["fl_opt"]["aggregation"] == "fedprox":
            self.fedprox = True
        else:
            self.fedprox = False
        self.mu = 0.05

        self.client_weights = np.array([i for i in training_num_per_cls])
        self.client_weights = self.client_weights/self.client_weights.sum()

        # per-client accuracy and loss
        self.acc = [0 for i in range(self.num_clients)]
        self.losses_cls = [-1 for i in range(self.num_clients)]
        self.losses_kd = [-1 for i in range(self.num_clients)]

        print(f'=====> {config["metainfo"]["optimizer"]}, Client {idx} (fed.py)\n ')        

        ######## init backbone, classifier, optimizer and dataloader ########
        for client_i in range(self.num_clients):
            # list of network and optimizer_dict. One optimizer for one network.
            if client_i != self.client_idx:
                self.networks.append(None)
                self.optimizers.append(None)
                self.optimizers_stage2.append(None)
                self.train_loaders.append(None)
                self.train_loader_balanced.append(None)
                self.local_num_per_cls.append(None)
            else: 
                backbone = copy.deepcopy(self.server_network["feat_model"])
                classifier = copy.deepcopy(self.server_network["classifier"])
                self.networks.append({"feat_model": backbone, "classifier": classifier})
                self.optimizers.append(init_optimizers(self.networks[client_i], config))   
                optim_params_dict = {'params': self.networks[client_i]["classifier"].parameters(), 'lr': 0.001, 'momentum': 0.9, 'weight_decay': 0} 
                self.optimizers_stage2.append(torch.optim.SGD([optim_params_dict],))

                # dataloader
                num_workers = 0
                local_dataset = \
                    local_client_dataset(per_client_data, per_client_label, config)
                self.train_loaders.append(
                    torch.utils.data.DataLoader(
                        local_dataset, batch_size=self.local_bs, shuffle=True, 
                        num_workers=num_workers, pin_memory=False)
                )
                self.train_loader_balanced.append(
                    torch.utils.data.DataLoader(
                        local_dataset, batch_size=self.local_bs, sampler=local_dataset.get_balanced_sampler(), 
                        num_workers=num_workers, pin_memory=False)
                )
                self.local_num_per_cls.append(local_dataset.class_sample_count)

        """ clients do not need
        # centralized train dataset
        train_data_all, train_label_all = [], []
        for client_i in range(len(per_client_label)):
            train_data_all = train_data_all + per_client_data[client_i]
            train_label_all = train_label_all + per_client_label[client_i]
        self.train_dataset = local_client_dataset(train_data_all, train_label_all, config)
        self.test_dataset = test_dataset(test_data, test_label, config)
        """


    def run(self):
        """
        client-side code
        """
        self.server_network["feat_model"].to(self.device)
        self.server_network["classifier"].to(self.device)
        self.networks[self.client_idx]["feat_model"].to(self.device)
        self.networks[self.client_idx]["classifier"].to(self.device)

        while(1):
            while check_status(self.status_list, self.client_idx, 1) is False:
                time.sleep(0.1)
                    
            # model: mp_list --> server_network
            load_state_dict(self.server_network, self.state_dict_list[self.client_idx]) # model transfer
            self.train_lt(self.client_idx)     # local model updating 

            # self.networks[i] --> mp_list
            self.state_dict_list[self.client_idx] = return_state_dict(self.networks[self.client_idx])      # model transfer 
            set_status(self.status_list, self.client_idx, 2)


    def train_lt(self, idx):   
        """
        client-side code
        ---
        Argus:
        - idx: the index in all clients (e.g., 50) or selected clients (e.g., 10).
        If self.prefetch is true: the index in selected clients,
        If self.prefetch is true: the index in all clients
        """ 
        idx_in_all = idx

        # server broadcast the model to clients 
        """
        # optimizer will not work if use this, because optimizer needs the params from the model
        # self.networks[idx_in_all] = copy.deepcopy(self.server_network) 
        """
        for net_name, net in self.server_network.items():   # feat_model, classifier
            state_dict = self.networks[idx_in_all][net_name].state_dict()
            for key, layer in net.state_dict().items():
                state_dict[key].data.copy_(layer.data)

        for net in self.networks[idx_in_all].values():
            net.train()
        for net in self.server_network.values():
            net.train()          
        teacher = self.server_network

        # torch.cuda.empty_cache()

        """
        (Per-cls) Covariance Calculation
        """
        if self.feat_aug:
            # probability for augmentation for every class
            max_num = max(self.local_num_per_cls[idx])     
            prob = torch.tensor([1.0-i/max_num for i in self.local_num_per_cls[idx]])

            # obtain features and labels under eval mode
            feat_list, label_list = [], []
            
            # self.networks[idx_in_all]['feat_model'].eval()
            
            for (imgs, labels, indexs) in self.train_loaders[idx]: 
                with torch.no_grad():
                    imgs = imgs.to(self.device)
                    feat_list.append(teacher['feat_model'](imgs).cpu())
                    label_list.append(labels)
            feat_list = torch.cat(feat_list, 0)

            # self.networks[idx_in_all]['feat_model'].train()

            label_list = torch.cat(label_list, 0)
            unique_labels = list(np.unique(label_list))   # e.g., size (6, )
            transformed_label_list = torch.tensor([unique_labels.index(i) for i in label_list])     # e.g., size (n, )

            # per-cls features
            feats_per_cls = [[] for i in range(len(unique_labels))] 
            for feats, label in zip(feat_list, transformed_label_list):
                feats_per_cls[label].append(feats)

            # calculate the variance
            sampled_data, sample_label = [], []
            per_cls_cov = []
            for feats in feats_per_cls:
                if len(feats) > 1:
                    per_cls_cov.append(np.cov(torch.stack(feats, 1).numpy()))
                else:
                    per_cls_cov.append(np.zeros((feats[0].shape[0], feats[0].shape[0])))
            per_cls_cov = np.array(per_cls_cov)
            # per_cls_cov = np.array([np.cov(torch.stack(feats, 1).numpy()) for feats in feats_per_cls])
            cov = np.average(per_cls_cov, axis=0, weights=self.local_num_per_cls[idx])  # covariance for feature dimension, shape: e.g., (128, 128)

            # pre-generate deviation
            divider = 500
            pointer = 0
            augs = torch.from_numpy(np.random.multivariate_normal(
                mean = np.zeros(cov.shape[0]), 
                cov = cov,  # covariance for feature dimension, shape: e.g., (128, 128)
                size = divider)).float().to(self.device)


        with torch.set_grad_enabled(True):  
            losses_cls = 0
            losses_kd = 0

            ##########################
            #### stage 1 training ####
            ##########################
            for epoch in range(self.local_ep):   

                """
                model update
                """
                if self.local_ep > 10:  # locla training mode
                    print(epoch, end=' ')

                if self.balanced_loader:
                    tmp_loader = self.train_loader_balanced[idx]
                else:
                    tmp_loader = self.train_loaders[idx]
                for (imgs, labels, indexs) in tmp_loader:
                    # to device
                    imgs = imgs.to(self.device)

                    # forward
                    feat = self.networks[idx_in_all]['feat_model'](imgs)
                    logits = self.networks[idx_in_all]['classifier'](feat)

                    # do feature space augmentation with a likelihood
                    if self.feat_aug:
                        # prob = torch.tensor([1.0 for i in self.local_num_per_cls[idx]])
                        rand_list = torch.rand(len(labels))
                        mask = rand_list < prob[torch.tensor([unique_labels.index(i) for i in labels])]
                        degree = 1
                        aug_num = sum(mask).item()
                        if aug_num > 0: 
                            if pointer + aug_num >= divider:
                                pointer = 0
                            feat_aug = feat.clone()
                            feat_aug[mask] = feat_aug[mask] + augs[pointer: pointer+aug_num]*degree
                            logits_aug = self.networks[idx_in_all]['classifier'](feat_aug)
                            pointer = pointer + aug_num  

                    # teacher
                    with torch.no_grad():
                        feat_teacher = teacher['feat_model'](imgs)
                        pred_teacher = teacher['classifier'](feat_teacher)                      

                    # loss    
                    labels = labels.to(self.device)
                    if self.config["criterions"]["def_file"].find("LwF") > 0:      
                        if self.feat_aug and aug_num > 0:
                            if len(labels) != len(logits_aug):  
                                raise RuntimeError 
                            loss, loss_cls, loss_kd = self.criterion(labels, pred_teacher, logits, logits_aug)
                        else: 
                            loss, loss_cls, loss_kd = self.criterion(labels, pred_teacher, logits)
                    elif self.config["criterions"]["def_file"].find("KDLoss") > 0:
                        loss, loss_cls, loss_kd = self.criterion(
                            logits, labels, feat, feat_teacher, 
                            classfier_weight=self.networks[idx_in_all]['classifier'].fc.weight
                            )

                    # fedprox loss: https://epione.gitlabpages.inria.fr/flhd/federated_learning/FedAvg_FedProx_MNIST_iid_and_noniid.html#federated-training-with-fedprox
                    if self.fedprox:
                        prox_loss = difference_models_norm_2(self.networks[idx_in_all], teacher)
                        print("FedProx Loss: ", prox_loss, loss)
                        loss += self.mu/2 * prox_loss

                    # backward
                    for optimizer in self.optimizers[idx_in_all].values():
                        optimizer.zero_grad()
                    loss.backward()
                    for optimizer in self.optimizers[idx_in_all].values():
                        optimizer.step()

                    # classifier L2-norm
                    if self.networks[idx_in_all]['classifier'].l2_norm:
                        self.networks[idx_in_all]['classifier'].weight_norm()
                    losses_cls += loss_cls.item()
                    losses_kd += loss_kd.item()

            self.losses_cls[idx_in_all] = losses_cls/len(self.train_loaders[idx])/self.local_ep
            self.losses_kd[idx_in_all] = losses_kd/len(self.train_loaders[idx])/self.local_ep

            ##########################
            #### stage 2 training ####
            ##########################
            if self.crt:
                self.networks[idx_in_all]['feat_model'].eval()

                if self.feat_aug:
                    # obtain features and labels
                    feat_list = []
                    label_list = []
                    for (imgs, labels, indexs) in self.train_loaders[idx]:
                        imgs = imgs.to(self.device)
                        with torch.no_grad():
                            feat_list.append(self.networks[idx_in_all]['feat_model'](imgs).cpu())
                            label_list.append(labels)
                    feat_list = torch.cat(feat_list, 0)
                    label_list = torch.cat(label_list, 0)
                    unique_labels = list(np.unique(label_list))   # e.g., size (6, )
                    transformed_label_list = torch.tensor([unique_labels.index(i) for i in label_list])     # e.g., size (n, )

                    # per-cls features
                    feats_per_cls = [[] for i in range(len(unique_labels))] 
                    for feat, label in zip(feat_list, transformed_label_list):
                        feats_per_cls[label].append(feat)

                    # determine the extra sample number for every existing samples
                    num_per_cls = np.array([len(np.where(label_list==t)[0]) for t in unique_labels])     # e.g., size (6, )
                    max_num = max(num_per_cls)
                    gen_nums = [ np.array([max_num//num_per_cls[i]-1 for _ in feats_per_cls[i]]) for i in range(len(unique_labels))]
                    for cls_i, nums in enumerate(gen_nums):
                        nums[:max_num % num_per_cls[cls_i]] = nums[:max_num % num_per_cls[cls_i]] + 1

                    # generate samples
                    sampled_data, sample_label = [], []
                    per_cls_cov = np.array([np.cov(torch.stack(feats, 1).numpy()) for feats in feats_per_cls])
                    cov = np.average(per_cls_cov, axis=0, weights=num_per_cls)                  
                    # print([np.mean(i) for i in per_cls_cov])
                    for cls_i, nums in enumerate(gen_nums):
                        for sample_i, num in enumerate(nums):
                            if num > 0:
                                sampled_data.append(
                                    torch.from_numpy(np.random.multivariate_normal(
                                        mean = feats_per_cls[cls_i][sample_i],
                                        cov = cov,  # covariance for feature dimension, shape: e.g., (128, 128)
                                        size = num)).float())
                                sample_label.append(torch.full((num, ), cls_i).long())

                    # add generated fetaures to training data
                    feat_list = torch.cat([feat_list, *sampled_data], 0)
                    label_list = torch.cat([transformed_label_list, *sample_label], 0)
                    
                    # build new dataloader
                    feats_dataset = local_client_dataset(feat_list, label_list, self.config)
                    feats_loader = torch.utils.data.DataLoader(
                        feats_dataset, batch_size=self.local_bs, shuffle=True, num_workers=0, pin_memory=False)

                    # train classifier
                    for epoch in range(5):   
                        for (feats, labels, indexs) in feats_loader:
                            feats = feats.to(self.device)
                            labels = labels.to(self.device) 
                            logits = self.networks[idx_in_all]['classifier'](feats)
                            loss = torch.nn.CrossEntropyLoss()(logits[:, unique_labels], labels)

                            self.optimizers_stage2[idx_in_all].zero_grad()
                            loss.backward()
                            self.optimizers_stage2[idx_in_all].step()
                        # print(loss)

                # re-sampling without feature augmentation
                else:  
                    for epoch in range(5):   
                        for (imgs, labels, indexs) in self.train_loader_balanced[idx]:
                            # to device
                            imgs = imgs.to(self.device)
                            # forward
                            with torch.no_grad():
                                feat = self.networks[idx_in_all]['feat_model'](imgs)
                            logits = self.networks[idx_in_all]['classifier'](feat)

                            pos_cls = torch.unique(labels).tolist()
                            transformed_labels = torch.tensor([pos_cls.index(i) for i in labels]).to(self.device)
                            loss = torch.nn.CrossEntropyLoss()(logits[:, pos_cls], transformed_labels)

                            self.optimizers_stage2[idx_in_all].zero_grad()
                            loss.backward()
                            self.optimizers_stage2[idx_in_all].step()
                            # print(loss)

        print("=> ", end="")



def fedavg(w):
    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[k] += w[i][k]
        w_avg[k] = torch.div(w_avg[k]*1.0, len(w))
    return w_avg


# See: https://arxiv.org/abs/1909.06335
def fedavgm(new_ws, old_w, vel, args):
    """
    fedavg + momentum
    - new_ws (list of OrderedDict): The new calculated global model
    - old_w (OrderedDict) : Initial state of the global model (which needs to be updated here)  
    """
    global_lr = 1
    beta1 = 0
    
    new_w = fedavg(new_ws)

    # For the first round: initialize old_w, create an Orderdict to store velocity
    if old_w is None:
        old_w = new_w
        new_v = OrderedDict()
        for key in old_w.keys():
	        new_v[key] = torch.zeros(old_w[key].shape, dtype=old_w[key].dtype).to(args.device)
    else:
        new_v = copy.deepcopy(vel)

    for key in new_w.keys():
        delta_w_tmp = old_w[key] - new_w[key]
        new_v[key] = beta1*new_v[key] + torch.mul(delta_w_tmp, global_lr)
        old_w[key] -= new_v[key]

    return old_w, new_v


def fedavgw(new_ws, old_w, args, round_i):
    """
    fedavg + adaptive updating parameter
    - new_ws (list of OrderedDict): The new calculated global model
    - old_w (OrderedDict) : Initial state of the global model (which needs to be updated here)  
    """
    
    new_w = fedavg(new_ws)

    # For the first round: initialize old_w
    if old_w is None:
        old_w = new_w

    for key in new_w.keys():
        old_w[key] = new_w[key]*(1/(round_i+1)) +  old_w[key]*(round_i/(round_i+1))

    # for key in new_w.keys():
    #     if key == "classifier.fc.weight":
    #         old_w[key] = new_w[key]*(1/(round_i+1)) +  old_w[key]*(round_i/(round_i+1))
    #     else:
    #         old_w[key] = new_w[key]

    return old_w
