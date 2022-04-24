import torch
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data.sampler import WeightedRandomSampler
from torchvision import transforms
import os, random
from PIL import Image


# Data transformation with augmentation
def get_data_transform(split, rgb_mean, rbg_std, key='default'):
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(rgb_mean, rbg_std)
        ]) if key == 'iNaturalist18' else transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0),
            transforms.ToTensor(),
            transforms.Normalize(rgb_mean, rbg_std)
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(rgb_mean, rbg_std)
        ]),
        'test': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(rgb_mean, rbg_std)
        ])
    }
    return data_transforms[split]


def pil_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


def non_iidness_cal(labels, idx_per_client, img_per_client):
        """
        Argu:
            labels: list with length of n, where n is the dataset size.
            idx_per_client: list. idx_per_client[i] is the img idx in the dataset for client i
            img_per_client: list. Number of images per client.
        Return:
            - non_iidness
        """
        client_num = len(idx_per_client)
        class_num = max(labels)+1
        label_per_client_count = np.zeros((client_num, class_num))

        # generate per client label counting
        labels = np.array(labels)
        for i in range(client_num):
            count = np.bincount(labels[idx_per_client[i]])
            count_pad = np.pad(count, (0, class_num-len(count)), 'constant', constant_values=(0,0))
            label_per_client_count[i] += count_pad

        # obtain non_iidness 
        summation = 0
        label_per_client_count /= np.array([img_per_client]).T  # broadcast
        for i, client_i in enumerate(label_per_client_count):
            for client_j in label_per_client_count[i:]:
                summation += np.linalg.norm(client_i-client_j, ord=1)
        
        non_iidness = summation/(client_num*(client_num-1))

        return non_iidness


def tao_sampling(img_per_client, tao):
    """
    Do non-iid or iid sampling, according to "tao". 
    We will sample number of "tao" images for every client in turn. 
    --- 
    Argu:
        - img_per_client: list. Number of images per client.
        - tao: number of sampled image for each client in each round. 
        We use tao to control the non-iidness. When tao==1, nearly iid; 
        when tao is large, it becomes non-iid. 
        "tao <= min(img_per_client/2)" to let each client has at least 2 classes
    Return:
        - idx_per_client: list. idx_per_client[i] is the img idx in the dataset for client i
    """
    # prepare parameters
    total_img_num = sum(img_per_client)
    client_num = len(img_per_client)
    idx_per_client = [[] for i in range(client_num)]
    # assert tao <= min(img_per_client)/2 
    
    available_per_client = img_per_client
    tao_count = 0
    client_k = 0
    idx = 0
    client_count = 0
    client_order = [*range(client_num)]

    while idx < total_img_num:      # assign every samples to a client

        client_k = client_order[client_count]
        if available_per_client[client_k] > 0 and tao_count < tao:
            idx_per_client[client_k].append(total_img_num-idx-1)    # reverse the head and tail
            tao_count += 1
            idx += 1
            available_per_client[client_k] -= 1
        
        # the client is already full, or tao samples are already assigned
        else:
            client_count = client_count + 1
            # shuffle the order of clients if a round is finished
            if client_count >= client_num:
                random.shuffle(client_order)
                client_count = 0
            tao_count = 0
            continue

    return idx_per_client


def gen_fl_data(train_label_all, num_per_cls, config):
    """
    Generate distributed data for FL training.
    ---
    Argu:
        - train_label_all: object of a class inheriting from torch.utils.data.Dataset 
            Or a list pre-stored in the RAM.
        - config: configuration dictionary
    Return:
        - idx_per_client: list. The i^th item is the img idx of the training set for client i
        - tao: int
        - non_iidness: the calculated non_iidness
    """      
    # generate img_per_client
    client_num = config["fl_opt"]["num_clients"]
    img_per_client_dist = config["dataset"]["img_per_client_dist"]
    total_img_num = len(train_label_all)
    if img_per_client_dist == "uniform":
        img_per_client = np.full(client_num, total_img_num//client_num)
        img_per_client[:total_img_num % client_num] += 1
    else:    # use other img_per_client distributions: normal, LT, reverse LT
        pass

    # iid: tao=1; non_iid: tao=max(img_per_client)
    non_iidness_degree = config["dataset"]["non_iidness"]
    # tao_max = min(img_per_client)#//2
    # tao = round(1 + non_iidness_degree*(tao_max-1))
    tao = int(config["dataset"]["tao_ratio"] * num_per_cls[-1])
    idx_per_client = tao_sampling(img_per_client.copy(), tao)

    # calculate the real non_iidness on training set
    non_iidness = non_iidness_cal(train_label_all, idx_per_client, img_per_client)
    
    # classes per client
    cls_per_client = []
    num_per_cls_per_client = []
    for idxs in idx_per_client:
        cls, tmp = np.unique(np.array(train_label_all)[idxs], return_counts=True)
        num_per_cls = np.zeros(config["dataset"]["num_classes"], dtype=np.int)
        np.put_along_axis(num_per_cls, cls, tmp, axis=0)
        cls_per_client.append(cls)
        num_per_cls_per_client.append(num_per_cls)

    return idx_per_client, tao, non_iidness, cls_per_client, num_per_cls_per_client
    

from data.ImbalanceCIFAR import IMBALANCECIFAR10, IMBALANCECIFAR100
def load_CIFAR(root, cifar_select, train, num_classes, shot_num):
    """
    Load dataset CIFAR into memory. Shot version.
    """
    if num_classes > 10 and cifar_select == "CIFAR10":
        raise RuntimeError

    if train:
        if cifar_select == "CIFAR100":
            dataset = IMBALANCECIFAR100(
                "train", imbalance_ratio=1, root=root, test_imb_ratio=None, reverse=None)
        elif cifar_select == "CIFAR10":
            dataset = IMBALANCECIFAR10(
                "train", imbalance_ratio=1, root=root, test_imb_ratio=None, reverse=None)
        else:
            raise RuntimeError
    else:
        if cifar_select == "CIFAR100":
            dataset = IMBALANCECIFAR100(
                "test", imbalance_ratio=1, root=root, test_imb_ratio=None, reverse=None)
        elif cifar_select == "CIFAR10":
            dataset = IMBALANCECIFAR10(
                "test", imbalance_ratio=1, root=root, test_imb_ratio=None, reverse=None)
        else:
            raise RuntimeError
    
    ###############################################
    ####### load the whole dataset into RAM #######
    ###############################################
    # without transformation
    if cifar_select == "CIFAR10":       # 5000*10+1000*10
        num_per_cls = 5000
        if not train:
            num_per_cls = 1000
            shot_num = 1000
    elif cifar_select == "CIFAR100":    # 500*100+100*100
        num_per_cls = 500   
        if not train:
            num_per_cls = 100
            shot_num = 100

    # transformation: data are pre-loaded and do not support augmentation
    train_transform = transforms.Compose([ 
        # transforms.RandomCrop(32, padding=4),     
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    all_imgs = []
    all_targets = []
    cnt = 0
    for img, label in zip(dataset.data, dataset.labels):
        cnt += 1
        if train and cnt % num_per_cls >= shot_num:
            continue
        if train:
            all_imgs.append(train_transform(Image.fromarray(img)))
        else:
            all_imgs.append(test_transform(Image.fromarray(img)))
        all_targets.append(label)

    return all_imgs, all_targets


def load_CIFAR_imb(root, cifar_select, train, num_classes, imb_ratio):
    """
    Load CIFAR into memory. Imbalance version.
    """
    if num_classes > 10 and cifar_select == "CIFAR10":
        raise RuntimeError

    if train:
        if cifar_select == "CIFAR100":
            dataset = IMBALANCECIFAR100(
                "train", imbalance_ratio=imb_ratio, root=root, test_imb_ratio=None, reverse=None,
            )
        elif cifar_select == "CIFAR10":
            dataset = IMBALANCECIFAR10(
                "train", imbalance_ratio=imb_ratio, root=root, test_imb_ratio=None, reverse=None,
            )
        else:
            raise RuntimeError
    else:
        if cifar_select == "CIFAR100":
            dataset = IMBALANCECIFAR100(
                "test", imbalance_ratio=1, root=root, test_imb_ratio=None, reverse=None
            )
        elif cifar_select == "CIFAR10":
            dataset = IMBALANCECIFAR10(
                "test", imbalance_ratio=1, root=root, test_imb_ratio=None, reverse=None
            )
        else:
            raise RuntimeError

    print("Number of items per class: ", dataset.get_cls_num_list())

    ###############################################
    ####### load the whole dataset into RAM #######
    ###############################################

    # transformation: data are pre-loaded and do not support augmentation
    train_transform = transforms.Compose(
        [
            # transforms.RandomCrop(32, padding=4),
            # transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )
    test_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )

    all_imgs = []
    all_targets = []
    cnt = 0
    for img, label in zip(dataset.data, dataset.labels):
        cnt += 1
        if train:
            all_imgs.append(train_transform(Image.fromarray(img)))
        else:
            all_imgs.append(test_transform(Image.fromarray(img)))
        all_targets.append(label)
    return all_imgs, all_targets, dataset.get_cls_num_list()


def CIFAR_FL(root, config):
    """
    Divide CIFAR dataset into small ones for FL.  
    mode: base, novel or all classes
    shot_num: the number of n (n-shot)
    Return: 
    ---
    per_client_data, per_client_label: list of lists  
    test_data, test_label: both are lists  
    """    
    # shot_num = config['dataset']['shot']
    # assert shot_num != 0

    num_classes = config["networks"]["classifier"]["params"]["num_classes"]
    imb_ratio = config["dataset"]["imb_ratio"]

    # training
    cifar_select = config["dataset"]["name"]
    train_data_all, train_label_all, train_num_per_cls = load_CIFAR_imb(   
        root, cifar_select, train=True, num_classes=num_classes, imb_ratio=imb_ratio
        )
    # test
    test_data, test_label, test_num_per_cls = load_CIFAR_imb(
        root, cifar_select, train=False, num_classes=num_classes, imb_ratio=imb_ratio
        )

    # generate per-client FL data
    idx_per_client, tao, non_iidness, cls_per_client, num_per_cls_per_client \
        = gen_fl_data(train_label_all, train_num_per_cls, config)

    client_num = config["fl_opt"]["num_clients"]
    per_client_data = [[] for i in range(client_num)]
    per_client_label = [[] for i in range(client_num)]
    for client_i in range(client_num):
        for j in idx_per_client[client_i]:
            per_client_data[client_i].append(train_data_all[j]) 
            per_client_label[client_i].append(train_label_all[j]) 

    print("tao:", tao, "non-iidness:", non_iidness)
    return per_client_data, per_client_label, test_data, test_label, cls_per_client, num_per_cls_per_client, train_num_per_cls


def CIFAR_FL_mixed(root, config):
    """
    Divide CIFAR dataset into small ones for FL.  
    (iid + many shot) for half of all classes; (non-iid + few shot) for remaining half classes.
    mode: base, novel or all classes
    shot_num: the number of n (n-shot)
    Return: 
    ---
    per_client_data, per_client_label: list of lists  
    test_data, test_label: both are lists  
    """    
    shot_num = config['dataset']['shot']
    few_shot_num = config['dataset']['shot_few']
    assert (shot_num != 0 and few_shot_num != 0)

    num_classes = config["networks"]["classifier"]["params"]["num_classes"]

    # training
    cifar_select = config["dataset"]["name"]
    train_data_all, train_label_all = load_CIFAR(
        root, cifar_select, train=True, num_classes=num_classes, shot_num=shot_num
        )
    # test
    test_data, test_label = load_CIFAR(
        root, cifar_select, train=False, num_classes=num_classes, shot_num=shot_num
        )

    # per-client FL data for the first half (iid + many shot) classes
    half_data_len = int(len(train_label_all)/2)
    iid_train_data_all, iid_train_label_all = train_data_all[:half_data_len], train_label_all[:half_data_len]
    config["dataset"]["non_iidness"] = 0
    iid_idx_per_client, tao, non_iidness, iid_cls_per_client = gen_fl_data(iid_train_label_all, config)
    print("IID, tao:", tao, "non-iidness:", non_iidness)

    # per-client FL data for the remaining half (non-iid + few shot) classes
    noniid_train_data_all = []
    noniid_train_label_all = []
    cnt = 0
    for img, label in zip(train_data_all[half_data_len:], train_label_all[half_data_len:]):
        cnt += 1
        if cnt % shot_num >= few_shot_num:
            continue
        noniid_train_data_all.append(img)
        noniid_train_label_all.append(label)
    config["dataset"]["non_iidness"] = 1
    noniid_idx_per_client, tao, non_iidness, noniid_cls_per_client = gen_fl_data(noniid_train_label_all, config)
    print("Non-IID, tao:", tao, "non-iidness:", non_iidness)

    # iid + non-iid combination
    client_num = config["fl_opt"]["num_clients"]
    per_client_data = [[] for i in range(client_num)]
    per_client_label = [[] for i in range(client_num)]
    for client_i in range(client_num):
        for j in iid_idx_per_client[client_i]:
            per_client_data[client_i].append(iid_train_data_all[j]) 
            per_client_label[client_i].append(iid_train_label_all[j]) 
        for j in noniid_idx_per_client[client_i]:
            per_client_data[client_i].append(noniid_train_data_all[j]) 
            per_client_label[client_i].append(noniid_train_label_all[j]) 

    cls_per_client = []
    for iid_cls, noniid_cls in zip(iid_cls_per_client, noniid_cls_per_client):
        cls_per_client.append(np.concatenate((iid_cls, noniid_cls)))

    return per_client_data, per_client_label, test_data, test_label, cls_per_client


from matplotlib import pyplot as plt 
class local_client_dataset(Dataset):
    def __init__(self, per_client_data, per_client_label, config, aug=False):
        self.data = per_client_data
        self.label = per_client_label
        self.dataset_name = config["dataset"]["name"]
        self.aug = aug
        if self.dataset_name == "CUB":
            self.train_transform = transforms.Compose(
                [transforms.Resize(256),
                transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.4856, 0.4994, 0.4324], std=[0.2321, 0.2277, 0.2665])
                ])

        elif self.dataset_name in ["CIFAR10", "CIFAR100"]:
            self.train_transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])

    
    def __len__(self):
        return len(self.label)

    def __getitem__(self, index):        
        data = self.data[index]
        # plt.imshow(data); plt.savefig("./a.jpg")

        if self.dataset_name == "CUB":
            return self.val_transform(data), self.label[index], index
        else:
            if self.aug:
                return self.train_transform(data), self.label[index], index
            else:
                return data, self.label[index], index

    def get_balanced_sampler(self):
        labels = np.array(self.label)       # e.g., size (n, )
        unique_labels = list(np.unique(labels))   # e.g., size (6, )
        transformed_labels = torch.tensor([unique_labels.index(i) for i in labels])             # e.g., size (n, )
        class_sample_count = np.array([len(np.where(labels==t)[0]) for t in unique_labels])     # e.g., size (6, )
        weight = 1. / class_sample_count    # make every class to have balanced chance to be chosen
        samples_weight = torch.tensor([weight[t] for t in transformed_labels])
        self.class_sample_count = class_sample_count
        sampler = WeightedRandomSampler(samples_weight.type('torch.DoubleTensor'), len(samples_weight))
        return sampler


class test_dataset(Dataset):
    def __init__(self, per_client_data, per_client_label, config):
        self.data = per_client_data
        self.label = per_client_label
        self.dataset = config["dataset"]["name"]
        if self.dataset == "CUB":
            self.val_transform = transforms.Compose(
                [transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.4856, 0.4994, 0.4324], std=[0.2321, 0.2277, 0.2665])
                ])

    def __len__(self):
        return len(self.label)

    def __getitem__(self, index):
        
        data = self.data[index]
        # plt.imshow(data)
        # plt.savefig("./a.jpg")

        if self.dataset == "CUB":
            return self.val_transform(data), self.label[index], index
        else:
            return data, self.label[index], index