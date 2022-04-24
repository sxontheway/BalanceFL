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
    

def Speech_FL(root, config):
    """
    Divide Speech-Command dataset into small ones for FL.  
    shot_num: the number of n (n-shot)
    Return: 
    ---
    per_client_data, per_client_label: list of lists  
    test_data, test_label: both are lists  
    """    
    num_classes = config["networks"]["classifier"]["params"]["num_classes"]
    imb_ratio = config["dataset"]["imb_ratio"]
    
    # only invoke 'ImbalanceSpeech' for one time during the dataset generation, as it will cause multi-process problems
    data_file_path = f"./data/training_data.pt"
    if os.path.exists(data_file_path) is False:
        from data.ImbalanceSpeech import ImbalanceSpeech
        train_dataset = ImbalanceSpeech("training", imbalance_ratio=imb_ratio, root=root, reverse=None) # imb_ratio only for training set
        test_dataset = ImbalanceSpeech("testing", imbalance_ratio=None, root=root, reverse=None)
        val_dataset = ImbalanceSpeech("validation", imbalance_ratio=None, root=root, reverse=None)
    else:
        # train_dataset = ImbalanceSpeech_clean("training", imbalance_ratio=imb_ratio, root=root, reverse=None)   # imb_ratio only for training set
        # val_dataset = ImbalanceSpeech_clean("validation", imbalance_ratio=None, root=root, reverse=None)
        # test_dataset = ImbalanceSpeech_clean("testing", imbalance_ratio=None, root=root, reverse=None)

        # training
        train_data_all, train_label_all, train_num_per_cls = \
            ImbalanceSpeech_clean("training", imb_ratio, root)

        # validation
        val_data, val_label, val_num_per_cls = \
            ImbalanceSpeech_clean("validation", None, root)

        # testing
        test_data, test_label, test_num_per_cls = \
            ImbalanceSpeech_clean("testing", None, root)

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
    return per_client_data, per_client_label, \
        test_data, test_label, val_data, val_label, \
            cls_per_client, num_per_cls_per_client, train_num_per_cls


def ImbalanceSpeech_clean(mode, imbalance_ratio, root, imb_type='exp', test_imb_ratio=None, reverse=False):
    """
    mode: training, testing, validation  
    imbalance_ratio: only for training
    root: data folder  

    Important paramters
    ---
    Return: (all are lists)
    data: [a, b, ...], a and b are tensor of 1*32*32
    labels: [a, b, ...], a and b are scalar
    """
    
    cls_num = 35
    data_file_path = f"./data/{mode}_data.pt"
    target_file_path = f"./data/{mode}_target.pt"

    if os.path.exists(data_file_path) is False:
        raise RuntimeError

    data = torch.load(data_file_path)
    targets = torch.load(target_file_path)

    # obtain the distribution
    num_per_cls = [0 for i in range(cls_num)]
    for label in targets:
        num_per_cls[label] += 1
    print(mode, "original num_per_cls:", num_per_cls)

    # change the distribution to long-tail
    if mode == "training":
        selected_num_per_cls = get_num_per_cls(num_per_cls, imb_type, imbalance_ratio, reverse=reverse)
        data, targets = gen_imbalanced_data(data, targets, selected_num_per_cls)
        print("selected num_per_cls:", mode, selected_num_per_cls)
    else:
        selected_num_per_cls = num_per_cls

    # adjust the data according to max/min/mean ~ (-68, 51, -18)
    min, max, mean = torch.min(data), torch.max(data), torch.mean(data)
    print("min/max/mean:", min, max, mean)
    data = (data + 9)/61
    print("data/target shape:", data.shape, targets.shape, "\n")

    # change data and targets into list
    data_list, targets_list = [], []
    for i, j in zip(data, targets):
        data_list.append(i)
        targets_list.append(int(j))

    return data_list, targets_list, selected_num_per_cls
    

def gen_imbalanced_data(data, targets, selected_num_per_cls):
    """
    self.data: numpy.array
    self.targets: list
    """
    new_data = []
    new_targets = []
    targets_np = np.array(targets, dtype=np.int64)
    classes = np.unique(targets_np)

    num_per_cls_dict = dict()
    for the_class, the_img_num in zip(classes, selected_num_per_cls):
        num_per_cls_dict[the_class] = the_img_num
        idx = np.where(targets_np == the_class)[0]
        # print(the_class, the_img_num, idx)
        np.random.shuffle(idx)
        selec_idx = idx[:the_img_num]
        new_data.append(data[selec_idx, ...])
        new_targets.extend([the_class, ] * the_img_num)
    data = torch.cat(new_data, 0)        # n*32*32*3
    targets = torch.tensor(new_targets)   # list of length n
    return data, targets


def get_num_per_cls(num_per_cls, imb_type, imb_factor, reverse=False):
        cls_num = len(num_per_cls)
        img_max = 3000      # manually defined
        img_min = min(num_per_cls)

        selected_num_per_cls = []
        if imb_factor == 1:
            selected_num_per_cls = [img_min for i in range(cls_num)]

        elif imb_factor <= 0.1:
            if imb_type == 'exp':
                for cls_idx in range(cls_num):
                    if reverse:
                        num = img_max * (imb_factor**((cls_num - 1 - cls_idx) / (cls_num - 1.0)))
                    else:
                        num = img_max * (imb_factor**(cls_idx / (cls_num - 1.0)))
                    selected_num_per_cls.append(int(num))
            elif imb_type == 'step':
                for cls_idx in range(cls_num // 2):
                    selected_num_per_cls.append(int(img_max))
                for cls_idx in range(cls_num // 2):
                    selected_num_per_cls.append(int(img_max * imb_factor))
            else:
                selected_num_per_cls.extend([int(img_max)] * cls_num)
        return selected_num_per_cls


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