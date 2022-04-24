import os
import random

import numpy as np
import torch
from matplotlib import pyplot as plt
from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data.sampler import WeightedRandomSampler
from torchvision import transforms


# Data transformation with augmentation
def get_data_transform(split, rgb_mean, rbg_std, key="default"):
    data_transforms = {
        "train": transforms.Compose(
            [
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(rgb_mean, rbg_std),
            ]
        )
        if key == "iNaturalist18"
        else transforms.Compose(
            [
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(
                    brightness=0.4, contrast=0.4, saturation=0.4, hue=0
                ),
                transforms.ToTensor(),
                transforms.Normalize(rgb_mean, rbg_std),
            ]
        ),
        "val": transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(rgb_mean, rbg_std),
            ]
        ),
        "test": transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(rgb_mean, rbg_std),
            ]
        ),
    }
    return data_transforms[split]


def pil_loader(path):
    with open(path, "rb") as f:
        img = Image.open(f)
        return img.convert("RGB")


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
    class_num = max(labels) + 1
    label_per_client_count = np.zeros((client_num, class_num))

    # generate per client label counting
    labels = np.array(labels)
    for i in range(client_num):
        count = np.bincount(labels[idx_per_client[i]])
        count_pad = np.pad(
            count, (0, class_num - len(count)), "constant", constant_values=(0, 0)
        )
        label_per_client_count[i] += count_pad

    # obtain non_iidness
    summation = 0
    label_per_client_count /= np.array([img_per_client]).T  # broadcast
    for i, client_i in enumerate(label_per_client_count):
        for client_j in label_per_client_count[i:]:
            summation += np.linalg.norm(client_i - client_j, ord=1)

    non_iidness = summation / (client_num * (client_num - 1))

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

    while idx < total_img_num:  # assign every samples to a client

        client_k = client_order[client_count]
        if available_per_client[client_k] > 0 and tao_count < tao:
            idx_per_client[client_k].append(
                total_img_num - idx - 1
            )  # reverse the head and tail
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
        img_per_client = np.full(client_num, total_img_num // client_num)
        img_per_client[: total_img_num % client_num] += 1
    else:  # use other img_per_client distributions: normal, LT, reverse LT
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


def IMU_FL(root, config):
    data_dim = config["networks"]["feat_model"]["data_dim"]
    train_dataset = np.load(
        os.path.join(root, "aiot_train.npy"), allow_pickle=True
    ).item()
    test_dataset = np.load(
        os.path.join(root, "aiot_test.npy"), allow_pickle=True
    ).item()
    for i in range(len(train_dataset["data"])):
        train_dataset["data"][i] = train_dataset["data"][i][:, :data_dim].reshape(-1)
    for i in range(len(test_dataset["data"])):
        test_dataset["data"][i] = test_dataset["data"][i][:, :data_dim].reshape(-1)

    test_data = list(np.array(test_dataset["data"], dtype=np.float32))
    test_label = test_dataset["label"]

    client_num = config["fl_opt"]["num_clients"]
    per_client_data = []
    per_client_label = []
    cls_per_client = []
    num_per_cls_per_client = []
    train_num_per_cls = np.unique(train_dataset["label"], return_counts=True)[1]

    for client_i in range(client_num):
        pid = np.array(train_dataset["person_id"])
        data = np.array(train_dataset["data"], dtype=np.float32)
        label = np.array(train_dataset["label"])
        per_client_data.append(list(data[pid == client_i]))
        per_client_label.append(list(label[pid == client_i]))
        clss, num = np.unique(per_client_label[-1], return_counts=True)
        num_per_cls = np.zeros(config["dataset"]["num_classes"], dtype=np.int)
        np.put_along_axis(num_per_cls, clss, num, axis=0)
        cls_per_client.append(clss)
        num_per_cls_per_client.append(num_per_cls)
    return (
        per_client_data,
        per_client_label,
        test_data,
        test_label,
        cls_per_client,
        num_per_cls_per_client,
        train_num_per_cls,
    )


class local_client_dataset(Dataset):
    def __init__(self, per_client_data, per_client_label, config, aug=False):
        self.data = per_client_data
        self.label = per_client_label
        self.dataset_name = config["dataset"]["name"]
        self.aug = aug
        if self.dataset_name == "CUB":
            self.train_transform = transforms.Compose(
                [
                    transforms.Resize(256),
                    transforms.RandomCrop(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.4856, 0.4994, 0.4324], std=[0.2321, 0.2277, 0.2665]
                    ),
                ]
            )

        elif self.dataset_name in ["CIFAR10", "CIFAR100"]:
            self.train_transform = transforms.Compose(
                [
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                    ),
                ]
            )

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
        labels = np.array(self.label)  # e.g., size (n, )
        unique_labels = list(np.unique(labels))  # e.g., size (6, )
        transformed_labels = torch.tensor(
            [unique_labels.index(i) for i in labels]
        )  # e.g., size (n, )
        class_sample_count = np.array(
            [len(np.where(labels == t)[0]) for t in unique_labels]
        )  # e.g., size (6, )
        weight = (
            1.0 / class_sample_count
        )  # make every class to have balanced chance to be chosen
        samples_weight = torch.tensor([weight[t] for t in transformed_labels])
        self.class_sample_count = class_sample_count
        sampler = WeightedRandomSampler(
            samples_weight.type("torch.DoubleTensor"), len(samples_weight)
        )
        return sampler


class test_dataset(Dataset):
    def __init__(self, per_client_data, per_client_label, config):
        self.data = per_client_data
        self.label = per_client_label
        self.dataset = config["dataset"]["name"]
        if self.dataset == "CUB":
            self.val_transform = transforms.Compose(
                [
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.4856, 0.4994, 0.4324], std=[0.2321, 0.2277, 0.2665]
                    ),
                ]
            )

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
