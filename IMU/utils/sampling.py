import numpy as np
from torch.utils.data import Dataset
import random

class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label, index = self.dataset[self.idxs[item]]
        return image, label, index


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
    assert tao <= min(img_per_client)/2 
    
    available_per_client = img_per_client
    tao_count = 0
    client_k = 0
    idx = 0
    client_count = 0
    client_order = [*range(client_num)]

    while idx < total_img_num:      # assign every samples to a client

        client_k = client_order[client_count]
        if available_per_client[client_k] > 0 and tao_count < tao:
            idx_per_client[client_k].append(idx)
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


def gen_fl_data(dataset, config):
    """
    Generate distributed data for FL training.
    ---
    Argu:
        - dataset: object of a class inheriting from torch.utils.data.Dataset 
        - config: configuration dictionary
    Return:
        - idx_per_client: list. The i^th item is the img idx of the training set for client i
        - tao: int
        - non_iidness: the calculated non_iidness
    """      
    # generate img_per_client
    client_num = config["fl_opt"]["num_clients"]
    img_per_client_dist = config["dataset"]["img_per_client_dist"]
    total_img_num = len(dataset.labels)
    if img_per_client_dist == "uniform":
        img_per_client = np.full(client_num, total_img_num//client_num)
        img_per_client[:total_img_num % client_num] += 1
    else:    # use other img_per_client distributions: normal, LT, reverse LT
        pass

    # iid: tao=1; non_iid: tao=max(img_per_client)
    non_iidness_degree = config["dataset"]["non_iidness"]
    tao_max = min(img_per_client)//2
    tao = round(1 + non_iidness_degree*(tao_max-1))
    idx_per_client = tao_sampling(img_per_client.copy(), tao)

    # calculate the real non_iidness on training set
    non_iidness = non_iidness_cal(dataset.labels, idx_per_client, img_per_client)
    
    # classes per client
    cls_per_client = []
    for idxs in idx_per_client:
        cls_per_client.append(np.unique(np.array(dataset.labels)[idxs]))

    return idx_per_client, tao, non_iidness, cls_per_client


def gen_ptest_data(train_dataset, train_idx_per_client, test_dataset):
    """
    Generate personalized test data for every client.
    ---
    Argu:
        - train_dataset: object of a class inheriting from torch.utils.data.Dataset 
        - train_idx_per_client: see function "gen_fl_data"
        - test_dataset: the same with train_dataset
    Return:
        - cls_per_client: list. 
        - test_idx_per_client: list. The i^th item is the img idx of the test set for client i
    """ 
    # classes per client
    cls_per_client = []
    for idxs in train_idx_per_client:
        cls_per_client.append(np.unique(np.array(train_dataset.labels)[idxs]))

    # bkpt stores the idx of the starter of every class
    bkpt = [0]
    for num_of_sample_this_cls in test_dataset.img_num_list:   # samples per cls 
        bkpt.append(bkpt[-1] + num_of_sample_this_cls)

    # idx for every client
    test_idx_per_client = []
    for classes in cls_per_client:
        # idx for a client
        idx = []
        for class_i in classes:
            idx += [*range(bkpt[class_i], bkpt[class_i+1])]
        test_idx_per_client.append(idx)

    return cls_per_client, test_idx_per_client