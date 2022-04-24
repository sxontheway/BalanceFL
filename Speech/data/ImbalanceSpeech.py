"""
Adopted from https://github.com/Megvii-Nanjing/BBN
Customized by Kaihua Tang
"""

import torchaudio.transforms as transforms_au
import numpy as np
from PIL import Image
from torchaudio.datasets import SPEECHCOMMANDS
import librosa
import os
import torch


class SubsetSC(SPEECHCOMMANDS):
    def __init__(self, subset, root):
        super().__init__(root, download=True)

        def load_list(filename):
            filepath = os.path.join(self._path, filename)
            with open(filepath) as fileobj:
                return [os.path.join(self._path, line.strip()) for line in fileobj]

        if subset == "validation":
            self._walker = load_list("validation_list.txt")
        elif subset == "testing":
            self._walker = load_list("testing_list.txt")
        elif subset == "training":
            excludes = load_list("validation_list.txt") + load_list("testing_list.txt")
            excludes = set(excludes)
            self._walker = [w for w in self._walker if w not in excludes]


class ImbalanceSpeech():

    def __init__(self, mode, imbalance_ratio, root, imb_type='exp', test_imb_ratio=None, reverse=False):
        """
        mode: training, testing, validation  
        imbalance_ratio: only for training
        root: data folder  

        Important paramters
        ---
        Generate: self.data, self.labels (both torch.tensor), self.num_per_cls
        
        Original Dataset:
        ---
        Train # Counter({'zero': 3250, 'five': 3240, 'yes': 3228, 'seven': 3205, 'nine': 3170, 'one': 3140, 'down': 3134, 'no': 3130, 'stop': 3111, 'two': 3111, 'go': 3106, 'six': 3088, 'on': 3086, 'left': 3037, 'eight': 3033, 'right': 3019, 'off': 2970, 'three': 2966, 'four': 2955, 'up': 2948, 'house': 1727, 'wow': 1724, 'dog': 1711, 'marvin': 1710, 'bird': 1697, 'cat': 1657, 'happy': 1632, 'sheila': 1606, 'bed': 1594, 'tree': 1407, 'backward': 1346, 'visual': 1288, 'learn': 1286, 'follow': 1275, 'forward': 1256}) 
       
        Val: # Counter({'no': 406, 'yes': 397, 'seven': 387, 'zero': 384, 'six': 378, 'down': 377, 'off': 373, 'four': 373, 'go': 372, 'five': 367, 'right': 363, 'on': 363, 'nine': 356, 'three': 356, 'left': 352, 'one': 351, 'stop': 350, 'up': 350, 'eight': 346, 'two': 345, 'happy': 219, 'bed': 213, 'sheila': 204, 'dog': 197, 'marvin': 195, 'house': 195, 'wow': 193, 'bird': 182, 'cat': 180, 'tree': 159, 'backward': 153, 'forward': 146, 'visual': 139, 'follow': 132, 'learn': 128}) 
        
        Test: # Counter({'five': 445, 'up': 425, 'two': 424, 'yes': 419, 'zero': 418, 'left': 412, 'stop': 411, 'eight': 408, 'nine': 408, 'seven': 406, 'down': 406, 'no': 405, 'three': 405, 'go': 402, 'off': 402, 'four': 400, 'one': 399, 'right': 396, 'on': 396, 'six': 394, 'dog': 220, 'sheila': 212, 'bed': 207, 'wow': 206, 'happy': 203, 'marvin': 195, 'cat': 194, 'tree': 193, 'house': 191, 'bird': 185, 'follow': 172, 'backward': 165, 'visual': 165, 'learn': 161, 'forward': 155})
        """
        
        cls_num = 35

        # label_list = sorted(list(set(datapoint[2] for datapoint in val_set)))    # sort by name
        label_list= ['zero', 'five', 'yes', 'seven', 'nine', 'one', 'down', 'no', 'stop', 'two', 'go', 'six', 'on', 'left', 'eight', 'right', 'off', 'three', 'four', 'up', 'house', 'wow', 'dog', 'marvin', 'bird', 'cat', 'happy', 'sheila', 'bed', 'tree', 'backward', 'visual', 'learn', 'follow', 'forward']   # sort by training set amount
        label_dict= dict(zip(label_list, [*range(cls_num)]))  

        self.data = []
        self.targets = []
        data_file_path = f"./data/{mode}_data.pt"
        target_file_path = f"./data/{mode}_target.pt"

        # fetch for the first time
        if os.path.exists(data_file_path) is False:

            train_set = SubsetSC("training", root)
            test_set = SubsetSC("testing", root)
            val_set = SubsetSC("validation", root)
            dataset_dict = {"training": train_set, "testing": test_set, "validation": val_set}

            for i in range(len(dataset_dict[mode])):   # len(dataset_dict[mode])
                # load
                waveform, sample_rate, label, speaker_id, utterance_number = dataset_dict[mode][i]
                # transform
                if waveform.shape[1] // 1024 > 8:
                    n_fft = 1024; hop_length = 512
                else:
                    n_fft = 512; hop_length = 256
                mel_spectrogram = transforms_au.MelSpectrogram(
                    sample_rate=sample_rate,
                    n_fft=n_fft,
                    win_length=None,
                    hop_length=hop_length,
                    power=2.0,
                    n_mels=32,  # how many rows
                )
                melspec = librosa.power_to_db(mel_spectrogram(waveform))
                if melspec.shape[2] != 32:
                    melspec = np.resize(melspec, (1, 32, 32))

                # append
                melspec = torch.from_numpy(melspec)
                if torch.sum(torch.isnan(melspec)) == 0:
                    self.data.append(melspec)  
                    self.targets.append(label_dict[label])
                else:
                    raise RuntimeError
                print("Doing FFT", i)

            self.data = torch.stack(self.data, 0)     # n*1*32*32
            self.targets = torch.tensor(self.targets)
            torch.save(self.data, data_file_path)
            torch.save(self.targets, target_file_path)

        # already fetched, load from file
        else:
            self.data = torch.load(data_file_path)
            self.targets = torch.load(target_file_path)

        # obtain the distribution
        self.num_per_cls = [0 for i in range(cls_num)]
        for label in self.targets:
            self.num_per_cls[label] += 1
        print(mode, "original num_per_cls:", self.num_per_cls)

        # change the distribution to long-tail
        if mode == "training":
            self.selected_num_per_cls = self.get_num_per_cls(self.num_per_cls, imb_type, imbalance_ratio, reverse=reverse)
            self.gen_imbalanced_data()
            print("selected num_per_cls:", mode, self.selected_num_per_cls)
        self.labels = self.targets

        # adjust the data according to max/min/mean ~ (-68, 51, -18)
        min, max, mean = torch.min(self.data), torch.max(self.data), torch.mean(self.data)
        print("min/max/mean:", min, max, mean)
        self.data = (self.data + 9)/61
        
        print("data/target shape:", self.data.shape, self.labels.shape, "\n")
        

    def _get_class_dict(self):
        class_dict = dict()
        for i, anno in enumerate(self.get_annotations()):
            cat_id = anno["category_id"]
            if not cat_id in class_dict:
                class_dict[cat_id] = []
            class_dict[cat_id].append(i)
        return class_dict


    def get_num_per_cls(self, num_per_cls, imb_type, imb_factor, reverse=False):
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


    def gen_imbalanced_data(self):
        """
        self.data: numpy.array
        self.targets: list
        """
        new_data = []
        new_targets = []
        targets_np = np.array(self.targets, dtype=np.int64)
        classes = np.unique(targets_np)

        # print(len(targets_np)) 84843

        self.num_per_cls_dict = dict()
        for the_class, the_img_num in zip(classes, self.selected_num_per_cls):
            self.num_per_cls_dict[the_class] = the_img_num
            idx = np.where(targets_np == the_class)[0]
            # print(the_class, the_img_num, idx)
            np.random.shuffle(idx)
            selec_idx = idx[:the_img_num]
            new_data.append(self.data[selec_idx, ...])
            new_targets.extend([the_class, ] * the_img_num)
        self.data = torch.cat(new_data, 0)        # n*32*32*3
        self.targets = torch.tensor(new_targets)   # list of length n
            

    def __getitem__(self, index):
        
        img, label = self.data[index], self.labels[index]
        # ensure consistency with all other datasets to return a PIL Image
        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            label = self.target_transform(label)

        return img, label, index


    def __len__(self):
        return len(self.labels)


    def get_num_classes(self):
        return self.cls_num


    def get_annotations(self):
        annos = []
        for label in self.labels:
            annos.append({'category_id': int(label)})
        return annos
