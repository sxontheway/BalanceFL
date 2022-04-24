
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import Counter


def calculate_prior(num_classes, img_max=None, prior=None, prior_txt=None, reverse=False):
    if prior_txt:
        labels = []
        with open(prior_txt) as f:
            for line in f:
                labels.append(int(line.split()[1]))
        occur_dict = dict(Counter(labels))
        img_num_per_cls = [occur_dict[i] for i in range(num_classes)]
    else:
        img_num_per_cls = []
        for cls_idx in range(num_classes):
            if reverse:
                num = img_max * (prior ** ((num_classes - 1 - cls_idx) / (num_classes - 1.0)))
            else:
                num = img_max * (prior ** (cls_idx / (num_classes - 1.0)))
            img_num_per_cls.append(int(num))
    return img_num_per_cls


# Balanced CE+Softmax
class PriorCELoss(nn.Module):
    # Also named as LADE-CE Loss
    def __init__(self, num_classes, img_num_per_cls=None, img_max=None, prior=None, prior_txt=None):
        super().__init__()
        # change for different datasets
        if img_num_per_cls is None:
            img_num_per_cls = calculate_prior(num_classes, 5000, 0.01, prior_txt)

        img_num_per_cls = torch.Tensor(img_num_per_cls).float().cuda()
        print(img_num_per_cls)
        self.prior = img_num_per_cls / img_num_per_cls.sum()
        self.criterion = nn.CrossEntropyLoss()
        self.num_classes = num_classes

    def forward(self, x, y):
        logits = x + torch.log(self.prior + 1e-9)
        loss = self.criterion(logits, y)
        return loss


def create_loss(num_cls, img_max=None, prior=None, prior_txt=None, device=None):
    print('Loading PriorCELoss Loss.')
    return PriorCELoss(
        num_classes=num_cls,
        img_max=img_max,
        prior=prior,
        prior_txt=prior_txt,
    )