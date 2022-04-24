import torch
import os
import numpy as np
from data import dataloader
from utils.misc import check_nan


def shot_acc(loss_per_cls, acc_per_cls, shot_num_per_cls, many_shot_thr=100, low_shot_thr=20):  
    """
    Return: [acc_many_shot, acc_mid_shot, acc_low_shot], [loss_many_shot, loss_mid_shot, loss_low_shot]
    """
    acc, loss = [], []  
    shot_group = np.array(list(
        map(lambda x: 0 if x>=many_shot_thr else (2 if x<low_shot_thr else 1), shot_num_per_cls)))
    for i in range(3):  # 0, 1, 2: many_shot, mid_shot, low_shot
        acc.append(np.array(acc_per_cls)[shot_group==i].mean())
        loss.append(np.array(loss_per_cls)[shot_group==i].mean())
    return loss, acc
        

def acc_loss_per_cls(loss_all, correct_all, labels_all, cls_num):
    """
    calculate per class accuracy and loss
    ---
    - input:
        loss, correct (True or False), labeled class for all testing samples. All in list.
        cls_num: number of class. Scalar.
    - output:
        Per-class accuracy and loss. All in list.
    """
    acc_record_per_cls = [[] for i in range(cls_num)]
    loss_record_per_cls = [[] for i in range(cls_num)]
    acc_per_cls, loss_per_cls = [], []

    for loss, correct, label in zip(loss_all, correct_all, labels_all):
        acc_record_per_cls[label].append(correct.item())
        loss_record_per_cls[label].append(loss.item())

    for cls_i in range(cls_num):
        acc_per_cls.append(np.array(acc_record_per_cls[cls_i]).mean())
        loss_per_cls.append(np.array(loss_record_per_cls[cls_i]).mean())

    return loss_per_cls, acc_per_cls


def validate_one_model(model, dataset, device, per_cls_acc=False):
    """
    Validate the accuracy on the entire dataset for on model.
    If per_cls_acc is True, return per-class accuracy/loss (arrays).
    Else, return the mean accuracy/loss of all classes (scalars). 
    """
    # !!!!!!!! can only using 0 under multi-process mode, or the program will stuck 
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size = 128, shuffle = True, num_workers = 0   
        )  
    for key in model.keys():
        model[key].eval()

    if per_cls_acc is True:

        loss_all, correct_all, labels_all = [], [], []
        tmp_criterion = torch.nn.CrossEntropyLoss(reduction="none")

        cls_num = len(np.unique(dataset.label))

        with torch.no_grad():
            for (imgs, labels, indexs) in dataloader:

                imgs = imgs.to(device)
                labels = labels.to(device)
                
                output = model['classifier'](model['feat_model'](imgs))
                prediction = output.argmax(dim=1, keepdim=True)
                
                labels_all.extend(labels)
                loss_all.extend(tmp_criterion(output, labels))
                correct_all.extend(prediction.eq(labels.view_as(prediction)).squeeze(1))

            loss_per_cls, acc_per_cls = acc_loss_per_cls(loss_all, correct_all, labels_all, cls_num)

        return (np.array(loss_per_cls), np.array(acc_per_cls))
    
    else:
        loss = 0.0
        correct = 0

        tmp_criterion = torch.nn.CrossEntropyLoss()
        with torch.no_grad():
            for (imgs, labels, indexs) in dataloader:
                imgs = imgs.to(device)
                labels = labels.to(device)
                output = model['classifier'](model['feat_model'](imgs))
                loss += tmp_criterion(output, labels).item()
                prediction = output.argmax(dim=1, keepdim=True)
                correct += prediction.eq(labels.view_as(prediction)).sum().item()

        loss /= len(dataloader)
        acc = correct/len(dataloader.dataset)
        return (loss, acc)


def save_checkpoint(state, is_best, filename):
    """save checkpoint"""
    if is_best:
        torch.save(state, filename)
    else:
        torch.save(state, "checkpoints/pretrained_last.pth")


def load_checkpoint(model, ckpt_pretrained):
    assert os.path.isfile(ckpt_pretrained), 'Error: no model checkpoint directory found!'
    checkpoint = torch.load(ckpt_pretrained)
    model.load_state_dict(checkpoint['state_dict'])
    print("loaded checkpoint, epoch: ", checkpoint["epoch"])
    return model
