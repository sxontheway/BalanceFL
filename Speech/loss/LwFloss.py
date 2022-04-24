import torch
import torch.nn as nn
import torch.nn.functional as F


def linear_combination(x, y, epsilon):
    return epsilon * x + (1 - epsilon) * y

def reduce_loss(loss, reduction='mean'):
    return loss.mean() if reduction == 'mean' else loss.sum() if reduction == 'sum' else loss

class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, epsilon=0.01, reduction='mean'):
        super().__init__()
        self.epsilon = epsilon
        self.reduction = reduction

    def forward(self, preds, target):
        n = preds.size()[-1]
        log_preds = F.log_softmax(preds, dim=-1)
        loss = reduce_loss(-log_preds.sum(dim=-1), self.reduction)
        nll = F.nll_loss(log_preds, target, reduction=self.reduction)
        return linear_combination(loss / n, nll, self.epsilon)


class DistillKL(nn.Module):
    """KL divergence for distillation"""
    def __init__(self, Temp, mode):
        super(DistillKL, self).__init__()
        self.T = Temp
        self.mode = mode

    def forward(self, y_s, y_t):
        outputs = torch.log_softmax(y_s/self.T, dim=1)
        labels = torch.softmax(y_t/self.T, dim=1)

        if self.mode == "kl":
            # implemention 1
            loss = F.kl_div(outputs, labels)
        elif self.mode == "ce":
            # implemention 2
            outputs = torch.sum(outputs * labels, dim=1, keepdim=False)
            loss = -torch.mean(outputs, dim=0, keepdim=False)
        else:
            raise NotImplementedError()
        return loss


class LwF_Loss(nn.Module):
    # knowledge distlillation loss + CE Loss (or hinge loss)
    def __init__(self, Temp, lamda, loss_cls, loss_kd, num_cls, device):
        super().__init__()
        self.lamda = lamda
        self.loss_cls = loss_cls
        self.loss_kd = loss_kd
        self.num_cls = num_cls
        self.device = device
        
        if loss_cls == "ce":
            self.criterion_cls = nn.CrossEntropyLoss()
        elif loss_cls == "sce": # smooth ce
            self.criterion_cls = LabelSmoothingCrossEntropy()
        else:
            raise NotImplementedError()

        self.criterion_kd = DistillKL(Temp, self.loss_kd)

    def forward(self, labels, teacher_pred, logits, logit_aug=None):
        if logit_aug is None:
            logit_aug = logits
        pos_cls = torch.unique(labels).tolist()
        neg_cls = list(set([*range(self.num_cls)]).difference(set(pos_cls)))
        transformed_labels = torch.tensor([pos_cls.index(i) for i in labels]).to(logits.device)
        # print(logit_aug[:, pos_cls].shape, transformed_labels.shape, labels.shape)
        loss_cls = self.criterion_cls(logit_aug[:, pos_cls], transformed_labels)
        loss_kd = self.criterion_kd(logits[:, neg_cls], teacher_pred[:, neg_cls])

        # self entropy penality
        # preds = torch.softmax(logits, dim=1)
        # logs_preds = torch.log_softmax(logits, dim=1)
        preds = torch.softmax(logits[:, pos_cls], dim=1)
        logs_preds = torch.log_softmax(logits[:, pos_cls], dim=1)
        loss_ent = torch.sum(-preds*logs_preds)
        # print(loss_cls, loss_kd, loss_ent)

        if torch.isnan(loss_cls):
            print(pos_cls)
            print(logits[:, pos_cls].shape, labels.shape, max(transformed_labels))
            print(loss_cls)

        # select according to classes
        loss = loss_cls + self.lamda*loss_kd - 0.002*loss_ent
        return loss, loss_cls, loss_kd



def create_loss(Temp, lamda, loss_cls, loss_kd, num_classes, device):
    print('Loading LwF_Loss (CE+KD).')
    return LwF_Loss(
        Temp = Temp,    # e.g., 2
        lamda = lamda,  # e.g., 10 
        loss_cls = loss_cls,    # e.g., "ce"
        loss_kd = loss_kd,    # e.g., "ce", "kl"
        num_cls = num_classes, 
        device = device
    )
