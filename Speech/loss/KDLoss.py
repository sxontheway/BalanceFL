import torch
import torch.nn as nn
import torch.nn.functional as F


class DistillKL(nn.Module):
    """KL divergence for distillation"""
    def __init__(self, T):
        super(DistillKL, self).__init__()
        self.T = T

    def forward(self, y_s, y_t):
        p_s = F.log_softmax(y_s/self.T, dim=1)
        p_t = F.softmax(y_t/self.T, dim=1)
        loss = F.kl_div(p_s, p_t, size_average=False) * (self.T**2) / y_s.shape[0]
        return loss


class Contras(nn.Module):
    """Contrastive Loss"""
    def __init__(self, T):
        super(DistillKL, self).__init__()
        self.T = T
        self.cos = torch.nn.CosineSimilarity(dim=-1)
        self.criterion = torch.nn.CrossEntropyLoss

    def forward(self, y_s, y_t, y_p):

        posi = self.cos(y_s, y_t)
        logits = posi.reshape(-1,1)

        nega = self.cos(y_s, y_p)
        logits = torch.cat((logits, nega.reshape(-1,1)), dim=1)/self.T

        labels = torch.zeros(y_s.size(0)).cuda().long()
        loss = self.criterion(logits, labels)
        return loss


class Attention(nn.Module):
    """attention transfer loss"""
    def __init__(self, p=2):
        super(Attention, self).__init__()
        self.p = p

    def forward(self, g_s, g_t):
        return [self.at_loss(f_s, f_t) for f_s, f_t in zip(g_s, g_t)]

    def at_loss(self, f_s, f_t):
        s_H, t_H = f_s.shape[2], f_t.shape[2]
        if s_H > t_H:
            f_s = F.adaptive_avg_pool2d(f_s, (t_H, t_H))
        elif s_H < t_H:
            f_t = F.adaptive_avg_pool2d(f_t, (s_H, s_H))
        else:
            pass
        return (self.at(f_s) - self.at(f_t)).pow(2).mean()

    def at(self, f):
        return F.normalize(f.pow(self.p).mean(1).view(f.size(0), -1))


class HintLoss(nn.Module):
    """regression loss from hints"""
    def __init__(self):
        super(HintLoss, self).__init__()
        self.crit = nn.MSELoss()

    def forward(self, f_s, f_t):
        loss = self.crit(f_s, f_t)
        return loss


class Hinge_loss(nn.Module):
    """
    Squared hinge loss.  
    Input: (feature embedding, labels, classfier_weight), 
    the shapes are (batch_size*feat_dim, batch_size, num_of_classes*feat_dim)
    """
    def __init__(self):
        super(Hinge_loss, self).__init__()
        self.crit = nn.MSELoss()
        self.margin = 0.9

    def forward(self, embedding, labels, weight):
        dist = torch.sum(embedding*weight[labels], 1) 
        loss = torch.square(torch.clamp(self.margin-dist, min=0)).sum()
        return loss


class CEKD_Loss(nn.Module):
    # knowledge distlillation loss + CE Loss (or hinge loss)
    def __init__(self, Temp, lamda, loss_cls, loss_kd):
        super().__init__()
        self.lamda = lamda
        self.loss_cls = loss_cls
        self.loss_kd = loss_kd
        
        if loss_cls == "ce":
            self.criterion_cls = nn.CrossEntropyLoss()
        elif loss_cls == "hinge":
            self.criterion_cls = Hinge_loss()
        elif loss_cls == "hinge_multi":
            raise NotImplementedError()

        if loss_kd == "hint":
            self.criterion_kd = nn.MSELoss()
        elif loss_kd == "kl":
            self.criterion_kd = DistillKL(Temp)
        elif loss_kd == "con":
            self.criterion_kd = Contras(Temp)
        else:
            raise NotImplementedError(loss_cls)

    def forward(self, logits, labels, feat=None, feat_teacher=None, classfier_weight=None):
        if self.loss_cls == "ce":
            loss_cls = self.criterion_cls(logits, labels)
        elif self.loss_cls == "hinge":
            loss_cls = self.criterion_cls(feat, labels, classfier_weight)

        if feat_teacher is not None and self.lamda != 0:
            if self.loss_kd in ["hint", "kl"]:
                loss_kd = self.criterion_kd(feat, feat_teacher)
            elif self.loss_kd in ["con"]:
                raise RuntimeError
                # loss_kd = self.criterion_kd(feat, feat_teacher, prev_feature)
            loss = loss_cls + self.lamda*loss_kd
        else:   
            loss = loss_cls
            loss_kd = torch.tensor(0)

        return loss, loss_cls, loss_kd




def create_loss(Temp, lamda, loss_cls, loss_kd, device):
    print('Loading CE+KD Loss.')
    return CEKD_Loss(
        Temp = Temp,    # e.g., 4
        lamda = lamda,  # e.g., 100 
        loss_cls = loss_cls,    # e.g., "ce", "hinge"
        loss_kd = loss_kd    # e.g., "hint", "kl"
    )
