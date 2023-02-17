from urllib.parse import unquote_plus
import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb
import math
import torch
import torch.nn as nn
import numpy as np

class CenterLossNet(nn.Module):
    def __init__(self, cls_num=10, feature_dim=512, margin=1, scale=None):
        super(CenterLossNet, self).__init__()
        self.classes = cls_num
        self.feature_dim = feature_dim
        self.centers = nn.Parameter(torch.randn(cls_num, self.feature_dim))
        nn.init.kaiming_normal_(self.centers, mode='fan_out')
        self.s = scale
        self.m = margin / cls_num 
        self.eps = 1e-7     

    def forward(self, features, labels, old_classes=None, old_features=None, reduction='mean'):
        # 特征向量归一化
        _features = F.normalize(features, dim=1)
        _centers = F.normalize(self.centers, dim=1)

        one_hot = F.one_hot(labels, self.classes)
        logits = torch.mm(_features, _centers.t())
        pos_logits = logits.gather(1,labels.unsqueeze(1))
        pos_theta = torch.acos(torch.clamp(pos_logits, -1.+self.eps, 1-self.eps))
        neg_theta = torch.acos(torch.clamp(logits, -1.+self.eps, 1-self.eps)) 
        variance = 0
        if old_classes > 0:
            mask = labels < old_classes
            pos_var = torch.var(pos_theta[mask]).item()
            neg_var = torch.var(pos_theta[~mask]).item()
            variance = max(self.eps, neg_var-pos_var)
            pos_theta[mask] += torch.normal(mean=0, std=variance, size=pos_theta[mask].size(), device=labels.device)
        pos_theta.clamp_(min=0, max=np.pi)
        neg_theta.clamp_(min=0, max=np.pi)
        numerator = torch.exp(self.s * (torch.cos(pos_theta)-self.m))
        denominator = numerator + torch.sum(torch.exp(torch.cos(neg_theta) * self.s) * (1-one_hot), dim=1, keepdim=True)
        L = torch.log(torch.div(numerator, denominator))
        loss = -torch.mean(L)

        if reduction == 'sum':  # 返回loss的和
            centers_batch = self.centers.index_select(dim=0, index=labels.long())
            centerloss = torch.sum(torch.pow(_features - centers_batch, 2)) / 2
            return centerloss
        elif reduction == 'mean':  # 返回loss和的平均值，默认为mean方式
            return loss, variance#dis_centers_features#
        else:
            raise ValueError("ValueError: {0} is not a valid value for reduction".format(reduction))
