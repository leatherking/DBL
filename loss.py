from urllib.parse import unquote_plus
import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb
import math
import torch
import torch.nn as nn

class CenterLossNet(nn.Module):
    def __init__(self, cls_num=10, feature_dim=512, proj=False):
        super(CenterLossNet, self).__init__()
        self.classes = cls_num
        self.feature_dim = feature_dim
        self.centers = nn.Parameter(torch.randn(cls_num, self.feature_dim))
        nn.init.kaiming_normal_(self.centers, mode='fan_out')
        self.s = 16


    def forward(self, features, labels, sample_per_class=None, reduction='mean'):
        # 特征向量归一化
        _features = F.normalize(features, dim=1)
        _centers = F.normalize(self.centers, dim=1)

        # top = torch.mm(_features, _centers.t()) #/ 2
        # logits = F.softmax(top*self.s) #
        # p = logits.gather(1, labels.unsqueeze(1))
        # focal = -(1-torch.pow(p,2)) * torch.log(p)
        # loss = focal.mean()
        
        # mu = torch.mean(features, dim=1)
        # var = torch.var(features, dim=1)
        # features = (features - mu.unsqueeze(1)) / torch.sqrt(var.unsqueeze(1))
        # features = torch.mul(self.scale, features) + self.bias

        # _features = F.normalize(features, dim=1)
        # _centers = self.centers#F.normalize(self.centers, dim=1)
        # '''infoNCE'''
        # top = torch.mm(_features, _centers.t())
        # info_NCE = F.log_softmax(top)
        # loss = F.nll_loss(info_NCE, labels)

        # 加入负样本
        all_centers = _centers.unsqueeze(0).repeat(self.classes, 1, 1)
        mask = torch.ones((self.classes,self.classes), device=_centers.device) - torch.eye(self.classes, device=_centers.device)
        mask = mask.unsqueeze(2).repeat(1,1,self.feature_dim)
        neg_centers = all_centers * mask
        neg_syn_centers = torch.sum(neg_centers, dim=1)
        neg_syn_norm = neg_syn_centers / torch.norm(neg_syn_centers, p=2, dim=-1).unsqueeze(1)
        # 69.3
        pos_centers = _centers - neg_syn_norm
        pos_centers = pos_centers / torch.norm(pos_centers, p=2, dim=-1).unsqueeze(1)
        theta = F.sigmoid(torch.mm(_features, pos_centers.t()))
        logits = theta.gather(1, labels.unsqueeze(1))
        entropy = - torch.log(logits)
        loss = entropy.mean()

        if reduction == 'sum':  # 返回loss的和
            centers_batch = self.centers.index_select(dim=0, index=labels.long())
            centerloss = torch.sum(torch.pow(_features - centers_batch, 2)) / 2
            return centerloss
        elif reduction == 'mean':  # 返回loss和的平均值，默认为mean方式
            return loss
        else:
            raise ValueError("ValueError: {0} is not a valid value for reduction".format(reduction))