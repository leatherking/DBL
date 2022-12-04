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
    def __init__(self, cls_num=10, feature_dim=512, proj=False):
        super(CenterLossNet, self).__init__()
        self.classes = cls_num
        self.feature_dim = feature_dim
        self.centers = nn.Parameter(torch.randn(cls_num, self.feature_dim))
        nn.init.kaiming_normal_(self.centers, mode='fan_out')
        self.s = 64
        self.m = 5 / cls_num 
        self.eps = 1e-7
        self.m2 = 5 / cls_num

    def forward(self, features, labels, old_classes=None):
        _features = F.normalize(features, dim=1)
        _centers = F.normalize(self.centers, dim=1)

        theta = torch.acos(torch.clamp(torch.matmul(_features, _centers.t()),-1.+self.eps, 1-self.eps))
        one_hot = F.one_hot(labels, self.classes)
        pos_theta = theta.gather(1,labels.unsqueeze(1))
        if old_classes == 0:
            numerator = torch.exp(self.s * torch.cos(pos_theta + self.m))
            neg_theta = torch.exp(self.s * torch.cos(theta)) * (1-one_hot)
            denominator = torch.sum(neg_theta, dim=1, keepdim=True) + numerator
        else:
            mask = torch.zeros_like(labels, device=labels.device).unsqueeze(1)
            mask[labels<old_classes] = 1
            sample_per_class = 2000 / old_classes
            numerator = torch.exp(self.s * torch.cos(pos_theta + self.m)) * (1-mask) * 500 + \
                        torch.exp(self.s * torch.cos(pos_theta + self.m * (500/sample_per_class))) * mask * sample_per_class
            
            neg_theta = torch.exp(self.s * torch.cos(theta)) * (1-one_hot)
            neg_theta[:,:old_classes] *= sample_per_class
            neg_theta[:,old_classes:] *= 500
            denominator = torch.sum(neg_theta, dim=1, keepdim=True) + numerator
        logits = -torch.log(torch.div(numerator, denominator))
        return torch.mean(logits)

        

    def _forward(self, features, labels, old_classes=None, reduction='mean'):
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
        neg_syn_norm = neg_syn_centers / (len(_centers)-1)
        neg_syn_norm = neg_syn_norm / torch.norm(neg_syn_norm, p=2, dim=-1).unsqueeze(1)
        # 69.3
        # pos_centers = _centers - neg_syn_norm
        # pos_centers = pos_centers / torch.norm(pos_centers, p=2, dim=-1).unsqueeze(1)

        # # theta = F.sigmoid(torch.mm(_features, pos_centers.t()))
        # theta = torch.mm(_features, pos_centers.t())
        # logits = theta.gather(1, labels.unsqueeze(1))
        # # entropy = - torch.log(logits)
        # loss = -logits.mean()
        '''pos_features = neg_features = _features
        if old_classes != 0:
            old_mask = labels < old_classes
            neg_features = _features
            pos_features = _features
            neg_features[old_mask] = features[old_mask]
            pos_features[~old_mask] = features[~old_mask]

        pos_logits = torch.mm(pos_features, _centers.t()).gather(1,labels.unsqueeze(1))
        neg_logits = torch.mm(neg_features, neg_syn_norm.t()).gather(1,labels.unsqueeze(1))
        denominator = torch.exp(pos_logits) + torch.exp(neg_logits)
        L = pos_logits - torch.log(denominator)
        loss = -torch.mean(L)'''

        # pos_logits = torch.mm(_features, _centers.t()).gather(1,labels.unsqueeze(1))
        # neg_logits = torch.mm(_features, neg_syn_norm.t()).gather(1,labels.unsqueeze(1))
        # denominator = torch.exp(pos_logits) + torch.exp(neg_logits)
        # L = pos_logits - torch.log(denominator)
        # loss = -torch.mean(L)

        pos_logits = torch.mm(_features, _centers.t()).gather(1,labels.unsqueeze(1))
        neg_logits = torch.mm(_features, neg_syn_norm.t()).gather(1,labels.unsqueeze(1))
        top = self.s * torch.cos(torch.acos(torch.clamp(pos_logits, -1.+self.eps, 1-self.eps)) + self.m)
        denominator = torch.exp(top) + torch.exp(neg_logits * self.s)
        L = top -torch.log(denominator)
        loss = -torch.mean(L)


        # pos_logits = torch.mm(_features, _centers.t()).gather(1,labels.unsqueeze(1))
        # neg_logits = torch.mm(_features, neg_syn_norm.t()).gather(1,labels.unsqueeze(1))
        # top = self.s * torch.cos(torch.acos(torch.clamp(pos_logits, -1.+self.eps, 1-self.eps)) + self.m)
        # if old_classes > 0:
        #     sample_per_class = 2000 / old_classes
        #     # neg_logits += torch.log(torch.from_numpy(np.array(500)))
        #     top += torch.log(torch.from_numpy(np.array(sample_per_class)))
        #     denominator = torch.exp(top) + torch.exp(neg_logits * self.s + torch.log(torch.from_numpy((self.classes - old_classes)* np.array(500))))
        # else:
        #     denominator = torch.exp(top) + torch.exp(neg_logits * self.s)
        # L = top -torch.log(denominator)
        # loss = -torch.mean(L)

        # # 改为高斯核函数
        # extend_centers = _centers.unsqueeze(0).repeat(len(_features), 1, 1)
        # extend_features = _features.unsqueeze(1).repeat(1, len(_centers), 1)
        # dis_centers_features = -torch.pow((extend_features - extend_centers)*0.9, 2).sum(-1) *5 (62.65%) #* 10 
        # loss = F.cross_entropy(dis_centers_features, labels)

        # angles = torch.acos(torch.mm(_features, _centers.t()))
        # theta = 2*math.pi / self.classes
        # logits = torch.cos(angles - theta)
        # loss = F.cross_entropy(logits, labels)

        if reduction == 'sum':  # 返回loss的和
            centers_batch = self.centers.index_select(dim=0, index=labels.long())
            centerloss = torch.sum(torch.pow(_features - centers_batch, 2)) / 2
            return centerloss
        elif reduction == 'mean':  # 返回loss和的平均值，默认为mean方式
            return loss
        else:
            raise ValueError("ValueError: {0} is not a valid value for reduction".format(reduction))
