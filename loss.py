from urllib.parse import unquote_plus
import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb
import math
import torch
import torch.nn as nn
import numpy as np

def _KD_loss(pred, soft, T):
    pred = torch.log_softmax(pred / T, dim=1)
    soft = torch.softmax(soft / T, dim=1)
    return -1 * torch.mul(soft, pred).sum() / pred.shape[0]

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
        self.bias = nn.Parameter(torch.zeros(cls_num,1))

    def _forward(self, features, labels, old_classes=None):
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
                        torch.exp(self.s * torch.cos(pos_theta + self.m + 0.1)) * mask * sample_per_class
            
            neg_theta = torch.exp(self.s * (torch.cos(theta) + self.m2)) * (1-one_hot)
            neg_theta[:,:old_classes] *= sample_per_class
            neg_theta[:,old_classes:] *= 500
            denominator = torch.sum(neg_theta, dim=1, keepdim=True) + numerator
        logits = -torch.log(torch.div(numerator, denominator))
        return torch.mean(logits)

        

    def forward(self, features, labels, old_classes=None, old_centers=None, old_features=None, reduction='mean'):
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
        # all_centers = _centers.unsqueeze(0).repeat(self.classes, 1, 1)
        # mask = torch.ones((self.classes,self.classes), device=_centers.device) - torch.eye(self.classes, device=_centers.device)
        # mask = mask.unsqueeze(2).repeat(1,1,self.feature_dim)
        # neg_centers = all_centers * mask
        # neg_syn_centers = torch.sum(neg_centers, dim=1)
        # neg_syn_norm = neg_syn_centers / (len(_centers)-1)
        # neg_syn_norm = neg_syn_norm / torch.norm(neg_syn_norm, p=2, dim=-1).unsqueeze(1)
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

        '''pos_logits = torch.mm(_features, _centers.t()).gather(1,labels.unsqueeze(1))
        neg_logits = torch.mm(_features, neg_syn_norm.t()).gather(1,labels.unsqueeze(1))
        top = self.s * torch.cos(torch.acos(torch.clamp(pos_logits, -1.+self.eps, 1-self.eps)) + self.m)
        denominator = torch.exp(top) + torch.exp(neg_logits * self.s)
        L = top -torch.log(denominator)
        loss = -torch.mean(L)'''

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

        # 改为高斯核函数
        extend_centers = _centers.unsqueeze(0).repeat(len(_features), 1, 1)
        extend_features = _features.unsqueeze(1).repeat(1, len(_centers), 1)
        dis_centers_features = -torch.pow((extend_features - extend_centers), 2).sum(-1) *5 #(62.65%) #* 10 
        if old_classes == 0:
            variance = 0
            loss = F.cross_entropy(dis_centers_features, labels)
        else:
            '''pos_metric = dis_centers_features.gather(1, labels.unsqueeze(1))
            mask = labels < old_classes
            one_hot = F.one_hot(labels, self.classes)
            pos_metric[mask] += torch.from_numpy(np.array(200)).log()
            # pdb.set_trace()
            numerator = torch.exp(pos_metric)
            denominator = torch.sum(torch.exp(dis_centers_features)*(1-one_hot), dim=1, keepdim=True) + numerator 
            logits = -torch.log(torch.div(numerator, denominator))
            loss = torch.mean(logits)'''

            # 68.67
            pos_metric = dis_centers_features.gather(1, labels.unsqueeze(1))
            pos_metric += self.bias[labels]
            one_hot = F.one_hot(labels, self.classes)
            numerator = torch.exp(pos_metric)
            denominator = torch.sum(torch.exp(dis_centers_features)*(1-one_hot), dim=1, keepdim=True) + numerator 
            logits = -torch.log(torch.div(numerator, denominator))
            variance = torch.var(pos_metric)

            loss = torch.mean(logits) + variance

            # dis_centers_features += self.bias
            # pos_metric = dis_centers_features.gather(1, labels.unsqueeze(1))
            # variance = torch.var(pos_metric)
            # loss = F.cross_entropy(dis_centers_features, labels) + variance

            # # KD 无效
            # _old_centers = F.normalize(old_centers, dim=1)
            # _old_features = F.normalize(old_features, dim=1)
            # extend_old_centers = _old_centers.unsqueeze(0).repeat(len(old_features), 1, 1)
            # extend_old_features = _old_features.unsqueeze(1).repeat(1, len(old_centers), 1)
            # old_dis = -torch.pow((extend_old_features - extend_old_centers), 2).sum(-1) *5
            # loss_kd = F.mse_loss(dis_centers_features[:,:len(old_centers)], old_dis) * 0.5
            # loss = F.cross_entropy(dis_centers_features, labels)

        if reduction == 'sum':  # 返回loss的和
            centers_batch = self.centers.index_select(dim=0, index=labels.long())
            centerloss = torch.sum(torch.pow(_features - centers_batch, 2)) / 2
            return centerloss
        elif reduction == 'mean':  # 返回loss和的平均值，默认为mean方式
            return loss, variance
        else:
            raise ValueError("ValueError: {0} is not a valid value for reduction".format(reduction))
