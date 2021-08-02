import torch
import torch.nn as nn
from ..builder import LOSSES
from functools import partial
import torch.nn.functional as F
import torch.distributed as dist
from mmdet.utils import get_root_logger



@LOSSES.register_module()
class EQLv2(nn.Module):
    def __init__(self, use_sigmoid=True, reduction='mean', class_weight=None, loss_weight=1.0,
                 num_classes=515, gamma=12, mu=0.8, alpha=4.0, vis_grad=False):
        super().__init__()
        self.use_sigmoid = True              # 是否使用sigmoid
        self.reduction = reduction           # 损失方式
        self.loss_weight = loss_weight       # 损失权重
        self.class_weight = class_weight     # 类别权重，默认为None，不使用
        self.num_classes = num_classes       # 类别数（需要修改）
        self.group = True
        # cfg for eqlv2
        self.vis_grad = vis_grad   # 没用
        self.gamma = gamma         # 标准化使用（拉伸）
        self.mu = mu               # 标准化使用（拉伸）
        self.alpha = alpha         # 正负样本平衡使用
        # initial variables
        self._pos_grad = None      # 前景梯度初始化
        self._neg_grad = None      # 背景梯度初始化
        self.pos_neg = None        # 正负梯度累积比

        # 标准化函数
        def _func(x, gamma, mu):
            return 1 / (1 + torch.exp(-gamma * (x - mu)))
        self.map_func = partial(_func, gamma=self.gamma, mu=self.mu)    # 标准化函数进行初始化
        logger = get_root_logger()                                      # 日志
        logger.info(f"build EQL v2, gamma: {gamma}, mu: {mu}, alpha: {alpha}")  # 日志

    def forward(self, cls_score, label, weight=None, avg_factor=None, reduction_override=None, **kwargs):
        self.n_i, self.n_c = cls_score.size()   # 输出的尺寸 n_i 图片张数，n_c 类别数
        self.gt_classes = label                 # 真实的label
        self.pred_class_logits = cls_score      # logits

        # 获取真实类别的one-hot表示
        def expand_label(pred, gt_classes):
            target = pred.new_zeros(self.n_i, self.n_c)      # 创建与预测相同的尺寸
            target[torch.arange(self.n_i), gt_classes] = 1   # one-hot
            return target

        target = expand_label(cls_score, label)              # groundtruth的one-hot表示
        pos_w, neg_w = self.get_weight(cls_score)            # 获取正负累积梯度的权重
        weight = pos_w * target + neg_w * (1 - target)       # 更新每个样本的权重
        cls_loss = F.binary_cross_entropy_with_logits(cls_score, target, reduction='none')   # 交叉熵
        cls_loss = torch.sum(cls_loss * weight) / self.n_i   # 每个类别进行权重矫正
        self.collect_grad(cls_score.detach(), target.detach(), weight.detach())              # 更新新的正负梯度累积比
        return self.loss_weight * cls_loss                   # 返回loss

    # 得到通道数（没用）
    def get_channel_num(self, num_classes):
        num_channel = num_classes + 1
        return num_channel

    # 获取类别激活图（没用）
    def get_activation(self, cls_score):
        cls_score = torch.sigmoid(cls_score)
        n_i, n_c = cls_score.size()
        bg_score = cls_score[:, -1].view(n_i, 1)
        cls_score[:, :-1] *= (1 - bg_score)
        return cls_score

    # 收集梯度信息，更新正负样本的梯度信息
    def collect_grad(self, cls_score, target, weight):
        prob = torch.sigmoid(cls_score)                                   # 概率值
        grad = target * (prob - 1) + (1 - target) * prob                  # 梯度越小越好
        grad = torch.abs(grad)                                            # 梯度绝对值
        # do not collect grad for objectiveness branch [:-1]
        pos_grad = torch.sum(grad * target * weight, dim=0)[:-1]          # 正样本梯度的累积权重矫正
        neg_grad = torch.sum(grad * (1 - target) * weight, dim=0)[:-1]    # 负样本梯度的累积权重矫正
        dist.all_reduce(pos_grad)
        dist.all_reduce(neg_grad)
        self._pos_grad += pos_grad   # 累积正样本梯度信息
        self._neg_grad += neg_grad   # 累积负样本梯度信息
        self.pos_neg = self._pos_grad / (self._neg_grad + 1e-10)          # 当前迭代次数下的正负梯度信息比

    # 获取正负样本的梯度比例
    def get_weight(self, cls_score):
        # 在初始阶段，没有正负梯度信息比
        if self._pos_grad is None:
            self._pos_grad = cls_score.new_zeros(self.num_classes)   # 默认累积梯度，每个类别都是1.0
            self._neg_grad = cls_score.new_zeros(self.num_classes)   # 默认累积梯度，每个类别都是1.0
            neg_w = cls_score.new_ones((self.n_i, self.n_c))         # 默认累积梯度，每个类别都是1.0
            pos_w = cls_score.new_ones((self.n_i, self.n_c))         # 默认累积梯度，每个类别都是1.0
        else:
            # 存在正负梯度比，则计算正样本梯度权重和负样本梯度权重
            neg_w = torch.cat([self.map_func(self.pos_neg), cls_score.new_ones(1)])   # f(x)函数，负样本梯度权重
            pos_w = 1 + self.alpha * (1 - neg_w)                                      # 正样本梯度权重
            neg_w = neg_w.view(1, -1).expand(self.n_i, self.n_c)                      # 尺度变化
            pos_w = pos_w.view(1, -1).expand(self.n_i, self.n_c)                      # 尺度变化
        return pos_w, neg_w