from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import copy
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

from lib.models.losses import FocalLoss, LBHingev2, RegL1Loss, RegLoss, NormRegL1Loss, RegWeightedL1Loss
from lib.utils.decode import ctdet_decode
from lib.utils.utils import _sigmoid, _transpose_and_gather_feat
from lib.utils.debugger import Debugger
from lib.utils.post_process import ctdet_post_process
from lib.Trainer.base_trainer import BaseTrainer
import cv2


class CtdetLoss(torch.nn.Module):
    """
    CenterNet
    
    Args:
        opt: 配置参数，包含各任务损失权重等
    """
    def __init__(self, opt):
        super(CtdetLoss, self).__init__()
        self.opt = opt
        
        # 损失函数
        self.crit = FocalLoss() # 关键点热力图损失
        self.crit_reg = RegL1Loss()  # 回归损失（L1损失）
        self.crit_wh = nn.L1Loss(reduction='sum') # 边界框尺寸损失
        self.wh_weight = 0.1  # 边界框损失权重
        self.hm_weight = 1  # 热力图损失权重
        self.off_weight = 1  # 偏移量损失权重
        self.track_weight = 1  # 跟踪损失权重
        self.seq_weight = 1  # 时序损失权
         
        # 模型配置
        self.ratios = [1]  # 特征图缩放比例
        self.num_stacks = 1  # 网络堆叠数量

    def forward(self, outputs, batches):
        """
        Args:
            model_outputs: 模型输出字典，包含各任务预测结果
            batch_data: 批次数据字典，包含真实标签
            
        Returns:
            total_loss: 总损失
            loss_dict: 各任务损失字典
        """
        hm_loss = torch.tensor(0., device=self.opt.device)
        wh_loss = torch.tensor(0., device=self.opt.device)
        off_loss = torch.tensor(0., device=self.opt.device)
        track_loss = torch.tensor(0., device=self.opt.device)
        seq_loss = torch.tensor(0., device=self.opt.device)        
        
        # 遍历不同缩放比例的输出
        for ratio in self.ratios:
            # 获取当前比例的模型输出和批次数据
            output = outputs[-1][ratio]
            batch = batches[ratio]

            output['hm'] = _sigmoid(output['hm'])
            output['hm_seq'] = _sigmoid(output['hm_seq'])

            hm_loss += self.crit(output['hm'], batch['hm'][:, -1])

            wh_loss += self.crit_reg(
                output['wh'], batch['reg_mask'],
                batch['ind'], batch['wh'])  / self.num_stacks

            off_loss += self.crit_reg(output['reg'], batch['reg_mask'],
                                    batch['ind'], batch['reg'])  / self.num_stacks
            
            if self.opt.seqLen > 1:
                track_loss += sum(
                    self.crit_reg(
                        output['dis'][:, i], batch['dis_mask'][:, i],
                        batch['dis_ind'][:, i], batch['dis'][:, i]
                    ) for i in range(self.opt.seqLen - 1)
                ) / (self.opt.seqLen - 1)

            # 时序热力图损失
            seq_loss += sum(
                self.crit(output['hm_seq'][:, i], batch['hm_seq'][:, i])
                for i in range(self.opt.seqLen)
            ) / self.opt.seqLen


        loss = self.hm_weight * hm_loss + self.wh_weight * wh_loss + \
               self.off_weight * off_loss + self.track_weight * track_loss + self.seq_weight * seq_loss

        loss_stats = {'loss': loss, 
                      'hm_loss': hm_loss,
                      'wh_loss': wh_loss, 
                      'off_loss': off_loss, 
                      'track_loss': track_loss, 
                      'seq_loss': seq_loss}

        return loss, loss_stats


class CtdetTrainer(BaseTrainer):
    """
    CenterNet 训练器，处理模型训练、调试和结果保存
    
    Args:
        opt: 配置参数
        model: 模型实例
        optimizer: 优化器实例
    """
    def __init__(self, opt, model, optimizer=None):
        super(CtdetTrainer, self).__init__(opt, model, optimizer=optimizer)

    def _get_losses(self, opt):
        loss_states = ['loss', 'hm_loss', 'wh_loss', 'off_loss', 'track_loss', 'seq_loss']
        loss = CtdetLoss(opt)
        return loss_states, loss

    
    def save_result(self, output, batch, results):

        # 条件获取偏移量回归预测
        reg = output['reg'] if self.opt.reg_offset else None

        # 直接解码并转换为numpy
        dets = ctdet_decode(
            output['hm'], output['wh'], reg=reg,
            cat_spec_wh=self.opt.cat_spec_wh, K=self.opt.K
        ).detach().cpu().numpy().reshape(1, -1, -1)

        # 后处理操作
        h, w, c = output['hm'].shape[2], output['hm'].shape[3], output['hm'].shape[1]
        dets_out = ctdet_post_process(
            dets.copy(),
            batch['meta']['c'].cpu().numpy(),
            batch['meta']['s'].cpu().numpy(),
            h, w, c
        )

        # 提取图像ID并保存结果
        img_id = batch['meta']['img_id'].cpu().numpy()[0]
        results[img_id] = dets_out[0]
    
