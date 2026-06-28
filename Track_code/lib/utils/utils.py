from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn

def _sigmoid(x):
    y = torch.clamp(x.sigmoid_(), min=1e-4, max=1-1e-4)
    return y

def _gather_feat(feat, ind, mask=None):
    # 获取特征维度
    batch_size, num_indices, channel_dim = feat.shape[0], ind.shape[1], feat.shape[2]
    
    # 将索引扩展为与特征维度匹配
    # [B, K] -> [B, K, C]
    expanded_ind = ind.unsqueeze(2).expand(batch_size, num_indices, channel_dim)
    
    # 根据索引收集特征
    gathered = feat.gather(1, expanded_ind)
    
    # 应用掩码
    if mask is not None:
        # 扩展掩码以匹配收集的特征形状
        expanded_mask = mask.unsqueeze(2).expand_as(gathered)
        # 应用掩码并重塑为 [B*K, C]
        return gathered[expanded_mask].view(-1, channel_dim)
    
    return gathered

def _transpose_and_gather_feat(feat, ind):
    # 1. 转置特征图维度：[B, C, H, W] -> [B, H, W, C]
    feat = feat.permute(0, 2, 3, 1).contiguous()
    
    # 2. 重塑为二维网格表示：[B, H, W, C] -> [B, H*W, C]
    batch_size, height, width, channels = feat.shape
    feat = feat.view(batch_size, height * width, channels)
    
    # 3. 根据索引收集特征：[B, H*W, C] -> [B, K, C]
    return _gather_feat(feat, ind)

def flip_tensor(x):
    return torch.flip(x, [3])
    # tmp = x.detach().cpu().numpy()[..., ::-1].copy()
    # return torch.from_numpy(tmp).to(x.device)

def flip_lr(x, flip_idx):
  tmp = x.detach().cpu().numpy()[..., ::-1].copy()
  shape = tmp.shape
  for e in flip_idx:
    tmp[:, e[0], ...], tmp[:, e[1], ...] = \
      tmp[:, e[1], ...].copy(), tmp[:, e[0], ...].copy()
  return torch.from_numpy(tmp.reshape(shape)).to(x.device)

def flip_lr_off(x, flip_idx):
  tmp = x.detach().cpu().numpy()[..., ::-1].copy()
  shape = tmp.shape
  tmp = tmp.reshape(tmp.shape[0], 17, 2, 
                    tmp.shape[2], tmp.shape[3])
  tmp[:, :, 0, :, :] *= -1
  for e in flip_idx:
    tmp[:, e[0], ...], tmp[:, e[1], ...] = \
      tmp[:, e[1], ...].copy(), tmp[:, e[0], ...].copy()
  return torch.from_numpy(tmp.reshape(shape)).to(x.device)


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        if self.count > 0:
          self.avg = self.sum / self.count