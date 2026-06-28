from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import math
import logging

import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
from lib.utils.utils import _transpose_and_gather_feat, _sigmoid
from .common import FeedForward,DASI,PSA_p,SpatiotemporalAttentionFull,SpatiotemporalAttentionBase,SpatiotemporalAttentionFullNotWeightShared
import torch.utils.model_zoo as model_zoo
from .conv_lstm import ConvLSTM
from .deconv_lstm import DeConvLSTM

PSA_en = True
DASI_en = True
FE_en = True

LSTM_en = True

Spatiotemporal_en = True


def get_window_permutation_order():
    """返回窗口划分的维度重排顺序"""
    # 对应维度：(Batch, NumH, NumW, Channel, WindowH, WindowW)
    return (0, 2, 4, 1, 3, 5)


BN_MOMENTUM = 0.1
logger = logging.getLogger(__name__)

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def feature_patching(x, window_size):
    B, C, H, W = x.shape
    num_h, num_w = H // window_size, W // window_size
    
    # 重塑并调整维度
    x = x.view(B, C, num_h, window_size, num_w, window_size)
    permute_order = get_window_permutation_order()
    windows = x.permute(*permute_order).contiguous().view(-1, C, window_size, window_size)
    return windows


class Attention(nn.Module):
    def __init__(self, dim, num_heads=4, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
            
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x1, temp):
        B, C, H, W = x1.shape
        N = H * W
        x1 = x1.view(B, C, -1).transpose(1, 2)    # nW * B, N, C
        temp = temp.view(B, C, -1).transpose(1, 2)    # nW * B, N, C

        q = self.q(x1).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        kv = self.kv(temp).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x.transpose(1, 2).view(B, C, H, W)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            inplanes, 
            planes, 
            kernel_size=3, 
            stride=stride,
            padding=1, 
            bias=False
        )
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            planes, 
            planes, 
            kernel_size=3, 
            stride=1,
            padding=1, 
            bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1,
                               bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion,
                                  momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

def fill_up_weights(up):
    w = up.weight.data
    f = math.ceil(w.size(2) / 2)
    c = (2 * f - 1 - f % 2) / (2. * f)
    for i in range(w.size(2)):
        for j in range(w.size(3)):
            w[0, 0, i, j] = \
                (1 - math.fabs(i / f - c)) * (1 - math.fabs(j / f - c))
    for c in range(1, w.size(0)):
        w[c, 0, :, :] = w[0, 0, :, :] 

def fill_fc_weights(layers):
    for m in layers.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.normal_(m.weight, std=0.001)
            # torch.nn.init.kaiming_normal_(m.weight.data, nonlinearity='relu')
            # torch.nn.init.xavier_normal_(m.weight.data)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
                
def restore_from_patches(windows, window_size, H, W):
    """
    将窗口重新组合为原始特征图
    
    Args:
        windows: 窗口张量，形状为 (num_windows*B, C, window_size, window_size)
        window_size: 窗口大小
        H: 原始图像高度
        W: 原始图像宽度
        
    Returns:
        x: 重组后的特征图，形状为 (B, C, H, W)
    """
    # 计算批量大小
    batch_size = windows.shape[0] // (H * W // window_size**2)
    
    # 计算网格尺寸
    grid_h, grid_w = H // window_size, W // window_size
    
    # 重塑窗口张量为网格结构
    x = windows.view(
        batch_size,     # 批量维度
        grid_h,         # 垂直网格数
        grid_w,         # 水平网格数
        -1,             # 通道维度
        window_size,    # 窗口高度
        window_size     # 窗口宽度
    )
    
    # 调整维度顺序并展平为原始形状
    x = x.permute(0, 3, 1, 4, 2, 5).contiguous()
    x = x.view(batch_size, -1, H, W)
    
    return x
                
class SpatialAttentionModule(nn.Module):
    def __init__(self):
        super(SpatialAttentionModule, self).__init__()
        self.conv2d = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=7, stride=1, padding=3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = torch.mean(x, dim=1, keepdim=True)
        maxout, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avgout, maxout], dim=1)
        out = self.sigmoid(self.conv2d(out))
        return out * x

class ECA(nn.Module):
    def __init__(self,in_channel,gamma=2,b=1):
        super(ECA, self).__init__()
        k=int(abs((math.log(in_channel,2)+b)/gamma))
        kernel_size=k if k % 2 else k+1
        padding=kernel_size//2
        self.pool=nn.AdaptiveAvgPool2d(output_size=1)
        self.conv=nn.Sequential(
            nn.Conv1d(in_channels=1,out_channels=1,kernel_size=kernel_size,padding=padding,bias=False),
            nn.Sigmoid()
        )

    def forward(self,x):
        out=self.pool(x)
        out=out.view(x.size(0),1,x.size(1))
        out=self.conv(out)
        out=out.view(x.size(0),x.size(1),1,1)
        return out*x
    
class ResNet(nn.Module):
    def __init__(self, block, layers, num_layers, pretrain=False):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.psa_en = PSA_en
        self.pretrain = pretrain
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)

        # self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if pretrain:
            self.init_weights(num_layers)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion, momentum=BN_MOMENTUM),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        if self.psa_en and not self.pretrain:
            for i in range(1, blocks):
                layers.append(block(self.inplanes, planes))
            layers.append(SpatialAttentionModule())
            layers.append(ECA(planes))
        else:
            for i in range(1, blocks):
                layers.append(block(self.inplanes, planes))       

        return nn.Sequential(*layers)
    
    def init_weights(self, num_layers):
        url = model_urls['resnet{}'.format(num_layers)]
        pretrained_state_dict = model_zoo.load_url(url)
        print('=> loading pretrained model {}'.format(url))
        self.load_state_dict(pretrained_state_dict, strict=False)
        print('=> init deconv weights from normal distribution')
    
    def forward(self, x, gru1):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        c1 = self.layer1(x + gru1)     # 64 ; 64 * 4 = 256
        c2 = self.layer2(c1)    # 128 ; 128 * 4 = 512
        c3 = self.layer3(c2)    # 256 ; 256 * 4 = 1024

        return c1, c2, c3


class ConvLSTMNet(nn.Module):
    def __init__(self):
        super(ConvLSTMNet, self).__init__()
        self.conv_lstm_layer1 = ConvLSTM(32, 32, (3, 3), 1)
        self.maxpool1 = nn.MaxPool3d(kernel_size=(3,3,3), stride=(1,2,2), padding=(1,1,1))

        self.conv_lstm_layer2 = ConvLSTM(32, 64, (3, 3), 1)
    
    def forward(self, c0_sta):
        # layer 1
        c0_seq, _ = self.conv_lstm_layer1(c0_sta)                    # B, N, 32, 512, 512
        c0_seq_down = self.maxpool1(c0_seq.transpose(1, 2)).transpose(1, 2)     # B, N, 32, 256, 256

        # layer 2
        c1_seq, _ = self.conv_lstm_layer2(c0_seq_down)               # B, N, 64, 256, 256
        return c0_seq, c1_seq


# ========== 模块A：闭环反馈（空间门控） ==========
class DeConvLSTMNet(nn.Module):
    def __init__(self):
        super(DeConvLSTMNet, self).__init__()
        self.proj0 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32)
        )
        self.proj_hm = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32)
        )
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        self.deconv_lstm_layer = DeConvLSTM(32, 32, (3, 3), 1)

        # 闭环反馈组件
        self.fusion_conv = nn.Conv2d(32, 64, kernel_size=1, bias=False)
        self.gate_conv = nn.Sequential(
            nn.Conv2d(64+64, 64, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, p0_sta, c0_lstm):
        B, N, _, H, W = p0_sta.shape
        p0_seq_lstm, _ = self.deconv_lstm_layer(p0_sta)          # B, N, 32, 512, 512
        final_hm = p0_seq_lstm[:, -1] + self.proj_hm(p0_sta[:, -1])
        for i in range(N - 1):
            current_input = torch.cat((p0_seq_lstm[:, i] + c0_lstm[:, i],
                                       p0_seq_lstm[:, i + 1] + c0_lstm[:, i + 1]), dim=1)
            if i == 0:
                combined = current_input
            else:
                prev_out = final_lstm[:, i-1]          # [B,32,H,W]
                prev_out_proj = self.fusion_conv(prev_out)   # [B,64,H,W]
                gate_input = torch.cat([current_input, prev_out_proj], dim=1)  # [B,128,H,W]
                gate = self.gate_conv(gate_input)       # [B,64,H,W]
                combined = current_input + gate * prev_out_proj
            if i == 0:
                final_lstm = self.proj0(combined).unsqueeze(1)
            else:
                final_temp = self.proj0(combined).unsqueeze(1)
                final_lstm = torch.cat((final_lstm, final_temp), dim=1)
        return final_lstm, final_hm


# ========== 模块B：软阈值掩码（用户原始正确版本） ==========
class MaskProp(nn.Module):
    def __init__(self, dim, win_size):
        super(MaskProp, self).__init__()
        self.ws = win_size
        self.low_thr = 0.2   # 保留但不再用于硬筛选（仅用于初始化）
        self.attn_cross = Attention(dim, 4)
        self.bn = nn.BatchNorm2d(dim)
        self.theta = nn.Parameter(torch.tensor(0.2))
        self.tau = nn.Parameter(torch.tensor(0.1))

    def forward(self, pre_feat, feat, mask, name=None):
        B, C, H, W = feat.shape
        _, _, H_org, W_org = mask.shape
        # downsample mask
        mask = F.interpolate(mask, [H, W], mode='bilinear', align_corners=False)

        # ----------- partition -----------
        feat_patch = feature_patching(feat, self.ws)           # Wn, C, Ws, Ws
        pre_feat_patch = feature_patching(pre_feat, self.ws)   # Wn, C, Ws, Ws
        mask_patch = feature_patching(mask, self.ws)           # Wn, 1, Ws, Ws
        Wn, _, _, _ = mask_patch.shape

        # ----------- 软阈值门控 -----------
        mask_max = mask_patch.view(-1, 1, self.ws * self.ws).max(dim=-1)[0].squeeze(1)  # Wn
        weight = torch.sigmoid((mask_max - self.theta) / self.tau)   # [Wn]

        # 对所有窗口计算注意力
        heat_attn_all = self.attn_cross(feat_patch, pre_feat_patch * mask_patch)   # [Wn, C, ws, ws]
        weight = weight.view(-1, 1, 1, 1)   # [Wn, 1, 1, 1]
        heat_attn_all = heat_attn_all * weight
        heat_compe = heat_attn_all   # 直接使用加权后的结果

        # ----------- reverse -----------
        heat_enhance = restore_from_patches(heat_compe, self.ws, H, W)       # B, C, H, W
        heat = self.bn(heat_enhance + feat)
        return heat


class PoseResNet(nn.Module):
    def __init__(self, block, layers, heads, head_conv, num_layers):

        self.inplanes = 256
        self.heads = heads
        self.prev_feat = None
        self.prev_state = False

        super(PoseResNet, self).__init__()
        self.DASI_en = DASI_en
        self.ffn_en = FE_en

        self.Spatiotemporal_en = Spatiotemporal_en
        if self.DASI_en:
            self.cross1 = DASI(64,64)
            self.cross2 = DASI(128,128)
            self.cross3 = DASI(256,256)
        if self.ffn_en:
            self.ffn1 = FeedForward(dim = 64, ffn_expansion_factor=2.08, bias=False)
            self.ffn2 = FeedForward(dim = 128, ffn_expansion_factor=2.08, bias=False)
            self.ffn3 = FeedForward(dim = 256, ffn_expansion_factor=2.08, bias=False)
        if self.Spatiotemporal_en:
            self.Spatiotemporal2 = SpatiotemporalAttentionFull(128)

        # ---------down---------
        # static down
        self.base_layer = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=7, stride=1, padding=3, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True))
        self.backbone = ResNet(block, layers, num_layers, pretrain=True)

        # sequence down
        self.LSTM_en = LSTM_en
        
        self.conv_gru = ConvLSTMNet()
        self.deconv_gru = DeConvLSTMNet()

        # ---------down---------

        # ---------mask prop---------
        self.mask_prop1 = MaskProp(64, 8)
        self.mask_prop2 = MaskProp(128, 8)
        # ---------mask prop---------

        # ---------up---------
        # used for deconv layers
        self.deconv_layer2 = self._make_deconv_layer(256, 128, 3, with_dcn=True)
        self.deconv_layer3 = self._make_deconv_layer(128, 64, 3, with_dcn=True)
        self.deconv_layer4 = self._make_deconv_layer(64, 32, 3, with_dcn=True)

        # self.smooth_layer1 = DeformConv(256, 256)
        self.smooth_layer2 = DeformConv(128, 128)
        self.smooth_layer3 = DeformConv(64, 64)
        self.smooth_layer4 = DeformConv(32, 32)


        # generate layer mask
        self.mask_layer = nn.Conv2d(32, 1, kernel_size=1, stride=1, padding=0, bias=True)
        self.mask_layer.bias.data.fill_(-4.6)
        # ---------up---------

        for head in self.heads:
            classes = self.heads[head]
            if head_conv > 0:
                fc = nn.Sequential(
                    nn.Conv2d(32, head_conv, kernel_size=3, stride=1, padding=1, bias=True),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(head_conv, classes, kernel_size=1, stride=1, padding=0, bias=True))
                if 'hm' in head:
                    fc[-1].bias.data.fill_(-4.6)
                else:
                    fill_fc_weights(fc)
            self.__setattr__(head, fc)


    def _get_deconv_cfg(self, deconv_kernel):
        if deconv_kernel == 4:
            padding = 1
            output_padding = 0
        elif deconv_kernel == 3:
            padding = 1
            output_padding = 1
        elif deconv_kernel == 2:
            padding = 0
            output_padding = 0

        return deconv_kernel, padding, output_padding

    def _make_deconv_layer(self, in_filters, planes, num_kernels, with_dcn=False):
        layers = []

        kernel, padding, output_padding = \
            self._get_deconv_cfg(num_kernels)

        up = nn.ConvTranspose2d(
                in_channels=planes,
                out_channels=planes,
                kernel_size=kernel,
                stride=2,
                padding=padding,
                output_padding=output_padding,
                bias=False)
        fill_up_weights(up)

        if with_dcn:
            fc = nn.Conv2d(in_filters, planes,
                kernel_size=(3,3), stride=1,
                padding=1, dilation=1)
            layers.append(fc)
            layers.append(nn.BatchNorm2d(planes, momentum=BN_MOMENTUM))
            layers.append(nn.ReLU(inplace=True))

        layers.append(up)
        layers.append(nn.BatchNorm2d(planes, momentum=BN_MOMENTUM))
        layers.append(nn.ReLU(inplace=True))
        return nn.Sequential(*layers)

    def train_feat(self, x, gru, i, p1_pre=None, p2_pre=None, mask=None):
        B, C, H, W = x.shape

        if mask == None:
            mask = torch.zeros((B, 1, H, W), dtype=torch.float32, device=x.device)
            p1_pre = torch.zeros((B, 64, H // 2, W // 2), dtype=torch.float32, device=x.device)
            p2_pre = torch.zeros((B, 128, H // 4, W // 4), dtype=torch.float32, device=x.device)
        # static down
        c1, c2, c3 = self.backbone(x, gru[1][:, i])   # 64, 128, 256
        
        c_1 = c1
        c_2 = c2
        c_3 = c3
        if self.DASI_en:
            c1 = self.cross1(c_1,c2,None)
            c2 = self.cross2(c_2,c_3,c_1)
            c3 = self.cross3(c_3,None,c_2)
        if self.ffn_en:
            c1 = self.ffn1(c1)
            c2 = self.ffn2(c2)
            c3 = self.ffn3(c3)
    
        # static up
        p3 = c3                      # B, 256, 64, 64
        p2 = self.smooth_layer2(self.deconv_layer2(p3) + self.mask_prop2(p2_pre, c2, mask))   # B, 128, 128, 128
        p1 = self.smooth_layer3(self.deconv_layer3(p2) + self.mask_prop1(p1_pre, c1, mask))   # B, 64, 256, 256
        p0 = self.smooth_layer4(self.deconv_layer4(p1) + gru[0][:, i])   # B, 32, 512, 512
        return p0, p1, p2, p3

    def track_feat(self, x, gru, i, N, p1_pre=None, p2_pre=None, mask=None, vid=None):
        B, _, H, W = x.shape

        if self.prev_state != False and i < N - 1:
            return self.prev_feat[i + 1][0], self.prev_feat[i + 1][1], self.prev_feat[i + 1][2], self.prev_feat[i + 1][3]
        else:
            if mask == None:
                mask = torch.zeros((B, 1, H, W), dtype=torch.float32, device=x.device)
                p1_pre = torch.zeros((B, 64, H // 2, W // 2), dtype=torch.float32, device=x.device)
                p2_pre = torch.zeros((B, 128, H // 4, W // 4), dtype=torch.float32, device=x.device)
            # static down
            c1, c2, c3 = self.backbone(x, gru[1][:, i])   # 64, 128, 256
            c_1 = c1
            c_2 = c2
            c_3 = c3
            if self.DASI_en:
                c1 = self.cross1(c_1,c2,None)
                c2 = self.cross2(c_2,c_3,c_1)
                c3 = self.cross3(c_3,None,c_2)
            if self.ffn_en:
                c1 = self.ffn1(c1)
                c2 = self.ffn2(c2)
                c3 = self.ffn3(c3)
            # static up
            p3 = c3                      # B, 256, 64, 64
            p2 = self.smooth_layer2(self.deconv_layer2(p3) + self.mask_prop2(p2_pre, c2, mask, None))   # B, 128, 128, 128
            p1 = self.smooth_layer3(self.deconv_layer3(p2) + self.mask_prop1(p1_pre, c1, mask, vid+'_p1'))   # B, 64, 256, 256
            p0 = self.smooth_layer4(self.deconv_layer4(p1) + gru[0][:, i])   # B, 32, 512, 512
            return p0, p1, p2, p3

    def forward(self, img_input, training=True, vid=None):
        # x : B, N, C, H, W
        # ..., -3, -2, -1, 0
        B, N, _, H, W = img_input.shape

        mask = None
        p1_pre = None
        p2_pre = None
        gru_set = []
        temp_feat = []
        ret_temp = {}

        # ------------------down begin------------------
        # sequence down
        for i in range(N):
            if i == 0:
                c0_seq = self.base_layer(img_input[:, i]).unsqueeze(1)
            else:
                c0_temp = self.base_layer(img_input[:, i]).unsqueeze(1)
                c0_seq = torch.cat((c0_seq, c0_temp), dim=1)
        c0_gru, c1_gru = self.conv_gru(c0_seq)
        gru_set.extend([c0_seq + c0_gru, c1_gru])

        # static down and up
        for i in range(N):
            x = img_input[:, i, :]
            if training:
                p0, p1, p2, p3 = self.train_feat(x, gru_set, i, p1_pre, p2_pre, mask)
                temp_feat.append([p0, p1, p2, p3])    # N, 4
            else:
                p0, p1, p2, p3 = self.track_feat(x, gru_set, i, N, p1_pre, p2_pre, mask, vid)
                temp_feat.append([p0, p1, p2, p3])    # N, 4

            if i > 0 and self.Spatiotemporal_en:
                p2_,p2 = self.Spatiotemporal2(p2_pre,p2)

            p1_pre = p1
            p2_pre = p2

            # mask prop
            if i == 0:
                mask_out = self.mask_layer(p0).unsqueeze(1)                                 # B, N, 1, 512, 512
            else:
                mask_out = torch.cat((mask_out, self.mask_layer(p0).unsqueeze(1)), dim=1)

            mask = F.sigmoid(mask_out[:, i])

            # construct sequence
            if i == 0:
                p0_seq = p0.unsqueeze(1)
                p1_seq = p1.unsqueeze(1)
                p2_seq = p2.unsqueeze(1)
            else:
                p0_seq = torch.cat((p0_seq, p0.unsqueeze(1)), dim=1)    # B, N, 32, 512, 512
                p1_seq = torch.cat((p1_seq, p1.unsqueeze(1)), dim=1)    # B, N, 64, 256, 256
                p2_seq = torch.cat((p2_seq, p2.unsqueeze(1)), dim=1)    # B, N, 128, 128, 128
        # ------------------down end------------------

        # ------------------up begin------------------
        # sequence up
        final_seq, final_hm = self.deconv_gru(p0_seq, c0_gru)       # B, N - 1, 32, 512, 512
        # ------------------up end------------------

        ret_temp['hm_seq'] = mask_out       # B, N, 1, 512, 512

        # only for track
        if not training:
            self.prev_feat = temp_feat
            self.prev_state = True

        ret = {}
        for head in self.heads:
            if 'dis' in head:
                trk_out = self.__getattr__(head)(final_seq[:, 0]).unsqueeze(1)
                for i in range(1, N - 1):
                    x = self.__getattr__(head)(final_seq[:, i]).unsqueeze(1)
                    trk_out = torch.cat((trk_out, x), dim=1)
                ret_temp[head] = trk_out
            else:
                ret_temp[head] = self.__getattr__(head)(final_hm)
        ret[1] = ret_temp

        return [temp_feat, ret]


class DeformConv(nn.Module):
    def __init__(self, chi, cho):
        super(DeformConv, self).__init__()
        self.actf = nn.Sequential(
            nn.BatchNorm2d(cho, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True)
        )
        self.conv = nn.Conv2d(chi, cho, kernel_size=(3, 3), stride=1, padding=1, dilation=1)
        for name, m in self.actf.named_modules():
            if isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.conv(x)
        x = self.actf(x)
        return x


resnet_spec = {18: (BasicBlock, [2, 2, 2, 2]),
               34: (BasicBlock, [3, 4, 6, 3]),
               50: (Bottleneck, [3, 4, 6, 3]),
               101: (Bottleneck, [3, 4, 23, 3]),
               152: (Bottleneck, [3, 8, 36, 3])}


def res_fpn_net(heads, head_conv=128):
    num_layers = 34
    block_class, layers = resnet_spec[num_layers]

    model = PoseResNet(block_class, layers, heads, head_conv=head_conv, num_layers=num_layers)
    return model


class FEM(nn.Module):
    def __init__(self, in_planes, out_planes, stride=1, scale=0.1, map_reduce=8):
        super(FEM, self).__init__()
        self.scale = scale
        self.out_channels = out_planes
        inter_planes = in_planes // map_reduce
        self.branch0 = nn.Sequential(
            Basicconv(in_planes, 2 * inter_planes, kernel_size=1, stride=stride),
            Basicconv(2 * inter_planes, 2 * inter_planes, kernel_size=3, stride=1, padding=1, relu=False)
        )
        self.branch1 = nn.Sequential(
            Basicconv(in_planes, inter_planes, kernel_size=1, stride=1),
            Basicconv(inter_planes, (inter_planes // 2) * 3, kernel_size=(1, 3), stride=stride, padding=(0, 1)),
            Basicconv((inter_planes // 2) * 3, 2 * inter_planes, kernel_size=(3, 1), stride=stride, padding=(1, 0)),
            Basicconv(2 * inter_planes, 2 * inter_planes, kernel_size=3, stride=1, padding=5, dilation=5, relu=False)
        )
        self.branch2 = nn.Sequential(
            Basicconv(in_planes, inter_planes, kernel_size=1, stride=1),
            Basicconv(inter_planes, (inter_planes // 2) * 3, kernel_size=(3, 1), stride=stride, padding=(1, 0)),
            Basicconv((inter_planes // 2) * 3, 2 * inter_planes, kernel_size=(1, 3), stride=stride, padding=(0, 1)),
            Basicconv(2 * inter_planes, 2 * inter_planes, kernel_size=3, stride=1, padding=5, dilation=5, relu=False)
        )

        self.convLinear = Basicconv(6 * inter_planes, out_planes, kernel_size=1, stride=1, relu=False)
        self.shortcut = Basicconv(in_planes, out_planes, kernel_size=1, stride=stride, relu=False)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)

        out = torch.cat((x0, x1, x2), 1)
        out = self.convLinear(out)
        short = self.shortcut(x)
        out = out * self.scale + short
        out = self.relu(out)

        return out