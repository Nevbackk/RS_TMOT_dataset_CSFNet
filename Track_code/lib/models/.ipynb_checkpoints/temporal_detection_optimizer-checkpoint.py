import torch
import numpy as np
from scipy.optimize import linear_sum_assignment
from collections import defaultdict
from lib.utils.utils import _transpose_and_gather_feat
import copy

def hungarian_assignment(dist):
    """
    使用匈牙利算法解决二分图最优匹配问题
    
    Args:
        dist: 代价矩阵，形状为 [N, M]
        
    Returns:
        匹配索引对，形状为 [K, 2]，其中 K 是匹配数量
    """
    if dist.size == 0:  # 处理空矩阵情况
        return np.empty((0, 2), dtype=np.int32)
    
    # 复制距离矩阵以避免修改原始数据
    cost_matrix = dist.copy()
    
    # 将极大值替换为合理的大值，避免匈牙利算法出错
    max_valid = np.max(cost_matrix[cost_matrix < 1e16])
    cost_matrix[cost_matrix >= 1e16] = max_valid * 1000
    
    # 使用匈牙利算法求解
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    
    # 筛选出有效匹配（距离小于阈值）
    valid_matches = cost_matrix[row_ind, col_ind] < max_valid * 1000
    valid_row = row_ind[valid_matches]
    valid_col = col_ind[valid_matches]
    
    # 返回匹配结果
    return np.column_stack((valid_row, valid_col)).astype(np.int32)

def check(opt, file_folder_buffer, dets_buffer, dis_buffer, inds_buffer):
    """
    多帧检测结果关联优化
    
    Args:
        opt: 配置参数
        file_folder_buffer: 文件路径缓冲区
        dets_buffer: 检测结果缓冲区 [seqLen, num_dets, 7]
        dis_buffer: 位移信息缓冲区 [seqLen-1, B, 2, H, W]
        inds_buffer: 索引缓冲区 [seqLen, num_inds]
        
    Returns:
        dets_buffer: 优化后的检测结果缓冲区
        inds_buffer: 优化后的索引缓冲区
    """
    # 检查文件路径是否变化
    if file_folder_buffer[opt.seqLen - 1] != file_folder_buffer[0]:
        return dets_buffer, inds_buffer
    
    # 初始化当前帧位移图
    cur_dismap = dis_buffer[opt.seqLen - 1]  # dis: B, N-1, 2, H, W
    _, _, _, H, W = cur_dismap.shape
    
    # 初始化结果字典
    cur_dets = defaultdict(list)
    whs = defaultdict(list)
    hits = defaultdict(list)
    match_inds = defaultdict(list)
    unmatch_preds = defaultdict(list)
    unmatch_curs = defaultdict(list)
    inds_temp_buffer = copy.deepcopy(inds_buffer)
    
    # 多帧关联处理循环
    for i in range(opt.seqLen - 1):
        # 合并当前帧检测结果
        cur_dets[i] = cur_dets[i] + dets_buffer[i + 1]
        M = len(cur_dets[i])
        
        # 提取当前帧检测中心和尺寸
        cur_cts = np.array([((item[0] + item[2]) / 2, (item[1] + item[3]) / 2) for item in cur_dets[i]])
        cur_whs = np.array([[(item[2] - item[0]), (item[3] - item[1]), item[4], item[5], item[6]] for item in cur_dets[i]])
        cur_sizes = np.array([(item[2] - item[0]) * (item[3] - item[1]) for item in cur_dets[i]])
        
        # 提取前一帧检测尺寸
        if i == 0:
            pre_whs = np.array([[(item[2] - item[0]), (item[3] - item[1]), item[4], item[5], item[6]] for item in dets_buffer[0]])
        else:
            pre_whs = np.array(whs[i - 1])
        K = len(inds_temp_buffer[i])
        
        # 计算前一帧检测中心坐标
        pre_ind = torch.from_numpy(inds_temp_buffer[i]).unsqueeze(0)
        pre_ys = (pre_ind / W).int().float()
        pre_xs = (pre_ind % W).int().float()
        pre_cts = torch.cat([pre_xs.unsqueeze(-1), pre_ys.unsqueeze(-1)], dim=2)
        
        # 计算前一帧到当前帧的位移
        pre2cur_dis = _transpose_and_gather_feat(cur_dismap[:, i], pre_ind).view(1, -1, 2)
        pred_cts = (pre_cts + pre2cur_dis)[0].numpy()
        pred_cts = np.clip(pred_cts, 0, max(W-1, 0))
        pred_dis = np.sum(pre2cur_dis[0].numpy()**2, axis=1)
        pre_sizes = np.array([item[0] * item[1] for item in pre_whs])
        
        # 计算匹配距离矩阵
        dists = np.sum((cur_cts.reshape(1, -1, 2) - pred_cts.reshape(-1, 1, 2))** 2, axis=2)
        invalid = (dists > cur_sizes.reshape(1, M)) | (dists > pre_sizes.reshape(K, 1))
        dists = dists + invalid * 1e18
        
        # 匈牙利算法匹配
        match_inds[i + 1] = hungarian_assignment(copy.deepcopy(dists))
        if i == 0:
            unmatch_preds[i + 1] = [d for d in range(pred_cts.shape[0]) if d not in match_inds[i + 1][:, 0]]
            unmatch_curs[i + 1] = [d for d in range(cur_cts.shape[0]) if d not in match_inds[i + 1][:, 1]]
        
        # 更新检测结果和索引
        if i == 0:
            inds_temp = inds_temp_buffer[i + 1].tolist()
            whs_temp = cur_whs.tolist()
            
            cts_update = pred_cts[unmatch_preds[i + 1]].astype(np.int32)
            inds_update = cts_update[:, 0] + cts_update[:, 1] * W
            whs_update = pre_whs[unmatch_preds[i + 1]].tolist()
            unmatch_pred0_update = (M + np.arange(len(unmatch_preds[i + 1]))).tolist()
            
            inds_temp += inds_update.tolist()
            inds_temp_buffer[i + 1] = np.array(inds_temp, np.int64)
            whs[i] = whs_temp + whs_update
            
            hits[i] = np.zeros_like(inds_temp_buffer[i + 1])
            hits[i][match_inds[i + 1][:, 1]] = 1
        else:
            hits[i] = np.zeros_like(inds_temp_buffer[i])
            hits[i][match_inds[i + 1][:, 0]] = 1
            cts_update = pred_cts.astype(np.int32)
            inds_update = cts_update[:, 0] + cts_update[:, 1] * W
            
            for j in range(match_inds[i + 1].shape[0]):
                pre_whs[match_inds[i + 1][j, 0]][:5] = cur_whs[match_inds[i + 1][j, 1]][:5]
                inds_update[match_inds[i + 1][j, 0]] = cur_whs[match_inds[i + 1][j, 1]][4]
            
            inds_temp_buffer[i + 1] = np.array(inds_update, np.int64)
            whs[i] = pre_whs.tolist()
    
    # 处理第一帧检测结果
    hits_sum = hits[0] + hits[1] + hits[2] + hits[3] if opt.seqLen == 5 else hits[0] + hits[1] + hits[2]
    remain_inds = np.where(hits_sum > 0)[0]
    
    inds_temp1 = inds_buffer[1].tolist()
    inds_temp2 = inds_buffer[2].tolist()
    inds_temp3 = inds_buffer[3].tolist()
    inds_temp4 = inds_buffer[4].tolist() if opt.seqLen == 5 else []
    
    # 处理假阳性(FP)
    FP_ind = [i for i, ind in enumerate(inds_temp1) 
              if ind not in inds_temp_buffer[1][remain_inds].tolist() and i in unmatch_curs[1]]
    for i in sorted(FP_ind, reverse=True):
        del dets_buffer[1][i]
        del inds_temp1[i]
    
    # 处理假阴性(FN)
    unmatch_pred0_update = [M + i for i in unmatch_preds[1]]
    for i, num in enumerate(unmatch_pred0_update):
        temp_ind = inds_temp_buffer[1][num]
        if 0 < temp_ind < H * W:
            ct_y, ct_x = temp_ind // W, temp_ind % W
            w, h, score, cls, _ = whs[0][num]
            if num in remain_inds:
                hits[0][num] += 1
                inds_temp1.append(temp_ind)
                dets_buffer[1].append([
                    ct_x - w/2, ct_y - h/2, 
                    ct_x + w/2, ct_y + h/2, 
                    score, cls, temp_ind
                ])
    inds_buffer[1] = np.array(inds_temp1)
    
    # 处理后续帧检测结果
    if opt.seqLen == 5:
        hits_sum = hits[1] + hits[2] + hits[3]
        remain_inds = np.where((hits[0] > 0) & (hits_sum > 1))[0]
        
        FN_ind = [i for i in remain_inds if hits[1][i] == 0]
        for i in FN_ind:
            temp_ind = inds_temp_buffer[2][i]
            if 0 < temp_ind < H * W:
                ct_y, ct_x = temp_ind // W, temp_ind % W
                w, h, score, cls, _ = whs[1][i]
                inds_temp2.append(temp_ind)
                dets_buffer[2].append([
                    ct_x - w/2, ct_y - h/2, 
                    ct_x + w/2, ct_y + h/2, 
                    score, cls, temp_ind
                ])
        inds_buffer[2] = np.array(inds_temp2)
        
        FN_ind = [i for i in remain_inds if hits[2][i] == 0]
        for i in FN_ind:
            temp_ind = inds_temp_buffer[3][i]
            if 0 < temp_ind < H * W:
                ct_y, ct_x = temp_ind // W, temp_ind % W
                w, h, score, cls, _ = whs[2][i]
                inds_temp3.append(temp_ind)
                dets_buffer[3].append([
                    ct_x - w/2, ct_y - h/2, 
                    ct_x + w/2, ct_y + h/2, 
                    score, cls, temp_ind
                ])
        inds_buffer[3] = np.array(inds_temp3)
    
    return dets_buffer, inds_buffer