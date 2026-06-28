import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import path_config
import pickle
import numpy as np
import torch
import motmetrics as mm
from tqdm import tqdm
import time
import glob

from lib.utils.opts import opts
from lib.models.stNet import get_det_net, load_model
from lib.dataset.coco_icpr import COCO
from lib.utils.decode import ctdet_decode
from lib.utils.post_process import generic_post_process
from lib.utils.sort import Sort

def get_opt():
    opt = opts().parse()
   # opt.data_dir = '/root/autodl-tmp/MOT_Dataset'
    
    # 自动寻找最新的权重文件
    weight_pattern = './ICPR_caronly/ResFPN/weights*/model_last.pth'
    weight_files = glob.glob(weight_pattern)
    if not weight_files:
        raise FileNotFoundError(f"No weight file found matching pattern: {weight_pattern}")
    # 按文件修改时间排序，取最新的
    latest_weight = max(weight_files, key=os.path.getmtime)
    opt.load_model = latest_weight
    print(f"Using model: {opt.load_model}")
    
    opt.gpus = '0'
    opt.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    opt.conf_thres = 0.05   #降低评估置信度阈值到0.05（原来是0.3）
    opt.seqLen = 3
    opt.num_classes = 2
    return opt

# 其余函数保持不变（load_model_and_data, process_frame, post_process, merge_outputs,
# run_tracking, load_ground_truth, compute_metrics, save_trajectory_data, main）
# 请确保下面完整复制您原有的这些函数，这里为了简洁省略，但实际使用时必须保留。
# 以下是原有函数的占位符，您需要将您原来的所有函数定义复制到此处。

def load_model_and_data(opt, split='test'):
    dataset = COCO(opt, split)
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=2)
    head = {'hm': opt.num_classes, 'wh': 2, 'reg': 2, 'dis': 2}
    model = get_det_net(head, opt.model_name)
    model = load_model(model, opt.load_model).to(opt.device).eval()
    return dataset, loader, model

def process_frame(model, image, opt):
    with torch.no_grad():
        output_all = model(image)[-1]
        output = output_all[1]
        hm = output['hm'].sigmoid_()
        wh = output['wh']
        reg = output['reg']
        dets = ctdet_decode(hm, wh, reg=reg, num_classes=opt.num_classes, K=opt.K)
        for k in dets:
            dets[k] = dets[k].detach().cpu().numpy()
    return dets

def post_process(dets_all, meta, opt):
    dets = generic_post_process(
        dets_all, [meta['c']], [meta['s']],
        meta['out_height'] // opt.down_ratio, meta['out_width'] // opt.down_ratio)
    return dets[0]

def merge_outputs(dets, conf_thres):
    results = []
    for item in dets:
        if item['score'][0] > conf_thres:
            bbox = item['bbox']
            results.append([bbox[0], bbox[1], bbox[2], bbox[3], item['score'][0], item['class'], item['ind']])
    print(f"检测到 {len(results)} 个目标")   # 验证模型是否为有效检测
    return results

def run_tracking(loader, model, opt):
    tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)
    results = []
    frame_id = 0
    start_time = time.time()
    for _, batch in tqdm(loader, desc='Tracking'):
        frame_id += 1
        image = batch['input'].to(opt.device)
        h, w = image.shape[3], image.shape[4]
        c = np.array([w/2., h/2.], dtype=np.float32)
        s = max(h, w) * 1.0
        meta = {'c': c, 's': s, 'out_height': h, 'out_width': w}

        dets_all = process_frame(model, image, opt)
        dets = post_process(dets_all, meta, opt)
        detections = merge_outputs(dets, opt.conf_thres)

        if len(detections) == 0:
            trackers = tracker.update(np.empty((0,5)))
        else:
            dets_np = np.array([[d[0], d[1], d[2], d[3], d[4]] for d in detections])
            trackers = tracker.update(dets_np)

        for t in trackers:
            x1, y1, x2, y2, track_id = t
            results.append([frame_id, int(track_id), x1, y1, x2, y2, 1, 1, 1])
    elapsed = time.time() - start_time
    fps = frame_id / elapsed if elapsed > 0 else 0
    return results, fps

def load_ground_truth(gt_path):
    gt = []
    with open(gt_path, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) < 6:
                continue
            frame = int(parts[0])
            obj_id = int(parts[1])
            x = float(parts[2])
            y = float(parts[3])
            w = float(parts[4])
            h = float(parts[5])
            x2 = x + w
            y2 = y + h
            gt.append([frame, obj_id, x, y, x2, y2, 1, 1, 1])
    return gt

def compute_metrics(gt, pred, seq_name, out_dir, fps):
    gts = [(g[0], g[1], g[2], g[3], g[4], g[5]) for g in gt]
    preds = [(p[0], p[1], p[2], p[3], p[4], p[5], p[6]) for p in pred]

    acc = mm.MOTAccumulator(auto_id=True)
    frames = set([g[0] for g in gt] + [p[0] for p in pred])
    for f in frames:
        gt_ids = [g[1] for g in gt if g[0]==f]
        gt_boxes = [g[2:6] for g in gt if g[0]==f]
        pred_ids = [p[1] for p in pred if p[0]==f]
        pred_boxes = [p[2:6] for p in pred if p[0]==f]
        if len(gt_ids)==0 and len(pred_ids)==0:
            continue
        ious = mm.distances.iou_matrix(gt_boxes, pred_boxes, max_iou=0.5)
        acc.update(gt_ids, pred_ids, ious)

    mh = mm.metrics.create()
    summary = mh.compute(acc, metrics=['num_frames', 'mota', 'motp', 'idf1', 'idp', 'idr',
                                       'num_switches', 'num_false_positives', 'num_misses',
                                       'num_objects', 'mostly_tracked', 'mostly_lost'])

    def to_scalar(x):
        if hasattr(x, 'iloc'):
            return x.iloc[0]
        return x

    mota = to_scalar(summary['mota']) * 100
    idf1 = to_scalar(summary['idf1']) * 100
    ids = int(to_scalar(summary['num_switches']))
    fp = int(to_scalar(summary['num_false_positives']))
    fn = int(to_scalar(summary['num_misses']))
    mt = int(to_scalar(summary['mostly_tracked']))
    ml = int(to_scalar(summary['mostly_lost']))

    print("\n========== Evaluation Summary ==========")
    print(f"Sequence: {seq_name}")
    print(f"MOTA : {mota:.2f}%")
    print(f"IDF1 : {idf1:.2f}%")
    print(f"IDs  : {ids}")
    print(f"FP   : {fp}")
    print(f"FN   : {fn}")
    print(f"MT   : {mt}")
    print(f"ML   : {ml}")
    print(f"FPS  : {fps:.2f}")
    print("========================================\n")

    with open(os.path.join(out_dir, f'{seq_name}_metrics_summary.txt'), 'w') as f:
        f.write(f"MOTA: {mota:.2f}%\n")
        f.write(f"IDF1: {idf1:.2f}%\n")
        f.write(f"IDs: {ids}\n")
        f.write(f"FP: {fp}\n")
        f.write(f"FN: {fn}\n")
        f.write(f"MT: {mt}\n")
        f.write(f"ML: {ml}\n")
        f.write(f"FPS: {fps:.2f}\n")

    with open(os.path.join(out_dir, f'{seq_name}_metrics_detailed.txt'), 'w') as f:
        f.write(str(summary))

    return summary

def save_trajectory_data(gt, pred, seq_name, out_dir, img_width=128, img_height=128):
    gt_trajs = {}
    pred_trajs = {}
    for g in gt:
        frame, obj_id, x1, y1, x2, y2 = g[:6]
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        if 1 <= cx <= img_width and 1 <= cy <= img_height:
            gt_trajs.setdefault(obj_id, []).append((cx, cy))
    for p in pred:
        frame, obj_id, x1, y1, x2, y2 = p[:6]
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        if 1 <= cx <= img_width and 1 <= cy <= img_height:
            pred_trajs.setdefault(obj_id, []).append((cx, cy))
    data = {
        'gt_trajs': gt_trajs,
        'pred_trajs': pred_trajs,
        'img_width': img_width,
        'img_height': img_height,
        'seq_name': seq_name
    }
    with open(os.path.join(out_dir, f'{seq_name}_trajectory_data.pkl'), 'wb') as f:
        pickle.dump(data, f)
    print(f"轨迹数据已保存: {os.path.join(out_dir, f'{seq_name}_trajectory_data.pkl')}")

def main():
    opt = get_opt()
    seq_name = path_config.TEST_SEQ
    gt_path = os.path.join(opt.data_dir, seq_name, 'gt.txt')
    out_dir = './evaluation_results'
    os.makedirs(out_dir, exist_ok=True)

    dataset, loader, model = load_model_and_data(opt, split='test')
    print('Running tracking on test set...')
    pred_results, fps = run_tracking(loader, model, opt)

    pred_file = os.path.join(out_dir, f'{seq_name}_pred.txt')
    with open(pred_file, 'w') as f:
        for p in pred_results:
            f.write(','.join(map(str, p)) + '\n')
    print(f"预测结果已保存: {pred_file}")

    gt = load_ground_truth(gt_path)
    compute_metrics(gt, pred_results, seq_name, out_dir, fps)

    save_trajectory_data(gt, pred_results, seq_name, out_dir, img_width=128, img_height=128)

    print(f'\n测试完成，结果保存在 {out_dir}')

if __name__ == '__main__':
    main()