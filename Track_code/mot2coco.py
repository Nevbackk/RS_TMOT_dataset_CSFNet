import os
import json
import cv2
from collections import defaultdict

# 导入统一的配置文件
import path_config

def get_image_size(img_dir):
    """获取第一张图片的尺寸，若失败返回默认 1920x1080"""
    img_files = [f for f in os.listdir(img_dir) if f.endswith(('.jpg', '.png'))]
    if img_files:
        img = cv2.imread(os.path.join(img_dir, img_files[0]))
        if img is not None:
            return img.shape[1], img.shape[0]  # width, height
    return 1920, 1080

def convert_mot_to_coco(data_root, output_dir, train_seqs, val_seqs, test_seqs, train_ratio=0.8):
    """
    按帧划分：每个序列的前 train_ratio 帧作为训练，后 1-train_ratio 帧作为验证。
    测试序列取全部帧。
    """
    os.makedirs(output_dir, exist_ok=True)
    categories = [{"id": 1, "name": "vehicle"}]

    splits_data = {
        'train': {'images': [], 'annotations': [], 'frame_to_img_id': {}, 'img_id': 1, 'ann_id': 1},
        'val':   {'images': [], 'annotations': [], 'frame_to_img_id': {}, 'img_id': 1, 'ann_id': 1},
        'test':  {'images': [], 'annotations': [], 'frame_to_img_id': {}, 'img_id': 1, 'ann_id': 1}
    }

    def process_seq(seq_name, split, frame_range=None):
        seq_path = os.path.join(data_root, seq_name)
        img_dir = os.path.join(seq_path, 'img1')
        gt_path = os.path.join(seq_path, 'gt.txt')
        if not os.path.exists(img_dir) or not os.path.exists(gt_path):
            print(f"跳过 {seq_name}: 缺少 img1 或 gt.txt")
            return

        width, height = get_image_size(img_dir)
        all_img_files = sorted([f for f in os.listdir(img_dir) if f.endswith(('.jpg', '.png'))])
        if not all_img_files:
            return

        # 根据帧范围截取
        if frame_range is not None:
            start, end = frame_range
            img_files = all_img_files[start:end]
        else:
            img_files = all_img_files

        data = splits_data[split]
        # 为每张图片分配 image_id
        for img_file in img_files:
            frame_id = int(os.path.splitext(img_file)[0])
            coco_file_name = f"{seq_name}_{frame_id:06d}.jpg"
            data['images'].append({
                "id": data['img_id'],
                "file_name": coco_file_name,
                "width": width,
                "height": height
            })
            data['frame_to_img_id'][(seq_name, frame_id)] = data['img_id']
            data['img_id'] += 1

        # 解析 gt.txt
        with open(gt_path, 'r') as f:
            for line in f:
                parts = line.strip().split(',')
                if len(parts) < 9:
                    continue
                frame = int(parts[0])
                obj_id = int(parts[1])          # 目标ID
                x = float(parts[2])
                y = float(parts[3])
                w = float(parts[4])
                h = float(parts[5])
                # cat_id = int(parts[7]) if len(parts) > 7 else 1
                key = (seq_name, frame)
                if key not in data['frame_to_img_id']:
                    continue
                image_id = data['frame_to_img_id'][key]
                ann = {
                    "id": data['ann_id'],
                    "image_id": image_id,
                    "category_id": 1,          # 统一为第1类
                    "bbox": [x, y, w, h],
                    "area": w * h,
                    "iscrowd": 0,
                    "obj_id": obj_id           # 添加目标ID
                }
                data['annotations'].append(ann)
                data['ann_id'] += 1

    # 1. 处理训练集：每个序列取前 train_ratio 帧
    for seq in train_seqs:
        seq_path = os.path.join(data_root, seq)
        img_dir = os.path.join(seq_path, 'img1')
        img_files = sorted([f for f in os.listdir(img_dir) if f.endswith(('.jpg', '.png'))])
        n = len(img_files)
        split_idx = int(n * train_ratio)
        process_seq(seq, 'train', frame_range=(0, split_idx))

    # 2. 处理验证集：每个序列取后 1-train_ratio 帧
    for seq in val_seqs:
        seq_path = os.path.join(data_root, seq)
        img_dir = os.path.join(seq_path, 'img1')
        img_files = sorted([f for f in os.listdir(img_dir) if f.endswith(('.jpg', '.png'))])
        n = len(img_files)
        split_idx = int(n * train_ratio)
        process_seq(seq, 'val', frame_range=(split_idx, n))

    # 3. 处理测试集：全部帧
    for seq in test_seqs:
        process_seq(seq, 'test')

    # 保存 JSON 文件
    for split in ['train', 'val', 'test']:
        data = splits_data[split]
        if not data['images']:
            print(f"警告: {split} 没有图片，跳过")
            continue
        coco_dict = {
            "images": data['images'],
            "annotations": data['annotations'],
            "categories": categories
        }
        output_json = os.path.join(output_dir, f"instances_{split}.json")
        with open(output_json, 'w') as f:
            json.dump(coco_dict, f, indent=2)
        print(f"已生成 {output_json}: {len(data['images'])} 张图片, {len(data['annotations'])} 个标注")

def main():
    # 从统一的 path_config 中读取配置
    data_root = path_config.DATA_ROOT
    output_dir = path_config.ANNOTATIONS_DIR
    train_seqs = path_config.TRAIN_SEQS
    val_seqs = path_config.VAL_SEQS
    test_seqs = path_config.TEST_SEQS
    train_ratio = getattr(path_config, 'TRAIN_RATIO', 0.8)

    convert_mot_to_coco(data_root, output_dir, train_seqs, val_seqs, test_seqs, train_ratio)
    print("VISO 数据集转换完成！")

if __name__ == '__main__':
    main()