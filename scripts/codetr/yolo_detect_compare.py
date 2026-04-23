#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YOLO 检测 + Co-DETR 对比工具

功能：
1. 使用预训练 YOLO 模型检测行人和车辆
2. 与 Co-DETR 的标注结果对比，找出误检/漏检

使用:
    python yolo_detect_compare.py detect -i <图片目录> [-c <置信度>] [-m <模型路径>]
    python yolo_detect_compare.py compare --yolo <yolo标注> --codetr <codetr标注> --images <图片目录>
"""

import argparse
import os
import shutil
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm
import torch
from torchvision.ops import nms

# ==================== 类别配置 ====================
# YOLO class 0: person (行人)
# YOLO class 1: vehicle (车辆)

# 针对自训练 train_det 模型的类别映射
# train_det 模型: 0=person, 1=car, 2=background
TRAINDET_TO_YOLO = {
    0: 0,   # person -> person
    1: 1,   # car -> vehicle
}

# 针对 COCO 预训练模型的类别映射 (如 yolo11n.pt)
COCO_TO_YOLO = {
    0: 0,   # person -> person
    2: 1,   # car -> vehicle
    5: 1,   # bus -> vehicle
    7: 1,   # truck -> vehicle
}

# 默认使用 train_det 映射
MODEL_CLASS_MAP = TRAINDET_TO_YOLO

TARGET_COCO_IDS = list(MODEL_CLASS_MAP.keys())
CLASS_NAMES = {0: 'person', 1: 'vehicle'}
COLORS = {0: (0, 255, 0), 1: (255, 0, 0)}  # person: 绿色, vehicle: 蓝色

# 行人框最小尺寸过滤
PERSON_MIN_WIDTH = 32   # 行人框最小宽度 (像素)
PERSON_MIN_HEIGHT = 64  # 行人框最小高度 (像素)


# ==================== 工具函数 ====================

def xywh_to_xyxy(box, img_w, img_h):
    """YOLO 格式 (cx, cy, w, h) 归一化 -> (x1, y1, x2, y2) 像素坐标"""
    cx, cy, w, h = box
    x1 = (cx - w / 2) * img_w
    y1 = (cy - h / 2) * img_h
    x2 = (cx + w / 2) * img_w
    y2 = (cy + h / 2) * img_h
    return [x1, y1, x2, y2]


def xyxy_to_xywh(box, img_w, img_h):
    """(x1, y1, x2, y2) 像素坐标 -> YOLO 格式 (cx, cy, w, h) 归一化"""
    x1, y1, x2, y2 = box
    cx = (x1 + x2) / 2 / img_w
    cy = (y1 + y2) / 2 / img_h
    w = (x2 - x1) / img_w
    h = (y2 - y1) / img_h
    return [cx, cy, w, h]


def compute_iou(box1, box2):
    """计算两个框的 IoU"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter

    return inter / union if union > 0 else 0


def filter_boxes_by_size(boxes, classes, scores=None):
    """过滤行人框：宽度 >= PERSON_MIN_WIDTH, 高度 >= PERSON_MIN_HEIGHT
    返回过滤后的 (boxes, classes, scores)
    """
    filtered_boxes = []
    filtered_classes = []
    filtered_scores = [] if scores is not None else None
    
    for i, (box, cls_id) in enumerate(zip(boxes, classes)):
        x1, y1, x2, y2 = box
        box_w = x2 - x1
        box_h = y2 - y1
        
        # 如果是行人 (class 0)，检查尺寸
        if cls_id == 0:  # person
            if box_w < PERSON_MIN_WIDTH or box_h < PERSON_MIN_HEIGHT:
                continue  # 跳过太小的行人框
        
        filtered_boxes.append(box)
        filtered_classes.append(cls_id)
        if scores is not None:
            filtered_scores.append(scores[i])
    
    return filtered_boxes, filtered_classes, filtered_scores


def load_yolo_labels(label_path, img_w, img_h, target_classes=None, score_thres=None):
    """加载 YOLO 格式标注，返回 (boxes, classes, scores) 列表
    支持带得分和不带得分的格式
    """
    boxes = []
    classes = []
    scores = []
    if not os.path.exists(label_path):
        return boxes, classes, scores

    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                cls_id = int(parts[0])
                xywh = [float(x) for x in parts[1:5]]
                # 如果有第6列，它是得分
                score = float(parts[5]) if len(parts) >= 6 else 1.0
                
                # 类别过滤
                if target_classes is not None and cls_id not in target_classes:
                    continue
                # 得分过滤
                if score_thres is not None and score < score_thres:
                    continue
                    
                xyxy = xywh_to_xyxy(xywh, img_w, img_h)
                boxes.append(xyxy)
                classes.append(cls_id)
                scores.append(score)
    return boxes, classes, scores


def save_yolo_labels_no_score(label_path, boxes, classes, img_w, img_h):
    """保存 YOLO 格式标注 (不包含得分)
    格式: class_id cx cy w h
    """
    with open(label_path, 'w') as f:
        for box, cls_id in zip(boxes, classes):
            xywh = xyxy_to_xywh(box, img_w, img_h)
            f.write(f"{cls_id} {xywh[0]:.6f} {xywh[1]:.6f} {xywh[2]:.6f} {xywh[3]:.6f}\n")


def save_yolo_labels(label_path, boxes, classes, img_w, img_h, scores=None):
    """保存 YOLO 格式标注，支持得分
    格式: class_id cx cy w h [score]
    """
    with open(label_path, 'w') as f:
        for i, (box, cls_id) in enumerate(zip(boxes, classes)):
            xywh = xyxy_to_xywh(box, img_w, img_h)
            if scores is not None and i < len(scores):
                f.write(f"{cls_id} {xywh[0]:.6f} {xywh[1]:.6f} {xywh[2]:.6f} {xywh[3]:.6f} {scores[i]:.4f}\n")
            else:
                f.write(f"{cls_id} {xywh[0]:.6f} {xywh[1]:.6f} {xywh[2]:.6f} {xywh[3]:.6f}\n")


# ==================== 检测模式 ====================

def run_detection(args):
    """使用 YOLO 模型运行检测"""
    try:
        from ultralytics import YOLO
    except ImportError:
        print("❌ ultralytics 未安装，请运行: pip install ultralytics")
        return

    print(f"🚀 加载 YOLO 模型: {args.model}")
    model = YOLO(args.model)

    # 获取图片列表
    img_dir = Path(args.input)
    image_files = sorted(list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png")))

    output_dir = img_dir.parent / "labels_yolo"
    vis_dir = img_dir.parent / "vis_yolo"
    output_dir.mkdir(exist_ok=True)
    vis_dir.mkdir(exist_ok=True)

    print(f"📂 输入: {args.input}")
    print(f"📁 标注输出: {output_dir}")
    print(f"🖼️  可视化输出: {vis_dir}")
    print(f"📊 置信度阈值: {args.conf_thres}")
    print(f"📊 NMS IoU 阈值: {args.nms_thres}")
    print(f"🔍 共 {len(image_files)} 张图片")

    detected_count = 0

    for img_path in tqdm(image_files, desc="YOLO 检测"):
        img = cv2.imread(str(img_path))
        if img is None:
            continue

        h, w = img.shape[:2]

        # YOLO 推理
        results = model.predict(img, conf=args.conf_thres, iou=args.nms_thres, verbose=False)

        all_boxes = []
        all_classes = []
        all_scores = []

        for result in results:
            if result.boxes is not None:
                for box in result.boxes:
                    coco_cls = int(box.cls.item())
                    if coco_cls in MODEL_CLASS_MAP:
                        yolo_cls = MODEL_CLASS_MAP[coco_cls]
                        xyxy = box.xyxy[0].cpu().numpy().tolist()
                        score = float(box.conf.item())
                        all_boxes.append(xyxy)
                        all_classes.append(yolo_cls)
                        all_scores.append(score)

        # 保存标注 (包含得分)
        label_path = output_dir / f"{img_path.stem}.txt"
        save_yolo_labels(label_path, all_boxes, all_classes, w, h, scores=all_scores)

        if len(all_boxes) > 0:
            detected_count += 1

        # 可视化
        vis_img = img.copy()
        for box, cls_id in zip(all_boxes, all_classes):
            x1, y1, x2, y2 = map(int, box)
            color = COLORS.get(cls_id, (0, 255, 0))
            cv2.rectangle(vis_img, (x1, y1), (x2, y2), color, 2)
            label = CLASS_NAMES.get(cls_id, 'obj')
            cv2.putText(vis_img, label, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        cv2.imwrite(str(vis_dir / img_path.name), vis_img)

    print(f"\n✅ 检测完成！检测到目标的图片: {detected_count}/{len(image_files)}")


# ==================== 对比模式 ====================

def run_compare(args):
    """对比 YOLO 和 Co-DETR 的检测结果"""
    yolo_dir = Path(args.yolo_labels)
    codetr_dir = Path(args.codetr_labels)
    img_dir = Path(args.images)

    if not yolo_dir.exists():
        print(f"❌ YOLO 标注目录不存在: {yolo_dir}")
        return
    if not codetr_dir.exists():
        print(f"❌ Co-DETR 标注目录不存在: {codetr_dir}")
        return

    # 从目录名推断模型名称
    codetr_model = codetr_dir.name.replace("labels_", "")
    yolo_model = "yolo"
    
    # 输出目录结构: results/{codetr_model}_{yolo_model}/
    output_base = img_dir.parent / f"{codetr_model}_{yolo_model}"
    
    # 子目录
    image_dir = output_base / "image"           # 所有有问题的图片
    label_dir = output_base / "label"           # 对应的标注
    label_vis_dir = output_base / "label_vis"   # Co-DETR 检测可视化 (带类别和得分)
    fp_only_dir = output_base / "fp_only"       # 仅误检
    fn_only_dir = output_base / "fn_only"       # 仅漏检
    fp_fn_both_dir = output_base / "fp_fn_both" # 同时有误检和漏检
    compare_vis_dir = output_base / "compare_vis"  # 对比可视化

    # 清空输出目录
    if output_base.exists():
        print(f"🗑️  清空输出目录: {output_base}")
        shutil.rmtree(output_base)
    
    for d in [output_base, image_dir, label_dir, label_vis_dir, fp_only_dir, fn_only_dir, fp_fn_both_dir, compare_vis_dir]:
        d.mkdir(exist_ok=True)

    image_files = sorted(list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png")))

    iou_threshold = args.iou_thres
    gt_score_thres = args.gt_score_thres if hasattr(args, 'gt_score_thres') else 0.5

    fp_only_images = []      # 仅误检
    fn_only_images = []      # 仅漏检
    fp_fn_both_images = []   # 同时有误检和漏检
    all_problem_images = []  # 所有有问题的图片

    match_stats = {'person': {'fp': 0, 'fn': 0, 'tp': 0}, 'vehicle': {'fp': 0, 'fn': 0, 'tp': 0}}

    print(f"📁 输出目录: {output_base}")
    print(f"📊 IoU 阈值: {iou_threshold}")
    print(f"📊 GT 得分阈值: {gt_score_thres}")
    print(f"🔍 对比 {len(image_files)} 张图片...")

    for img_path in tqdm(image_files, desc="对比中"):
        img = cv2.imread(str(img_path))
        if img is None:
            continue

        h, w = img.shape[:2]
        stem = img_path.stem

        # 加载标注
        yolo_label = yolo_dir / f"{stem}.txt"
        codetr_label = codetr_dir / f"{stem}.txt"

        # YOLO 标注不过滤得分 (已在检测时过滤)
        yolo_boxes, yolo_classes, _ = load_yolo_labels(yolo_label, w, h)
        # Co-DETR 标注按 GT 得分阈值过滤
        codetr_boxes, codetr_classes, codetr_scores = load_yolo_labels(codetr_label, w, h, score_thres=gt_score_thres)
        # 对行人框按尺寸过滤 (宽>=32, 高>=64)
        codetr_boxes, codetr_classes, codetr_scores = filter_boxes_by_size(codetr_boxes, codetr_classes, codetr_scores)

        # 匹配分析
        yolo_matched = [False] * len(yolo_boxes)
        codetr_matched = [False] * len(codetr_boxes)

        # 对每个 CoDETR 框找最佳匹配的 YOLO 框
        for i, (gt_box, gt_cls) in enumerate(zip(codetr_boxes, codetr_classes)):
            best_iou = 0
            best_j = -1
            for j, (pred_box, pred_cls) in enumerate(zip(yolo_boxes, yolo_classes)):
                if pred_cls == gt_cls:  # 同类才匹配
                    iou = compute_iou(gt_box, pred_box)
                    if iou > best_iou:
                        best_iou = iou
                        best_j = j

            if best_iou >= iou_threshold:
                codetr_matched[i] = True
                if best_j >= 0:
                    yolo_matched[best_j] = True
                cls_name = CLASS_NAMES.get(gt_cls, 'other')
                match_stats[cls_name]['tp'] += 1

        # 统计误检和漏检
        has_fp = False
        has_fn = False

        for j, matched in enumerate(yolo_matched):
            if not matched:
                has_fp = True
                cls_name = CLASS_NAMES.get(yolo_classes[j], 'other')
                match_stats[cls_name]['fp'] += 1

        for i, matched in enumerate(codetr_matched):
            if not matched:
                has_fn = True
                cls_name = CLASS_NAMES.get(codetr_classes[i], 'other')
                match_stats[cls_name]['fn'] += 1

        # 保存到对应目录
        if has_fp or has_fn:
            all_problem_images.append(img_path.name)
            # 复制图片到 image/
            shutil.copy(img_path, image_dir / img_path.name)
            
            # 保存过滤后的 Co-DETR 标注到 label/ (不包含得分)
            save_yolo_labels_no_score(label_dir / f"{stem}.txt", codetr_boxes, codetr_classes, w, h)
            
            # 生成 Co-DETR 标注可视化 (显示类别和得分)
            label_vis_img = img.copy()
            for i, (box, cls_id) in enumerate(zip(codetr_boxes, codetr_classes)):
                x1, y1, x2, y2 = map(int, box)
                color = COLORS.get(cls_id, (0, 255, 0))
                cv2.rectangle(label_vis_img, (x1, y1), (x2, y2), color, 2)
                # 显示类别和得分
                score = codetr_scores[i] if codetr_scores and i < len(codetr_scores) else 0
                label_text = f"{CLASS_NAMES.get(cls_id, 'obj')} {score:.2f}"
                # 背景框
                (text_w, text_h), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(label_vis_img, (x1, y1 - text_h - 4), (x1 + text_w, y1), color, -1)
                cv2.putText(label_vis_img, label_text, (x1, y1 - 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.imwrite(str(label_vis_dir / img_path.name), label_vis_img)
            
            # 根据类型复制到子目录
            if has_fp and has_fn:
                fp_fn_both_images.append(img_path.name)
                shutil.copy(img_path, fp_fn_both_dir / img_path.name)
            elif has_fp:
                fp_only_images.append(img_path.name)
                shutil.copy(img_path, fp_only_dir / img_path.name)
            elif has_fn:
                fn_only_images.append(img_path.name)
                shutil.copy(img_path, fn_only_dir / img_path.name)

        # 生成对比可视化
        if has_fp or has_fn:
            vis_img = img.copy()
            # 画 CoDETR 框 (实线)
            for box, cls_id in zip(codetr_boxes, codetr_classes):
                x1, y1, x2, y2 = map(int, box)
                color = COLORS.get(cls_id, (0, 255, 0))
                cv2.rectangle(vis_img, (x1, y1), (x2, y2), color, 2)
                cv2.putText(vis_img, f"GT:{CLASS_NAMES.get(cls_id, '')}", (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

            # 画 YOLO 框 (虚线效果用不同颜色)
            for j, (box, cls_id) in enumerate(zip(yolo_boxes, yolo_classes)):
                x1, y1, x2, y2 = map(int, box)
                if not yolo_matched[j]:  # 未匹配的是误检
                    color = (0, 0, 255)  # 红色
                    cv2.rectangle(vis_img, (x1, y1), (x2, y2), color, 1)
                    cv2.putText(vis_img, f"FP:{CLASS_NAMES.get(cls_id, '')}", (x1, y2 + 15),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

            cv2.imwrite(str(compare_vis_dir / img_path.name), vis_img)

    # 输出报告
    print("\n" + "=" * 50)
    print("📊 对比结果报告")
    print("=" * 50)
    print(f"\n📁 输出目录: {output_base}")
    print(f"总图片数: {len(image_files)}")
    print(f"有问题图片总数: {len(all_problem_images)} 张")
    print(f"  ├── 仅误检 (FP only): {len(fp_only_images)} 张")
    print(f"  ├── 仅漏检 (FN only): {len(fn_only_images)} 张")
    print(f"  └── 同时误检+漏检: {len(fp_fn_both_images)} 张")

    print("\n按类别统计:")
    for cls_name, stats in match_stats.items():
        tp, fp, fn = stats['tp'], stats['fp'], stats['fn']
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        print(f"  {cls_name}: TP={tp}, FP={fp}, FN={fn}, Precision={precision:.3f}, Recall={recall:.3f}")

    # 保存 data.txt - 存储所有 image 文件路径
    data_txt_path = output_base / "data.txt"
    with open(data_txt_path, 'w') as f:
        for img_name in all_problem_images:
            f.write(f"image/{img_name}\n")
    
    # 保存各类别列表
    with open(output_base / "fp_only.txt", 'w') as f:
        f.write("\n".join(fp_only_images))
    with open(output_base / "fn_only.txt", 'w') as f:
        f.write("\n".join(fn_only_images))
    with open(output_base / "fp_fn_both.txt", 'w') as f:
        f.write("\n".join(fp_fn_both_images))

    print(f"\n📄 data.txt: {data_txt_path}")
    print(f"✅ 对比完成！")


# ==================== 主函数 ====================

def main():
    parser = argparse.ArgumentParser(description="YOLO 检测 + Co-DETR 对比工具")
    subparsers = parser.add_subparsers(dest="command", help="子命令")

    # detect 子命令
    detect_parser = subparsers.add_parser("detect", help="使用 YOLO 检测")
    detect_parser.add_argument("-i", "--input", required=True, help="输入图片目录")
    detect_parser.add_argument("-m", "--model", default="yolo11n.pt", help="YOLO 模型路径")
    detect_parser.add_argument("-c", "--conf-thres", type=float, default=0.3, help="置信度阈值")
    detect_parser.add_argument("-n", "--nms-thres", type=float, default=0.5, help="NMS IoU 阈值")

    # compare 子命令
    compare_parser = subparsers.add_parser("compare", help="对比 YOLO 和 Co-DETR 结果")
    compare_parser.add_argument("--yolo", "--yolo-labels", dest="yolo_labels", required=True, help="YOLO 标注目录")
    compare_parser.add_argument("--codetr", "--codetr-labels", dest="codetr_labels", required=True, help="Co-DETR 标注目录")
    compare_parser.add_argument("--images", required=True, help="原始图片目录")
    compare_parser.add_argument("--iou-thres", type=float, default=0.5, help="IoU 匹配阈值")
    compare_parser.add_argument("--gt-score-thres", type=float, default=0.5, help="GT 得分阈值 (Co-DETR 检测得分超过此值才视为真实目标)")

    args = parser.parse_args()

    if args.command == "detect":
        run_detection(args)
    elif args.command == "compare":
        run_compare(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
