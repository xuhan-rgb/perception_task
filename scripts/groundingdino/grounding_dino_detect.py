#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
GroundingDINO 检测封装脚本
输入：图片路径或目录、提示词、阈值
输出：bbox (xyxy格式，像素坐标)、得分、可视化图片、标签文件
"""

import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

# 添加项目路径
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

import groundingdino.datasets.transforms as T
from groundingdino.models import build_model
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap


# 默认配置路径
DEFAULT_CONFIG = SCRIPT_DIR / "groundingdino/config/GroundingDINO_SwinT_OGC.py"
DEFAULT_WEIGHTS = SCRIPT_DIR / "weights/groundingdino_swint_ogc.pth"
DEFAULT_VIS_DIR = SCRIPT_DIR / "vis"
DEFAULT_LABEL_DIR = SCRIPT_DIR / "label"

# 支持的图片格式
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}

# 可视化颜色（BGR格式）
COLORS = [
    (0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0),
    (255, 0, 255), (0, 255, 255), (128, 0, 255), (255, 128, 0),
]


class GroundingDINODetector:
    """GroundingDINO 检测器封装类"""

    def __init__(self, config_path=None, weights_path=None, device="cuda"):
        """
        初始化检测器

        Args:
            config_path: 配置文件路径，默认使用 SwinT 配置
            weights_path: 权重文件路径
            device: 运行设备 ("cuda" 或 "cpu")
        """
        self.config_path = config_path or str(DEFAULT_CONFIG)
        self.weights_path = weights_path or str(DEFAULT_WEIGHTS)
        self.device = device
        self.model = self._load_model()

        # 图像预处理
        self.transform = T.Compose([
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    def _load_model(self):
        """加载模型"""
        args = SLConfig.fromfile(self.config_path)
        args.device = self.device
        model = build_model(args)
        checkpoint = torch.load(self.weights_path, map_location="cpu")
        model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
        model.eval()
        model.to(self.device)
        return model

    def _preprocess_caption(self, caption):
        """预处理提示词"""
        result = caption.lower().strip()
        if not result.endswith("."):
            result = result + "."
        return result

    def detect(self, image_path, prompt, box_threshold=0.35, text_threshold=0.25):
        """
        执行检测

        Args:
            image_path: 图片路径
            prompt: 提示词，如 "cat . dog ." 或 "animal"
            box_threshold: 框置信度阈值
            text_threshold: 文本置信度阈值

        Returns:
            dict: {
                "boxes": np.ndarray,  # (N, 4) xyxy 格式，像素坐标
                "scores": np.ndarray, # (N,) 置信度得分
                "phrases": list,      # 检测到的短语列表
                "size": tuple,        # (width, height) 图片尺寸
            }
        """
        # 加载图片
        image_pil = Image.open(image_path).convert("RGB")
        width, height = image_pil.size

        # 预处理
        image_tensor, _ = self.transform(image_pil, None)
        image_tensor = image_tensor.to(self.device)

        caption = self._preprocess_caption(prompt)

        # 推理
        with torch.no_grad():
            outputs = self.model(image_tensor[None], captions=[caption])

        # 后处理
        pred_logits = outputs["pred_logits"].cpu().sigmoid()[0]  # (nq, 256)
        pred_boxes = outputs["pred_boxes"].cpu()[0]  # (nq, 4) cxcywh normalized

        # 过滤低置信度框
        mask = pred_logits.max(dim=1)[0] > box_threshold
        logits = pred_logits[mask]
        boxes = pred_boxes[mask]

        # 提取短语
        tokenizer = self.model.tokenizer
        tokenized = tokenizer(caption)

        phrases = []
        for logit in logits:
            phrase = get_phrases_from_posmap(
                logit > text_threshold, tokenized, tokenizer
            ).replace('.', '')
            phrases.append(phrase)

        # 转换框格式：cxcywh normalized -> xyxy pixels
        scores = logits.max(dim=1)[0].numpy()
        boxes = boxes.numpy()

        # cxcywh -> xyxy
        boxes_xyxy = np.zeros_like(boxes)
        boxes_xyxy[:, 0] = (boxes[:, 0] - boxes[:, 2] / 2) * width   # x1
        boxes_xyxy[:, 1] = (boxes[:, 1] - boxes[:, 3] / 2) * height  # y1
        boxes_xyxy[:, 2] = (boxes[:, 0] + boxes[:, 2] / 2) * width   # x2
        boxes_xyxy[:, 3] = (boxes[:, 1] + boxes[:, 3] / 2) * height  # y2

        return {
            "boxes": boxes_xyxy,
            "scores": scores,
            "phrases": phrases,
            "size": (width, height),
        }

    @staticmethod
    def visualize(image_path, results, output_path):
        """
        可视化检测结果并保存

        Args:
            image_path: 原始图片路径
            results: detect() 返回的结果字典
            output_path: 输出图片路径
        """
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"无法读取图片: {image_path}")

        boxes = results["boxes"]
        scores = results["scores"]
        phrases = results["phrases"]

        for i, (box, score, phrase) in enumerate(zip(boxes, scores, phrases)):
            color = COLORS[i % len(COLORS)]
            x1, y1, x2, y2 = map(int, box)

            # 绘制边框
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

            # 绘制标签背景
            label = f"{phrase} {score:.2f}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            cv2.rectangle(image, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)

            # 绘制标签文字
            cv2.putText(image, label, (x1 + 2, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        # 确保输出目录存在
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        cv2.imwrite(str(output_path), image)
        return str(output_path)

    @staticmethod
    def save_labels(results, output_path):
        """
        保存检测结果到标签文件（xyxy 像素坐标格式）

        格式: phrase score x1 y1 x2 y2

        Args:
            results: detect() 返回的结果字典
            output_path: 输出标签文件路径
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        boxes = results["boxes"]
        scores = results["scores"]
        phrases = results["phrases"]

        with open(output_path, 'w') as f:
            for box, score, phrase in zip(boxes, scores, phrases):
                x1, y1, x2, y2 = box
                f.write(f"{phrase} {score:.4f} {x1:.1f} {y1:.1f} {x2:.1f} {y2:.1f}\n")

        return str(output_path)


def get_image_files(input_path, recursive=True):
    """
    获取输入路径下的所有图片文件

    Args:
        input_path: 输入路径（文件或目录）
        recursive: 是否递归搜索子目录

    Returns:
        list: 图片文件路径列表
    """
    input_path = Path(input_path)

    if input_path.is_file():
        if input_path.suffix.lower() in IMAGE_EXTENSIONS:
            return [input_path]
        else:
            raise ValueError(f"不支持的图片格式: {input_path.suffix}")

    elif input_path.is_dir():
        image_files = []
        pattern = "**/*" if recursive else "*"
        for ext in IMAGE_EXTENSIONS:
            image_files.extend(input_path.glob(f"{pattern}{ext}"))
            image_files.extend(input_path.glob(f"{pattern}{ext.upper()}"))
        return sorted(image_files)

    else:
        raise ValueError(f"路径不存在: {input_path}")


def get_relative_path(image_path, input_root):
    """计算图片相对于输入根目录的相对路径（保持目录结构）"""
    image_path = Path(image_path)
    input_root = Path(input_root)

    if input_root.is_file():
        # 单文件输入，直接返回文件名
        return Path(image_path.name)

    try:
        return image_path.relative_to(input_root)
    except ValueError:
        return Path(image_path.name)


def main():
    parser = argparse.ArgumentParser(description="GroundingDINO 目标检测")
    parser.add_argument("-i", "--input", required=True, help="图片路径或目录（支持递归）")
    parser.add_argument("-t", "--text", required=True, help="提示词，如 'cat . dog .' 或 'animal'")
    parser.add_argument("--box_threshold", type=float, default=0.35, help="框置信度阈值 (默认: 0.35)")
    parser.add_argument("--text_threshold", type=float, default=0.25, help="文本置信度阈值 (默认: 0.25)")
    parser.add_argument("--device", default="cuda", help="运行设备 (默认: cuda)")
    parser.add_argument("-c", "--config", default=None, help="配置文件路径")
    parser.add_argument("-p", "--weights", default=None, help="权重文件路径")
    parser.add_argument("-o", "--vis_dir", default=None, help="可视化输出目录 (默认: vis)")
    parser.add_argument("-l", "--label_dir", default=None, help="标签输出目录 (默认: label)")
    parser.add_argument("--no_vis", action="store_true", help="不保存可视化结果")
    parser.add_argument("--no_label", action="store_true", help="不保存标签文件")
    parser.add_argument("--no_recursive", action="store_true", help="不递归搜索子目录")
    args = parser.parse_args()

    input_path = Path(args.input)

    # 获取图片列表
    image_files = get_image_files(args.input, recursive=not args.no_recursive)
    print(f"找到 {len(image_files)} 张图片")

    if len(image_files) == 0:
        print("没有找到图片文件")
        return

    # 初始化检测器
    detector = GroundingDINODetector(
        config_path=args.config,
        weights_path=args.weights,
        device=args.device
    )

    # 设置输出目录
    vis_dir = Path(args.vis_dir) if args.vis_dir else DEFAULT_VIS_DIR
    label_dir = Path(args.label_dir) if args.label_dir else DEFAULT_LABEL_DIR

    # 批量处理
    all_results = {}
    for image_path in tqdm(image_files, desc="检测中"):
        # 执行检测
        results = detector.detect(
            image_path=image_path,
            prompt=args.text,
            box_threshold=args.box_threshold,
            text_threshold=args.text_threshold
        )

        # 计算相对路径，保持目录结构
        rel_path = get_relative_path(image_path, input_path)
        rel_dir = rel_path.parent  # 相对目录，如 v11C1/
        image_name = rel_path.stem  # 文件名（无扩展名）

        all_results[str(image_path)] = results

        # 保存可视化（保持目录结构）
        if not args.no_vis:
            vis_path = vis_dir / rel_dir / (image_name + "_det" + image_path.suffix)
            GroundingDINODetector.visualize(image_path, results, vis_path)

        # 保存标签（保持目录结构）
        if not args.no_label:
            label_path = label_dir / rel_dir / (image_name + ".txt")
            GroundingDINODetector.save_labels(results, label_path)

    # 打印结果摘要
    print(f"\n{'='*60}")
    print(f"处理完成: {len(image_files)} 张图片")
    print(f"{'='*60}")

    total_objects = sum(len(r["scores"]) for r in all_results.values())
    print(f"总检测目标数: {total_objects}")

    if not args.no_vis:
        print(f"可视化保存目录: {vis_dir}")
    if not args.no_label:
        print(f"标签保存目录: {label_dir}")

    # 单张图片时打印详细结果
    if len(image_files) == 1:
        results = list(all_results.values())[0]
        print(f"\n检测结果: 共 {len(results['scores'])} 个目标")
        for i, (box, score, phrase) in enumerate(zip(
            results["boxes"], results["scores"], results["phrases"]
        )):
            print(f"[{i+1}] {phrase}")
            print(f"    bbox (xyxy): [{box[0]:.1f}, {box[1]:.1f}, {box[2]:.1f}, {box[3]:.1f}]")
            print(f"    score: {score:.4f}")

    return all_results


if __name__ == "__main__":
    main()
