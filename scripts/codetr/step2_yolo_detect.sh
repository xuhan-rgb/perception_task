#!/bin/bash
# 步骤2: YOLO 检测
# 使用训练好的 YOLO 模型检测行人和车辆

set -e

# 参数配置
CONF_THRES="${1:-0.3}"
NMS_THRES="${2:-0.5}"
INPUT_DIR="${3:-../chandao_data/results/crops}"
YOLO_MODEL="${4:-/mnt/data/ultralytics/runs/detect/train_det/weights/best.pt}"

echo "===== YOLO 检测 ====="
echo "置信度: $CONF_THRES"
echo "NMS: $NMS_THRES"
echo "输入: $INPUT_DIR"
echo "模型: $YOLO_MODEL"
echo ""

./run_yolo_compare.sh detect "$INPUT_DIR" "$CONF_THRES" "$YOLO_MODEL" "$NMS_THRES"
