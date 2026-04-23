#!/bin/bash
# 步骤2: YOLO 检测
# 使用训练好的 YOLO 模型检测行人和车辆

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 参数配置
CONF_THRES="${1:-0.3}"
NMS_THRES="${2:-0.5}"
INPUT_DIR="${3:-/mnt/data/04-DevTools/chandao_data/results/crops}"
YOLO_MODEL="${4:-/mnt/data/03-ML-Env/ultralytics/runs/ped_detect/train7/weights/best.pt}"

echo "===== YOLO 检测 ====="
echo "置信度: $CONF_THRES"
echo "NMS: $NMS_THRES"
echo "输入: $INPUT_DIR"
echo "模型: $YOLO_MODEL"
echo ""

"$SCRIPT_DIR/run_yolo_compare.sh" detect "$INPUT_DIR" "$CONF_THRES" "$YOLO_MODEL" "$NMS_THRES"
