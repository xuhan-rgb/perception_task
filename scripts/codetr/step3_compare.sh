#!/bin/bash
# 步骤3: 对比筛选
# 对比 YOLO 和 Co-DETR 结果，找出误检和漏检

set -e

# 参数配置
INPUT_DIR="${1:-../chandao_data/results/crops}"
CODETR_MODEL="${2:-swin_o365}"
IOU_THRES="${3:-0.5}"
GT_SCORE_THRES="${4:-0.5}"

# 自动推断标注目录
YOLO_LABELS="${INPUT_DIR%/*}/labels_yolo"
CODETR_LABELS="${INPUT_DIR%/*}/labels_${CODETR_MODEL}"

echo "===== 对比筛选 ====="
echo "YOLO 标注: $YOLO_LABELS"
echo "Co-DETR 标注: $CODETR_LABELS"
echo "图片目录: $INPUT_DIR"
echo "IoU 阈值: $IOU_THRES"
echo "GT 得分阈值: $GT_SCORE_THRES"
echo ""

# 检查标注目录是否存在
if [ ! -d "$YOLO_LABELS" ]; then
    echo "❌ YOLO 标注目录不存在: $YOLO_LABELS"
    echo "请先运行 step2_yolo_detect.sh"
    exit 1
fi

if [ ! -d "$CODETR_LABELS" ]; then
    echo "❌ Co-DETR 标注目录不存在: $CODETR_LABELS"
    echo "请先运行 step1_codetr_detect.sh 或 run_codetr_detect.sh"
    exit 1
fi

./run_yolo_compare.sh compare "$YOLO_LABELS" "$CODETR_LABELS" "$INPUT_DIR" "$IOU_THRES" "$GT_SCORE_THRES"

