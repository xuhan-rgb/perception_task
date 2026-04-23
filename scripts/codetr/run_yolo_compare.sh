#!/bin/bash
# YOLO 检测 + Co-DETR 对比工具
# 使用方法:
#   ./run_yolo_compare.sh detect <输入目录> [置信度]
#   ./run_yolo_compare.sh compare <yolo标注> <codetr标注> <图片目录> [IoU阈值]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT="$SCRIPT_DIR/yolo_detect_compare.py"

# 激活环境 (同时支持 ultralytics)
source ~/anaconda3/bin/activate sam3 2>/dev/null || source ~/anaconda3/bin/activate base

COMMAND="${1:-help}"

case "$COMMAND" in
    "detect")
        INPUT_DIR="${2:-}"
        CONF_THRES="${3:-0.3}"
        MODEL="${4:-yolo11n.pt}"
        NMS_THRES="${5:-0.5}"
        
        if [ -z "$INPUT_DIR" ]; then
            echo "用法: ./run_yolo_compare.sh detect <输入目录> [置信度] [模型路径] [NMS阈值]"
            exit 1
        fi
        
        echo "🚀 YOLO 检测模式"
        python "$SCRIPT" detect -i "$INPUT_DIR" -c "$CONF_THRES" -m "$MODEL" -n "$NMS_THRES"
        ;;
        
    "compare")
        YOLO_LABELS="${2:-}"
        CODETR_LABELS="${3:-}"
        IMAGES="${4:-}"
        IOU_THRES="${5:-0.5}"
        GT_SCORE_THRES="${6:-0.5}"
        
        if [ -z "$YOLO_LABELS" ] || [ -z "$CODETR_LABELS" ] || [ -z "$IMAGES" ]; then
            echo "用法: ./run_yolo_compare.sh compare <yolo标注> <codetr标注> <图片目录> [IoU阈值] [GT得分阈值]"
            exit 1
        fi
        
        echo "📊 对比模式"
        python "$SCRIPT" compare --yolo "$YOLO_LABELS" --codetr "$CODETR_LABELS" --images "$IMAGES" --iou-thres "$IOU_THRES" --gt-score-thres "$GT_SCORE_THRES"
        ;;
        
    *)
        echo "YOLO 检测 + Co-DETR 对比工具"
        echo ""
        echo "用法:"
        echo "  ./run_yolo_compare.sh detect <输入目录> [置信度] [模型路径]"
        echo "  ./run_yolo_compare.sh compare <yolo标注> <codetr标注> <图片目录> [IoU阈值]"
        echo ""
        echo "示例:"
        echo "  # 1. 使用 YOLO 检测"
        echo "  ./run_yolo_compare.sh detect ../chandao_data/results/crops 0.3"
        echo ""
        echo "  # 2. 与 Co-DETR 结果对比"
        echo "  ./run_yolo_compare.sh compare ../chandao_data/results/labels_yolo ../chandao_data/results/labels_swin_o365 ../chandao_data/results/crops"
        ;;
esac
