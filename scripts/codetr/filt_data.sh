#!/bin/bash
# 数据筛选工作流
# 步骤1: 运行 Co-DETR 检测
# 步骤2: 运行 YOLO 检测  
# 步骤3: 对比结果，筛选误检/漏检

# cd /mnt/data/dev-scripts
# ./scripts/filter_detection_data.sh ./input ./output --skip 5 --clean

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ========== 参数配置（环境变量可覆盖）==========
# 检测参数
CONF_THRES="${CONF_THRES:-0.3}"     # 置信度阈值
NMS_THRES="${NMS_THRES:-0.5}"       # NMS IoU 阈值

# 对比参数
IOU_THRES="${IOU_THRES:-0.5}"              # 匹配 IoU 阈值
GT_SCORE_THRES="${GT_SCORE_THRES:-0.5}"    # GT 得分阈值

# 路径配置
INPUT_DIR="${INPUT_DIR:-/mnt/data/04-DevTools/chandao_data/results/crops}"
YOLO_MODEL="${YOLO_MODEL:-/mnt/data/03-ML-Env/ultralytics/runs/ped_detect/train7/weights/best.pt}"
CODETR_MODEL="${CODETR_MODEL:-swin_o365}"

# ========== 步骤选择 ==========
STEP="${1:-all}"

case "$STEP" in
    "1"|"codetr")
        echo "===== 步骤1: Co-DETR 检测 ====="
        "$SCRIPT_DIR/run_codetr_detect.sh" $CODETR_MODEL $INPUT_DIR $CONF_THRES $NMS_THRES
        ;;
    
    "2"|"yolo")
        echo "===== 步骤2: YOLO 检测 ====="
        "$SCRIPT_DIR/step2_yolo_detect.sh" $CONF_THRES $NMS_THRES $INPUT_DIR $YOLO_MODEL
        ;;
    
    "3"|"compare")
        echo "===== 步骤3: 对比筛选 ====="
        "$SCRIPT_DIR/step3_compare.sh" $INPUT_DIR $CODETR_MODEL $IOU_THRES $GT_SCORE_THRES
        ;;
    
    "all")
        echo "===== 运行全部步骤 ====="
        echo ""
        echo "步骤1: Co-DETR 检测"
        "$SCRIPT_DIR/run_codetr_detect.sh" $CODETR_MODEL $INPUT_DIR $CONF_THRES $NMS_THRES
        echo ""
        echo "步骤2: YOLO 检测"
        "$SCRIPT_DIR/step2_yolo_detect.sh" $CONF_THRES $NMS_THRES $INPUT_DIR $YOLO_MODEL
        echo ""
        echo "步骤3: 对比筛选"
        "$SCRIPT_DIR/step3_compare.sh" $INPUT_DIR $CODETR_MODEL $IOU_THRES $GT_SCORE_THRES
        ;;
    
    *)
        echo "数据筛选工作流"
        echo ""
        echo "用法: ./filt_data.sh [步骤]"
        echo ""
        echo "步骤:"
        echo "  1, codetr  - 仅运行 Co-DETR 检测"
        echo "  2, yolo    - 仅运行 YOLO 检测"
        echo "  3, compare - 仅对比筛选 (需先完成1和2)"
        echo "  all        - 运行全部步骤 (默认)"
        echo ""
        echo "示例:"
        echo "  ./filt_data.sh 1       # 只运行 Co-DETR"
        echo "  ./filt_data.sh yolo    # 只运行 YOLO"
        echo "  ./filt_data.sh compare # 只对比"
        echo "  ./filt_data.sh all     # 全部"
        ;;
esac
