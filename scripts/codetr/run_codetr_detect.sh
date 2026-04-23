#!/bin/bash
# Co-DETR 检测脚本
# 使用方法: ./run_codetr_detect.sh [模型] [输入目录] [置信度] [NMS阈值] [类别]
# 模型: r50_1x, r50_3x, r50_9enc_1x, r50_9enc_3x, swin_1x, swin_2x, swin_3x, swin_o365
# 类别: 支持名称或索引，逗号分隔。如: person / 0 / person,car / 0,2,5,7
# 示例:
#   ./run_codetr_detect.sh swin_3x /path/to/images 0.3 0.5           # 检测所有默认类别
#   ./run_codetr_detect.sh swin_3x /path/to/images 0.3 0.5 person    # 只检测人
#   ./run_codetr_detect.sh swin_3x /path/to/images 0.3 0.5 person,car,bus  # 检测人和车辆

set -e

# 默认参数
MODEL_NAME="${1:-r50_9enc_1x}"
INPUT_DIR="${2:-/mnt/data/chandao_data/results/crops}"
CONF_THRES="${3:-0.3}"
NMS_THRES="${4:-0.5}"
CLASSES="${5:-}"  # 可选：指定检测类别，如 "person" 或 "0,2,car,bus"

# 路径配置
CODETR_ROOT="/mnt/data/Co-DETR"
CKPT_BASE="$CODETR_ROOT/checkpoints/co_dino"
CONFIG_BASE="$CODETR_ROOT/projects/configs/co_dino"
SCRIPT="/mnt/data/dev-scripts/tools/dino_detect_eval.py"

# 激活环境
source ~/anaconda3/bin/activate codetr

# 模型配置映射
case "$MODEL_NAME" in
    # ResNet-50 系列
    "r50_1x")
        CONFIG="$CONFIG_BASE/co_dino_5scale_r50_1x_coco.py"
        CKPT="$CKPT_BASE/swin_large/co_dino/resnet_detr/co_dino_5scale_r50_1x_coco.pth"
        echo "📦 使用模型: Co-DINO ResNet-50 1x (AP: ~52.1)"
        ;;
    "r50_3x")
        CONFIG="$CONFIG_BASE/co_dino_5scale_lsj_r50_3x_coco.py"
        CKPT="$CKPT_BASE/swin_large/co_dino/resnet_lsj/co_dino_5scale_lsj_r50_3x_coco.pth"
        echo "📦 使用模型: Co-DINO ResNet-50 LSJ 3x (AP: ~52.1)"
        ;;
    "r50_9enc_1x")
        CONFIG="$CONFIG_BASE/co_dino_5scale_9encoder_lsj_r50_1x_coco.py"
        CKPT="$CKPT_BASE/swin_large/co_dino/resnet_lsj/co_dino_5scale_9encoder_lsj_r50_1x_coco.pth"
        echo "📦 使用模型: Co-DINO ResNet-50 9encoder 1x (AP: ~52.5)"
        ;;
    "r50_9enc_3x")
        CONFIG="$CONFIG_BASE/co_dino_5scale_9encoder_lsj_r50_3x_coco.py"
        CKPT="$CKPT_BASE/swin_large/co_dino/resnet_lsj/co_dino_5scale_9encoder_lsj_r50_3x_coco.pth"
        echo "📦 使用模型: Co-DINO ResNet-50 9encoder 3x (AP: ~52.9)"
        ;;
    # Swin-Large 系列
    "swin_1x")
        CONFIG="$CONFIG_BASE/co_dino_5scale_lsj_swin_large_1x_coco.py"
        CKPT="$CKPT_BASE/swin_large/co_dino/swin_lsj/co_dino_5scale_lsj_swin_large_1x_coco.pth"
        echo "📦 使用模型: Co-DINO Swin-Large 1x (AP: ~58.9)"
        ;;
    "swin_2x")
        CONFIG="$CONFIG_BASE/co_dino_5scale_lsj_swin_large_2x_coco.py"
        CKPT="$CKPT_BASE/swin_large/co_dino/swin_lsj/co_dino_5scale_lsj_swin_large_2x_coco.pth"
        echo "📦 使用模型: Co-DINO Swin-Large 2x (AP: ~60.0)"
        ;;
    "swin_3x")
        CONFIG="$CONFIG_BASE/co_dino_5scale_lsj_swin_large_3x_coco.py"
        CKPT="$CKPT_BASE/swin_large/co_dino/swin_lsj/co_dino_5scale_lsj_swin_large_3x_coco.pth"
        echo "📦 使用模型: Co-DINO Swin-Large 3x (AP: ~64.1) ⭐ 推荐"
        ;;
    "swin_o365")
        CONFIG="$CONFIG_BASE/co_dino_5scale_swin_large_16e_o365tococo.py"
        CKPT="$CKPT_BASE/swin_large/co_dino/swin_detr/co_dino_5scale_swin_large_16e_o365tococo.pth"
        echo "📦 使用模型: Co-DINO Swin-Large O365→COCO (AP: ~66.0) ⭐⭐ 最佳"
        ;;
    *)
        echo "❌ 未知模型: $MODEL_NAME"
        echo ""
        echo "可用模型:"
        echo "  ResNet-50 系列 (较快):"
        echo "    r50_1x      - AP: ~52.1"
        echo "    r50_3x      - AP: ~52.1"
        echo "    r50_9enc_1x - AP: ~52.5"
        echo "    r50_9enc_3x - AP: ~52.9"
        echo ""
        echo "  Swin-Large 系列 (更准):"
        echo "    swin_1x     - AP: ~58.9"
        echo "    swin_2x     - AP: ~60.0"
        echo "    swin_3x     - AP: ~64.1 ⭐ 推荐"
        echo "    swin_o365   - AP: ~66.0 ⭐⭐ 最佳"
        exit 1
        ;;
esac

# 检查文件是否存在
if [ ! -f "$CKPT" ]; then
    echo "❌ 模型文件不存在: $CKPT"
    echo "请确保模型已下载完成"
    exit 1
fi

# 运行检测
echo "📂 输入目录: $INPUT_DIR"
echo "📊 置信度阈值: $CONF_THRES"
echo "📊 NMS 阈值: $NMS_THRES"
[ -n "$CLASSES" ] && echo "🎯 目标类别: $CLASSES"
echo ""

# 切换到 Co-DETR 目录，避免其他项目的 mmcv 干扰
cd "$CODETR_ROOT"

# 清空 PYTHONPATH 并设置为 Co-DETR，确保使用正确的 mmdet
export PYTHONPATH="$CODETR_ROOT"

# 构建命令
CMD="python $SCRIPT detect \
    -i \"$INPUT_DIR\" \
    --conf-thres $CONF_THRES \
    --config \"$CONFIG\" \
    --checkpoint \"$CKPT\" \
    --output-suffix \"$MODEL_NAME\" \
    --nms-thres $NMS_THRES"

# 添加类别参数（如果指定）
[ -n "$CLASSES" ] && CMD="$CMD --classes \"$CLASSES\""

eval $CMD
