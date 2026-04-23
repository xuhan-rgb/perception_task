# perception_task

本工程解决**数据集自动标注**的问题：在无标注或粗标注的图片数据（如 AP-36k 动物姿态数据集）上，用 Co-DETR 作为高精度 pseudo-GT 生成器，筛选下游 YOLO 模型的误检/漏检样本，为 finetune 提供有针对性的训练数据。

- **Co-DETR** 作高精度 pseudo-GT 来源（Swin-L backbone，COCO AP ~66.0）
- 通过 Co-DETR ↔ YOLO 的 FP/FN 对比工作流，筛出 YOLO 的误检/漏检样本

输入：图片目录；输出：YOLO 格式标注文件 + 可视化对比图 + 按问题类型分桶的图片列表（`fp_only/`、`fn_only/`、`fp_fn_both/`）。

## 目录结构

```
perception_task/
├── README.md
├── .gitignore
├── scripts/
│   └── codetr/                     # Co-DETR 相关调度脚本（依赖上游 Co-DETR 仓库）
│       ├── run_codetr_detect.sh    # 单独运行 Co-DETR 检测
│       ├── run_yolo_compare.sh     # YOLO 检测 / 与 Co-DETR 对比
│       ├── yolo_detect_compare.py  # 对比逻辑核心实现
│       ├── filt_data.sh            # 完整工作流（Co-DETR → YOLO → 对比）
│       ├── step2_yolo_detect.sh    # 分步脚本
│       └── step3_compare.sh
└── docs/
    ├── detect_compare.md           # 检测对比工作流详细说明
    └── codetr_classes.md           # Co-DETR swin_o365 类别表
```

## 依赖的上游工程

本工程**不**包含 Co-DETR 源码，请自行 clone：

| 上游 | 用途 | 版本 |
|---|---|---|
| [Sense-X/Co-DETR](https://github.com/Sense-X/Co-DETR) | 高精度检测（作为 pseudo-GT） | commit `2665352` |

建议 clone 到 `detection/` 下（已在 `.gitignore` 中忽略）：

```bash
mkdir -p detection && cd detection
git clone https://github.com/Sense-X/Co-DETR.git
cd ..
```

## 外部工具依赖

本工程调用了一个独立工具 `dino_detect_eval.py`，默认路径：

```
/mnt/data/04-DevTools/dev-scripts/tools/detection/dino_detect_eval.py
```

如果你放在其他位置，请通过环境变量覆盖：

```bash
export DINO_DETECT_EVAL=/your/path/to/dino_detect_eval.py
```

## 快速开始

详细说明见 [`docs/detect_compare.md`](docs/detect_compare.md)。

```bash
# 单独调用 Co-DETR 检测
./scripts/codetr/run_codetr_detect.sh swin_o365 <输入目录> 0.3 0.5

# 或者跑完整三步流程（Co-DETR 检测 → YOLO 检测 → 对比筛选）
./scripts/codetr/filt_data.sh all
```

脚本通过 `SCRIPT_DIR` 自动定位相关路径，从任意目录调用都能正常运行。

## 环境变量（可选覆盖默认值）

| 变量 | 默认值 | 说明 |
|---|---|---|
| `CODETR_ROOT` | `./detection/Co-DETR`（自动推断） | Co-DETR 仓库位置 |
| `DINO_DETECT_EVAL` | 见上 | 外部检测评估工具位置 |
| `YOLO_MODEL` | `/mnt/data/03-ML-Env/ultralytics/runs/ped_detect/train7/weights/best.pt` | YOLO 权重，需要时切到更新版本 |
| `INPUT_DIR` | `/mnt/data/04-DevTools/chandao_data/results/crops` | 待检测图片目录 |
| `CONF_THRES` / `NMS_THRES` | 0.3 / 0.5 | 检测阈值 |
| `IOU_THRES` / `GT_SCORE_THRES` | 0.5 / 0.5 | 对比阈值 |

## 数据与权重

以下内容**不**进仓库（见 `.gitignore`）：

- 模型权重（`*.pt`、`*.pth`、`checkpoints/`）
- 数据集与标注（`label/`、`label_out/`、`output/`）
- 推理可视化（`vis/`、`logs/`）

Co-DETR 权重下载请参考上游仓库 README。
