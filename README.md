# perception_task

<!-- TODO: 请你填写 5–8 行的项目介绍。下面是建议回答的要点（直接写成段落即可，删掉这段 TODO）：
  - 这个工程解决什么任务？（例如：在自采数据上筛选 YOLO 的误检/漏检，为下游 finetune 提供高质量样本）
  - 为什么选 Co-DETR + GroundingDINO 这两个开源框架？
  - 输入数据大概长什么样？输出是什么？
-->

## 目录结构

```
perception_task/
├── README.md
├── .gitignore
├── scripts/
│   ├── codetr/                 # Co-DETR 相关调度脚本（依赖上游 Co-DETR 仓库）
│   │   ├── run_codetr_detect.sh    # 单独运行 Co-DETR 检测
│   │   ├── run_yolo_compare.sh     # YOLO 检测 / 与 Co-DETR 对比
│   │   ├── yolo_detect_compare.py  # 对比逻辑核心实现
│   │   ├── filt_data.sh            # 完整工作流（Co-DETR → YOLO → 对比）
│   │   ├── step2_yolo_detect.sh    # 分步脚本
│   │   └── step3_compare.sh
│   └── groundingdino/
│       └── grounding_dino_detect.py  # GroundingDINO 批量检测封装
└── docs/
    ├── detect_compare.md       # 检测对比工作流详细说明
    └── codetr_classes.md       # Co-DETR swin_o365 类别表
```

## 依赖的上游工程

本工程**不**包含两个上游开源仓库的源码，请按需自行 clone：

| 上游 | 用途 | 版本 |
|---|---|---|
| [Sense-X/Co-DETR](https://github.com/Sense-X/Co-DETR) | 高精度检测（作为 pseudo-GT） | commit `2665352` |
| [IDEA-Research/GroundingDINO](https://github.com/IDEA-Research/GroundingDINO) | 文本提示开放集检测 | commit `856dde2` |

建议 clone 到 `detection/` 下（已在 `.gitignore` 中忽略）：

```bash
mkdir -p detection && cd detection
git clone https://github.com/Sense-X/Co-DETR.git
git clone https://github.com/IDEA-Research/GroundingDINO.git
cd ..
```

## 快速开始

### Co-DETR 检测对比工作流

详细说明见 [`docs/detect_compare.md`](docs/detect_compare.md)。

```bash
# 单独调用 Co-DETR 检测
./scripts/codetr/run_codetr_detect.sh swin_o365 <输入目录> 0.3 0.5

# 或者跑完整三步流程
./scripts/codetr/filt_data.sh all
```

> ⚠️ **路径须修改**：`run_codetr_detect.sh` 顶部的 `CODETR_ROOT`、`SCRIPT` 目前是作者机器的绝对路径，使用前请改成你本地路径。

### GroundingDINO 批量推理

```bash
# 在你 clone 下来的 GroundingDINO 目录里运行
cd detection/GroundingDINO
pip install -e .

# 调用封装好的批量检测脚本
python ../../scripts/groundingdino/grounding_dino_detect.py \
    -i <图片目录> \
    -t "animal." \
    --box_threshold 0.35 \
    --text_threshold 0.25 \
    -l <标注输出目录> \
    -o <可视化输出目录>
```

## 数据与权重

以下内容**不**进仓库（见 `.gitignore`）：

- 模型权重（`*.pt`、`*.pth`、`checkpoints/`、`weights/`）
- 数据集与标注（`label/`、`label_out/`、`output/`）
- 推理可视化（`vis/`、`logs/`）

请自行从各模型官方仓库下载对应权重。
