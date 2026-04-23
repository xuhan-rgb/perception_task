# 检测与对比工作流使用文档

## 概述

本工具用于对比 Co-DETR 和 YOLO 模型的检测结果，找出误检 (FP) 和漏检 (FN)。

## 脚本说明

| 脚本 | 功能 |
|------|------|
| `run_codetr_detect.sh` | 使用 Co-DETR 模型检测 |
| `run_yolo_compare.sh` | YOLO 检测和对比工具 |
| `step2_yolo_detect.sh` | 单独运行 YOLO 检测 |
| `step3_compare.sh` | 单独运行对比 |
| `filt_data.sh` | 完整工作流 |

---

## 配置参数

### 检测参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `CONF_THRES` | 0.3 | 置信度阈值，低于此值的检测被过滤 |
| `NMS_THRES` | 0.5 | NMS IoU 阈值，用于去除重叠框 |

### 对比参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `IoU 阈值` | 0.5 | 匹配时的 IoU 阈值，用于判断两个框是否匹配 |
| `GT 得分阈值` | 0.5 | Co-DETR 检测得分阈值，用于决定哪些检测作为 GT |

---

## GT 得分阈值配置说明

GT 得分阈值（`--gt-score-thres`）决定了 Co-DETR 检测中，哪些检测框被视为"真实目标"。

### 工作原理

```
Co-DETR 检测 + 得分阈值 → Ground Truth (GT)
                ↓
    对比 YOLO 检测结果
                ↓
    ┌───────────────────────────┐
    │ FP: YOLO 有但 GT 没有     │
    │ FN: GT 有但 YOLO 没有     │
    │ TP: 两者都有且匹配        │
    └───────────────────────────┘
```

### 阈值选择建议

| 阈值 | 模式 | 使用场景 |
|------|------|---------|
| 0.3 | 宽松 | 统计所有可能的错误，包括边界情况 |
| **0.5** | **推荐** | 只统计中高置信度的错误，平衡严格和宽松 |
| 0.7 | 严格 | 只统计高置信度的错误，排除模糊情况 |

### 说明

- **高阈值 (如 0.7)**：只有 Co-DETR 非常确定的检测才算 GT，可能低估漏检数量
- **低阈值 (如 0.3)**：Co-DETR 的边界检测也算 GT，可能高估错误数量
- **推荐值 (0.5)**：平衡选择，只有 Co-DETR 较为确定的检测才参与对比

---

## 类别配置

### train_det 模型（自训练）

```python
TRAINDET_TO_YOLO = {
    0: 0,   # person -> person
    1: 1,   # car -> vehicle
}
```

### COCO 预训练模型

```python
COCO_TO_YOLO = {
    0: 0,   # person -> person
    2: 1,   # car -> vehicle
    5: 1,   # bus -> vehicle
    7: 1,   # truck -> vehicle
}
```

默认使用 train_det 模型映射。

---

## 输出目录结构

运行对比后，输出目录格式为 `{codetr_model}_{yolo_model}/`：

```
swin_o365_yolo/
├── image/          # 所有有问题的图片
├── label/          # 对应的 Co-DETR 标注
├── fp_only/        # 仅误检图片
├── fn_only/        # 仅漏检图片
├── fp_fn_both/     # 同时有误检和漏检
├── compare_vis/    # 可视化对比
├── data.txt        # image/ 文件列表
├── fp_only.txt     # 仅误检图片列表
├── fn_only.txt     # 仅漏检图片列表
└── fp_fn_both.txt  # 同时有问题的图片列表
```

---

## 使用示例

### 完整工作流

```bash
# 运行全部步骤
./filt_data.sh all

# 或分步运行
./filt_data.sh codetr   # 步骤1: Co-DETR 检测
./filt_data.sh yolo     # 步骤2: YOLO 检测
./filt_data.sh compare  # 步骤3: 对比筛选
```

### 自定义参数

```bash
# Co-DETR 检测
./run_codetr_detect.sh swin_o365 <输入目录> 0.3 0.5

# YOLO 检测
./run_yolo_compare.sh detect <输入目录> 0.3 <模型路径> 0.5

# 对比（带 GT 得分阈值）
./run_yolo_compare.sh compare <yolo标注> <codetr标注> <图片目录> 0.5 0.5
```

---

## 可视化颜色说明

| 颜色 | 含义 |
|------|------|
| 🟢 绿色框 | GT:person - Co-DETR 检测到的行人 |
| 🔵 蓝色框 | GT:vehicle - Co-DETR 检测到的车辆 |
| 🔴 红色框 | FP - 误检（YOLO 检测到但 Co-DETR 没有）|
