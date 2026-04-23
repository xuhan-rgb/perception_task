

# 检测模块
## codetr模块
```
# 测试脚本
```

```shell
#环境准备
export CUDA_HOME=/usr/local/cuda-11.3

cd /mnt/data/perception_task/detection/GroundingDINO
pip install -e .

#demo测试
CUDA_VISIBLE_DEVICES=0 python demo/inference_on_a_image.py \
-c groundingdino/config/GroundingDINO_SwinT_OGC.py \
-p ./weights/groundingdino_swint_ogc.pth \
-i .asset/cat_dog.jpeg \
-o logs/1111 \
-t "There is a cat and a dog in the image ." \
--token_spans "[[[9, 10], [11, 14]], [[19, 20], [21, 24]]]"
 [--cpu-only] # open it for cpu mode

# /mnt/data/keypoint_data/mvlift_release/data/AP-36k-patr1/3cat/v10c1/frame1.jpg
CUDA_VISIBLE_DEVICES=0 python demo/inference_on_a_image.py \
-c groundingdino/config/GroundingDINO_SwinT_OGC.py \
-p ./weights/groundingdino_swint_ogc.pth \
-i /mnt/data/keypoint_data/mvlift_release/data/AP-36k-patr1/10gorilla/v11C1/frame1.jpg \
-o logs/1111 \
-t "animal."

# ********************* 封装了一下**********************
# 使用这个得到结果
CUDA_VISIBLE_DEVICES=0 python grounding_dino_detect.py \
    -i /mnt/data/keypoint_data/mvlift_release/data/AP-36k-patr1/ \
    -t "animal." \
    --box_threshold 0.35 \
    --text_threshold 0.25 \
    -l /mnt/data/keypoint_data/mvlift_release/data/AP-36k-patr1-label/ \
    -o output   # 保存可视化结果
```