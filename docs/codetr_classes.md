# Co-DETR `swin_o365` 模型类别说明

`swin_o365` 模型在 Objects365 上预训练后 finetune 到 COCO，因此**只支持 COCO 80 类**：

| ID | 类别 | ID | 类别 | ID | 类别 | ID | 类别 |
|---:|------|---:|------|---:|------|---:|------|
| 0  | person        | 20 | elephant      | 40 | wine glass    | 60 | dining table   |
| 1  | bicycle       | 21 | bear          | 41 | cup           | 61 | toilet         |
| 2  | car           | 22 | zebra         | 42 | fork          | 62 | tv             |
| 3  | motorcycle    | 23 | giraffe       | 43 | knife         | 63 | laptop         |
| 4  | airplane      | 24 | backpack      | 44 | spoon         | 64 | mouse          |
| 5  | bus           | 25 | umbrella      | 45 | bowl          | 65 | remote         |
| 6  | train         | 26 | handbag       | 46 | banana        | 66 | keyboard       |
| 7  | truck         | 27 | tie           | 47 | apple         | 67 | cell phone     |
| 8  | boat          | 28 | suitcase      | 48 | sandwich      | 68 | microwave      |
| 9  | traffic light | 29 | frisbee       | 49 | orange        | 69 | oven           |
| 10 | fire hydrant  | 30 | skis          | 50 | broccoli      | 70 | toaster        |
| 11 | stop sign     | 31 | snowboard     | 51 | carrot        | 71 | sink           |
| 12 | parking meter | 32 | sports ball   | 52 | hot dog       | 72 | refrigerator   |
| 13 | bench         | 33 | kite          | 53 | pizza         | 73 | book           |
| 14 | bird          | 34 | baseball bat  | 54 | donut         | 74 | clock          |
| 15 | cat           | 35 | baseball glove| 55 | cake          | 75 | vase           |
| 16 | dog           | 36 | skateboard    | 56 | chair         | 76 | scissors       |
| 17 | horse         | 37 | surfboard     | 57 | couch         | 77 | teddy bear     |
| 18 | sheep         | 38 | tennis racket | 58 | potted plant  | 78 | hair drier     |
| 19 | cow           | 39 | bottle        | 59 | bed           | 79 | toothbrush     |

## 备注

- COCO 有 `bear (21)` 但**没有 gorilla**。检测大猩猩可尝试：
  - `bear` — 可能有一定泛化能力
  - `elephant` — 体型相似的大型动物
