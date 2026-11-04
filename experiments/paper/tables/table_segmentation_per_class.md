# Per-class instance-level metrics

| Dataset | Class | GT inst. | Mask AP50 | Mask AP50-95 | Box AP50 | Box AP50-95 |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: |
| UECFoodPIXCOMPLETE test | Staple Food | 543 | 85.70 | 73.28 | 87.51 | 77.34 |
| UECFoodPIXCOMPLETE test | Side Dishes | 463 | 70.71 | 53.46 | 71.48 | 58.82 |
| UECFoodPIXCOMPLETE test | Vegetables | 224 | 55.42 | 40.59 | 55.54 | 44.07 |
| UECFoodPIXCOMPLETE test | Fruits | 0 | - | - | - | - |
| UECFoodPIXCOMPLETE test | Others | 754 | 65.17 | 50.57 | 65.90 | 52.24 |
| FoodSeg103 validation | Staple Food | 1629 | 63.87 | 49.92 | 65.55 | 54.95 |
| FoodSeg103 validation | Side Dishes | 1751 | 61.61 | 47.21 | 64.35 | 52.22 |
| FoodSeg103 validation | Vegetables | 4718 | 65.83 | 48.27 | 66.85 | 52.74 |
| FoodSeg103 validation | Fruits | 1649 | 74.96 | 56.59 | 72.59 | 58.29 |
| FoodSeg103 validation | Others | 2004 | 51.24 | 39.15 | 52.95 | 43.19 |
| FruitSeg30 | Staple Food | 0 | - | - | - | - |
| FruitSeg30 | Side Dishes | 0 | - | - | - | - |
| FruitSeg30 | Vegetables | 0 | - | - | - | - |
| FruitSeg30 | Fruits | 1993 | 17.29 | 13.90 | 17.68 | 13.30 |
| FruitSeg30 | Others | 0 | - | - | - | - |

Computed with `experiments/scripts/per_class_metrics.py` (pycocotools, ground truth rebuilt from the YOLO polygon labels, no images required). Classes with no GT in a split are left empty.
