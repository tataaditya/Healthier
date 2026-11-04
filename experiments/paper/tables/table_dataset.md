# Dataset summary

Candidate corpus: **19,087** images; **373 duplicates removed**; **18,714 unique** images used.

| Split | Source | Images | Staple | Side | Vegetables | Fruits | Others |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| Train | UECFoodPIXCOMPLETE | 7,478 | 4,188 | 3,614 | 1,466 | 0 | 4,996 |
| Train | FoodSeg103 | 4,206 | 3,369 | 3,685 | 9,223 | 3,513 | 4,103 |
| Validation | UECFoodPIXCOMPLETE | 1,285 | 749 | 608 | 257 | 0 | 927 |
| Validation | FoodSeg103 | 732 | 585 | 626 | 1,619 | 618 | 638 |
| External test | FoodSeg103 validation | 2,096 | 1,629 | 1,751 | 4,718 | 1,649 | 2,004 |
| External test | UECFoodPIXCOMPLETE test | 948 | 543 | 463 | 224 | 0 | 754 |
| External test | FruitSeg30 | 1,969 | 0 | 0 | 0 | 1,993 | 0 |

Cells are **instance counts**. External test splits were never used for training or model selection. The validation split is 15% of the training pool (not the 20% stated in the original draft).
