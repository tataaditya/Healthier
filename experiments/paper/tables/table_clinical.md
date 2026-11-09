# Clinical Rules Engine verification and end-to-end propagation (131 verified images)

- Daily-limit values matched against the specification: **36/36** (6 profiles × 6 limit fields)
- Status decisions matched against the specification: **30/30** (6 profiles × 5 food scenarios)

End-to-end decision propagation across 131 verified images × 6 clinical profiles = **786 decisions** (426 decisions on the 71 images with a nutrition table):

| Pipeline | Status unchanged (all) | Status unchanged (table images) | GT DANGER changed to SAFE | GT WARNING changed to SAFE |
| :--- | ---: | ---: | ---: | ---: |
| C. Raw PaddleOCR | 600/786 (76.3%) | 284/426 (66.7%) | 9/25 | 47/117 |
| D. Parser without grid | 644/786 (81.9%) | 308/426 (72.3%) | 17/25 | 71/117 |
| E. Full pipeline | 668/786 (85.0%) | 335/426 (78.6%) | 18/25 | 55/117 |
