# Track-A shortlist (TUM fr1 mean ATE)

Source EXPs under `results/20260922-*.json` and `results/20260923-*.json`.

| config | mean ATE (m) | vs FP16 | role |
|--------|-------------:|-------:|------|
| FP16 | 0.0295 | +0.0% | baseline |
| FP8 e4m3 | 0.0294 | -0.4% | accuracy champ |
| W8A8 | 0.0315 | +6.8% | uniform |
| W4 geom-protect K=7 | 0.0316 | +7.1% | method wrinkle |
| W4 mag-protect K=7 | 0.0328 | +10.9% | H3 proxy |
| W4 uniform | 0.0333 | +12.8% | uniform |

## H3 (matched K=7 W4 protect)

| allocator | mean ATE | vs FP16 | vs uniform W4 |
|-----------|---------:|--------:|--------------:|
| Geometry greedy | 0.0316 | +7.1% | −5.1% |
| Magnitude L1 | 0.0328 | +10.9% | −1.7% |
| Uniform W4 | 0.0333 | +12.8% | — |

Geometry beats magnitude by 3.6% relative mean ATE at equal unit budget.

## K-budget sweep (greedy W4 protect)

| K | mean ATE | vs FP16 |
|--:|---------:|--------:|
| 3 | 0.0337 | +14.2% |
| 5 | 0.0319 | +8.0% |
| 7 | 0.0316 | +7.1% |
| 9 | 0.0311 | +5.4% |
| 11 | 0.0335 | +13.5% |
