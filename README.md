# Propensity Score Matcher

A pure-Python (stdlib-only) implementation of propensity score matching with logistic regression, multiple matching algorithms, and covariate balance assessment.

## Features

- **Logistic regression**: IRLS/Newton-Raphson for propensity score estimation
- **Matching algorithms**:
  - Nearest neighbor (with caliper)
  - Optimal matching (greedy)
  - Stratification
- **Balance assessment**: Standardized Mean Difference (SMD)
  - SMD < 0.1: good balance
  - SMD > 0.1: imbalance
- **Treatment effect estimation**: ATT, ATC, ATE from matched pairs
- **Matching with/without replacement**
- **CSV batch processing**

## Formulas

### Propensity Score
P(T=1|X) = 1 / (1 + exp(-Xβ))

Estimated via logistic regression (IRLS).

### Standardized Mean Difference
SMD = |mean_treated - mean_control| / √((var_treated + var_control) / 2)

### Treatment Effect (from matched pairs)
ATE = Σ(Y_treated - Y_control) / n_pairs

## Usage

### Command Line

```bash
# Run matching
python cli.py match --treatment 1 1 0 0 1 0 \
    --covariates '[[0.5,65],[1.2,70],[0.8,55],[1.5,60],[0.3,72],[1.1,68]]' \
    --outcome 12.5 10.2 8.1 9.5 14.0 7.8

# Check balance
python cli.py balance --treatment 1 1 0 0 1 0 \
    --covariates '[[0.5],[1.2],[0.8],[1.5],[0.3],[1.1]]'

# Estimate propensity scores
python cli.py ps --treatment 1 1 0 0 1 0 \
    --covariates '[[0.5],[1.2],[0.8],[1.5],[0.3],[1.1]]'

# Batch CSV processing
python cli.py batch --input sample.csv --output results.csv
```

### Python API

```python
from ps_matcher import propensity_score_match, estimate_propensity_scores

# Full matching pipeline
result = propensity_score_match(
    treatment=[1, 1, 0, 0, 1, 0],
    covariates=[[0.5, 65], [1.2, 70], [0.8, 55], [1.5, 60], [0.3, 72], [1.1, 68]],
    outcome=[12.5, 10.2, 8.1, 9.5, 14.0, 7.8],
    method="nearest",
    caliper=0.2
)

print(f"Matched pairs: {result['matching']['n_matched']}")
print(f"Effect: {result['effect']['effect']:.4f}")
```

## CSV Format

| treatment | age | income | education | outcome |
|-----------|-----|--------|-----------|---------|
| 1         | 65  | 50000  | 16        | 12.5    |
| 0         | 55  | 45000  | 12        | 8.1     |

## Testing

```bash
python -m pytest test_ps_matcher.py -v
```

## License

MIT
