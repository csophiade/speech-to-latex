# System A on mathspeech (LLM: gpt-4.1-mini, n=1101)

```json
{
  "n": 1101,
  "cer": 0.3432,
  "nem": 0.4096,
  "symeq": 0.5749,
  "symeq_unparseable": 0.0763,
  "symeq_timeout": 0,
  "symeq_coverage": 0.9355,
  "symeq_on_parseable": 0.6078,
  "latency_median_s": 1.05,
  "chosen_parses": 0.9646,
  "chosen_from_beam_gt1": 0.0018,
  "pool_size_mean": 3.73,
  "oracle_nem": 0.5341
}
```

10 random outputs (seed 42):

| # | Whisper best-1 | Prediction | Ground truth | NEM | SymEq |
|---|---|---|---|---|---|
| 1 | 2x plus 2yy prime. | `2x + 2yy'` | `$2x+2yy^{\prime}$` | ✗ | equal |
| 2 | 6 minus 16 over 4 minus 12. | `\frac{6-16}{4-12}` | `$\frac{6 - 16}{4 - 12}$` | ✓ | equal |
| 3 | 4x minus 1. | `4x - 1` | `$(4x\ -\ 1)$` | ✗ | equal |
| 4 | Y n minus 2 plus y n minus 1 over 2. | `\frac{y_{n-2} + y_{n-1}}{2}` | `$\displaystyle \frac{y_{n-2}+y_{n-1}}{2}$` | ✓ | equal |
| 5 | L equals the integral from 1 to x dt over t. | `L=\int_1^x \frac{dt}{t}` | `$L(x)=\int_{1}^{x}\frac{dt}{t}$` | ✗ | not_equal |
| 6 | It's one plus a half v squared over c squared. | `1 + \frac{\frac{1}{2} v^2}{c^2}` | `$1+\frac{1}{2}v^{2}/c^{2}$` | ✗ | equal |
| 7 | y equals free t. | `y=3t` | `$y=3\mathfrak{t}$` | ✗ | not_equal |
| 8 | plus or minus a e to the minus x squared over 2. | `\pm a e^{-\frac{x^2}{2}}` | `$\pm ae^{-x^{2}/2}$` | ✗ | equal |
| 9 | y minus x squared y minus y cubed over 3. | `\frac{y - x^2 y - y^3}{3}` | `$y-x^{2}y-\frac{y^{3}}{3}$` | ✗ | not_equal |
| 10 | Natural log of e is equal to 1. | `\ln e = 1` | `$\ln(e)=1$` | ✗ | equal |
