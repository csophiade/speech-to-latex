# System C on s2l_test500 (LLM: gpt-4.1-mini, n=500)

```json
{
  "n": 500,
  "cer": 0.3592,
  "nem": 0.282,
  "symeq": 0.31,
  "symeq_unparseable": 0.156,
  "symeq_timeout": 0,
  "symeq_coverage": 0.854,
  "symeq_on_parseable": 0.3396,
  "latency_median_s": 1.278,
  "chosen_parses": 0.88,
  "chosen_from_beam_gt1": 0.072,
  "pool_size_mean": 3.58,
  "oracle_nem": 0.362,
  "n_human": 270,
  "cer_human": 0.3524,
  "n_tts": 230,
  "cer_tts": 0.3668
}
```

10 random outputs (seed 42):

| # | Whisper best-1 | Prediction | Ground truth | NEM | SymEq |
|---|---|---|---|---|---|
| 1 | The derivative of e to the power x with respect to x equals e to the power x. | `\frac{d}{dx} e^x = e^x` | `\frac{d}{dx} e^x = e^x` | ✓ | equal |
| 2 | alpha sub v superscript i subscript a superscript j | `\alpha_v^{i_a^j}` | `\alpha_ { v^i_a } ^j` | ✗ | not_equal |
| 3 | 3x minus 4y squared equals 8. | `3x - 4y^2 = 8` | `3x - 4y^2 = 8` | ✓ | equal |
| 4 | The sample variance equals the sum of x of i minus the sample mean squared divided by n minus 1. | `s^2 = \frac{\sum_{i=1}^n (x_i - \bar{x})^2}{n - 1}` | `s^2 = \frac{\sum_{i=1}^{n} (x_i - \bar{x})^2}{n-1}` | ✓ | equal |
| 5 | g of x equals x minus 1. | `g(x) = x - 1` | `\text{g}(x)=x-1` | ✗ | equal |
| 6 | f of x equals sending 3x now. | `f(x)=	ext{sin}(3x)` | `f(x) = \sin 3x` | ✗ | not_equal |
| 7 | f, the function f, 0, 1, 1, raised to the power of n, maps to the set of real numbers. | `f_{0,1,1}^n \to \mathbb{R}` | `f : [ 0,1 ] ^ { N } \rightarrow\mathbb { R }` | ✗ | unparseable |
| 8 | Bitter. | `b` | `\beta` | ✗ | not_equal |
| 9 | The QCD interaction term is GFABC, AMU-B, ANU-CL. | `G_F^{ABC}` | `\mathcal{L}_{int} = g_s f^{abc} A_\mu^b A_\nu^c` | ✗ | not_equal |
| 10 | alpha sub i plus 1 alpha sub i | `\alpha_{i+1} \alpha_i` | `\alpha_ { i+1 } \alpha_i` | ✓ | equal |
