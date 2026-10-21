# System B on s2l_test500 (LLM: gpt-4.1-mini, n=500)

```json
{
  "n": 500,
  "cer": 0.3738,
  "nem": 0.288,
  "symeq": 0.292,
  "symeq_unparseable": 0.554,
  "symeq_timeout": 0,
  "latency_median_s": 1.325,
  "parse_fail_first_candidate": 0.566,
  "output_unparseable": 0.538,
  "no_candidates": 4,
  "unverified_rate": 0.538,
  "symeq_human": 0.3296,
  "n_human": 270,
  "symeq_tts": 0.2478,
  "n_tts": 230
}
```

10 random outputs (seed 42):

| # | Whisper best-1 | Prediction | Ground truth | NEM | SymEq |
|---|---|---|---|---|---|
| 1 | The derivative of e to the power x with respect to x equals e to the power x. | `\frac{d}{dx} e^x = e^x` | `\frac{d}{dx} e^x = e^x` | ✓ | equal |
| 2 | alpha sub v superscript i subscript a superscript j | `\alpha_v^{i_a^j}` | `\alpha_ { v^i_a } ^j` | ✗ | not_equal |
| 3 | 3x minus 4y squared equals 8. | `3x - 4y^2 = 8` | `3x - 4y^2 = 8` | ✓ | equal |
| 4 | The sample variance equals the sum of x of i minus the sample mean squared divided by n minus 1. | `s^2 = \frac{\sum (x_i - \bar{x})^2}{n - 1}` | `s^2 = \frac{\sum_{i=1}^{n} (x_i - \bar{x})^2}{n-1}` | ✗ | unparseable |
| 5 | g of x equals x minus 1. | `g(x) = x - 1` | `\text{g}(x)=x-1` | ✗ | unparseable |
| 6 | f of x equals sending 3x now. | `f(x)=	ext{sin}(3x)` | `f(x) = \sin 3x` | ✗ | not_equal |
| 7 | f, the function f, 0, 1, 1, raised to the power of n, maps to the set of real numbers. | `f : \{0,1\}^n \to \mathbb{R}` | `f : [ 0,1 ] ^ { N } \rightarrow\mathbb { R }` | ✗ | unparseable |
| 8 | Bitter. | `b` | `\beta` | ✗ | unparseable |
| 9 | The QCD interaction term is GFABC, AMU-B, ANU-CL. | `G F^{A B C} \bar{\psi}_B \psi_C` | `\mathcal{L}_{int} = g_s f^{abc} A_\mu^b A_\nu^c` | ✗ | unparseable |
| 10 | alpha sub i plus 1 alpha sub i | `\alpha_{i+1} \alpha_i` | `\alpha_ { i+1 } \alpha_i` | ✓ | equal |
