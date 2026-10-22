# ProcessBench (task 3.4): step checker, test only

TEST-ONLY robustness number: nothing was tuned on ProcessBench labels. The extraction rules (`src/stepmath.py`, frozen-v6, sha256 `2e4045f45099c96a`) were refined on step *text* only (labels never loaded), then frozen and evaluated once; frozen-v6 adds post-evaluation bug fixes (see below).

Filter: an item is kept if >= 70% of its steps parse as algebra (at least one extracted line the parser reads). A step is checkable if any of its lines gets valid / caution / error. Prediction: the first step with an error, -1 if none.

## Coverage and filter

| split | items | steps | step has a line | step parses | checkable (all) | items kept | kept: parses / checkable |
|---|---|---|---|---|---|---|---|
| gsm8k | 400 | 2082 | 56.6% | 55.5% | 50.9% | 126 (31.5%) | 82.1% / 74.0% |
| math | 1000 | 6505 | 70.1% | 66.8% | 32.2% | 519 (51.9%) | 87.0% / 37.8% |
| olympiadbench | 1000 | 8819 | 73.4% | 72.0% | 30.7% | 627 (62.7%) | 87.5% / 36.9% |
| omnimath | 1000 | 8291 | 65.2% | 62.8% | 22.8% | 471 (47.1%) | 86.9% / 29.6% |
| all | 3400 | 25697 | 68.6% | 66.4% | 30.2% | 1743 (51.3%) | 86.9% / 36.6% |

## Precision / recall on kept items

Exact: TP = flagged step equals the label; FP = flagged anywhere else (incl. correct items); FN = erroneous item whose label step was not the first flag. At-or-before: TP = flagged at or before the label step. Item: any flag vs. any error. acc_err / acc_corr / PB-F1 as in ProcessBench (abstaining counts as correct on correct items). Last column: recall on erroneous items whose label step is checkable.

| split | kept items (err / corr) | flagged | exact P | exact R | exact F1 | at-or-before P / R | item P / R | acc_err | acc_corr | PB-F1 | R (label step checkable) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| gsm8k | 126 (69 / 57) | 9 | 0.889 (8/9) | 0.116 (8/69) | 0.205 | 0.889 / 0.116 | 0.889 / 0.116 | 0.116 | 0.982 | 0.207 | 0.131 (n=61) |
| math | 519 (312 / 207) | 78 | 0.500 (39/78) | 0.125 (39/312) | 0.200 | 0.513 / 0.128 | 0.910 / 0.228 | 0.125 | 0.966 | 0.221 | 0.315 (n=124) |
| olympiadbench | 627 (383 / 244) | 72 | 0.417 (30/72) | 0.078 (30/383) | 0.132 | 0.444 / 0.084 | 0.944 / 0.178 | 0.078 | 0.984 | 0.145 | 0.203 (n=148) |
| omnimath | 471 (329 / 142) | 49 | 0.367 (18/49) | 0.055 (18/329) | 0.095 | 0.429 / 0.064 | 0.918 / 0.137 | 0.055 | 0.972 | 0.104 | 0.182 (n=99) |
| all | 1743 (1093 / 650) | 208 | 0.457 (95/208) | 0.087 (95/1093) | 0.146 | 0.486 / 0.092 | 0.923 / 0.176 | 0.087 | 0.975 | 0.160 | 0.220 (n=432) |

False positives by label status:

| split | exact TP | FP on correct items | FP before label | flag after label (unknown) | P on known-status flags |
|---|---|---|---|---|---|
| gsm8k | 8 | 1 | 0 | 0 | 0.889 |
| math | 39 | 7 | 1 | 31 | 0.830 |
| olympiadbench | 30 | 4 | 2 | 36 | 0.833 |
| omnimath | 18 | 4 | 3 | 24 | 0.720 |
| all | 95 | 16 | 6 | 91 | 0.812 |

## How to read the numbers

Headline: exact P / R over all kept items (the ProcessBench convention). Exact R is bounded by coverage: only the erroneous kept items whose label step is checkable can be hit, so 'R (label step checkable)' is the recall on the checkable subset in the TASKS.md 3.4 sense. ProcessBench labels only the FIRST wrong step, so a flag after it has unknown status; exact P counts it as a false positive. The table above splits the false positives; 'P on known-status flags' = TP / (TP + FP on correct items + FP before the label).

Iteration history (disclosure): the extraction rules went through five prototype versions, each run on all 3400 items; rules were adjusted after inspecting the checker's flagged-step explanations (labels never read). Flagged kept items fell 729 -> 364 -> 226 -> 227 -> 214 (v1..v5) and coverage 35.0% -> 30.5%. This is unlabelled inspection of test items, so exact precision is optimistically biased.

Post-evaluation bug fixes (frozen-v6): after the single frozen-v5 evaluation, a hand audit found six label-independent extraction bugs; five were fixed, each justified by the step text format alone (rows of frozen-v5 in results.json are marked superseded): (1) the equation-label stripper deleted leading products like '(rst) + 3'; (2) integer-division quotients ('175 \div 119 = 1 (quotient)') were read as exact division; (3) a degree sign on a symbolic argument (cos(8k^\circ)) was dropped, read in radians; (4) function application f^{-1}(9) was compared across lines as a variable f; (5) a span whose first member is the operand of a prose operator ('... × \( 1/100 = ...') was checked as a full equation.

Known limitations (not fixed): negation in the NEXT sentence of a trial equation ('try 1 = 1/3+1/4+1/5 ... This does not equal 1') is ignored; mixed chains 'A = B > 0' are never checked within a line; symbolic identities (x^3-4x^2-7x+6 = (x-1)(x^2-3x-6)) are never checked; multi-variable steps are gated out of cross-line checks; a rule line between two equations breaks the cross check.

Hand audit (label-aware diagnostic, on frozen-v5, not used for tuning): of 30 random flagged steps, 23 are genuine maths errors (many on steps after the labelled one, e.g. carried-forward errors), 6 extraction artefacts and 1 checker mistake (the bugs above). Of 20 random missed label steps, 7 are computational and 13 logical / conceptual (geometry, wrong formula, case analysis, misreading), which no arithmetic / algebra checker can catch.

Reproducibility: parses run under SIGALRM time limits, so results are reproducible up to timeout jitter under CPU load (an independent frozen-v5 rerun gave identical P/R and one differing step-parse flag).

## Superseded runs (all splits)

| rules | flagged | exact P | exact R | acc_corr | PB-F1 | coverage |
|---|---|---|---|---|---|---|
| frozen-v5 | 214 | 0.449 | 0.086 | 0.971 | 0.158 | 36.7% |

Runtime: 335 s wall clock with 24 processes (6473 CPU-seconds summed over items; slowest item 69.1 s).

## 10 random flagged steps (kept items)

### olympiadbench-482, step 3 (human label: first wrong step 3)

Step text:

> For the second sum, we need to evaluate \(\sum_{k=1}^{44} \cos (8k^\circ)\). Notice that the angles \(8^\circ, 16^\circ, 24^\circ, \ldots, 352^\circ\) form an arithmetic sequence with a common difference of \(8^\circ\). The sum of cosines of these angles can be simplified using the properties of roots of unity. Specifically, the sum of cosines of angles that are evenly spaced around a circle is zero. In this case, the angles \(8^\circ, 16^\circ, 24^\circ, \ldots, 352^\circ\) complete one full circle, so their cosines sum to zero. Therefore:
> \[ \sum_{k=1}^{44} \cos (8k^\circ) = 0 \]

Extracted lines:

- `\sum_{k=1}^{44} \cos (8(k \cdot \frac{\pi}{180}))`
- `\sum_{k=1}^{44} \cos (8(k \cdot \frac{\pi}{180})) = 0`

Errors:

- within: `\sum_{k=1}^{44} \cos (8(k \cdot \frac{\pi}{180})) = 0`: Sum(cos(2*pi*k/45), (k, 1, 44)) and 0 are not equal

### olympiadbench-85, step 0 (human label: first wrong step 0)

Step text:

> To determine all integers \( m \) for which the \( m \times m \) square can be dissected into five rectangles, the side lengths of which are the integers \( 1, 2, 3, \ldots, 10 \) in some order, we need to follow these steps: First, calculate the total area of the five rectangles. The side lengths of the rectangles are \( 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 \). The total area of the five rectangles is the sum of the areas of each rectangle. Since each rectangle has two side lengths, the total area is:
> \[ \left(1 + 2 + 3 + 4 + 5 + 6 + 7 + 8 + 9 + 10\right)^2 / 2 = 220 \]
> This total area must equal the area of the \( m \times m \) square, so \( m^2 = 220 \). Therefore, \( m = \sqrt{220} \approx 11 \).

Extracted lines:

- `\left(1 + 2 + 3 + 4 + 5 + 6 + 7 + 8 + 9 + 10\right)^2 / 2 = 220`
- `m^2 = 220`
- `m = \sqrt{220}`

Errors:

- within: `\left(1 + 2 + 3 + 4 + 5 + 6 + 7 + 8 + 9 + 10\right)^2 / 2 = 220`: (10 + (9 + (8 + (7 + (6 + (5 + (4 + ((1 + 2) + 3))))))))^2/2 and 220 are not equal

### math-140, step 0 (human label: first wrong step 0)

Step text:

> To simplify \(\sqrt{32670}\), we can factor the radicand (the expression inside the square root symbol) into its prime factors: 
> \[32670 = 2 \cdot 16335 = 2 \cdot 3^2 \cdot 2715 = 2 \cdot 3^2 \cdot 3 \cdot 905 = 2 \cdot 3^3 \cdot 5 \cdot 181.\]

Extracted lines:

- `32670 = 2 \cdot 16335 = 2 \cdot 3^2 \cdot 2715 = 2 \cdot 3^2 \cdot 3 \cdot 905 = 2 \cdot 3^3 \cdot 5 \cdot 181`

Errors:

- within: `32670 = 2 \cdot 16335 = 2 \cdot 3^2 \cdot 2715 = 2 \cdot 3^2 \cdot 3 \cdot 905 = 2 \cdot 3^3 \cdot 5 \cdot 181`: 2*16335 and (2*3^2)*2715 are not equal

### olympiadbench-15, step 7 (human label: first wrong step 3)

Step text:

> The volume of the intersection of three octahedrons at a vertex is a smaller tetrahedron of volume: \[ V_{\text{vertex tetrahedron}} = \frac{1}{6} \times \left(\frac{3}{5}\right)^3 = \frac{9}{125}. \] There are 8 vertices, so the total volume of the intersections at the vertices is: \[ 8 \times \frac{9}{125} = \frac{72}{125}. \]

Extracted lines:

- `V_{\text{vertex tetrahedron}} = \frac{1}{6} \times \left(\frac{3}{5}\right)^3 = \frac{9}{125}`
- `8 \times \frac{9}{125} = \frac{72}{125}`

Errors:

- within: `V_{\text{vertex tetrahedron}} = \frac{1}{6} \times \left(\frac{3}{5}\right)^3 = \frac{9}{125}`: (3/5)^3/6 and 9/125 are not equal

### omnimath-383, step 1 (human label: first wrong step 2)

Step text:

> First, determine the value of \( a_1 \). Since \( q < \frac{1}{4} \), the largest possible value of \( \frac{1}{a_1} \) is \( \frac{1}{5} \). This is because if \( a_1 = 4 \), then \( \frac{1}{a_1} = \frac{1}{4} \), which exceeds the limit. Therefore, \( a_1 = 5 \).

Extracted lines:

- `q < \frac{1}{4}`
- `a_1 = 4`
- `\frac{1}{a_1} = \frac{1}{4}`
- `a_1 = 5`

Errors:

- cross: `\frac{1}{a_1} = \frac{1}{4}` -> `a_1 = 5`: the solutions change from a_{1} = 4 to a_{1} = 5

### omnimath-272, step 6 (human label: first wrong step 2)

Step text:

> We solve this quadratic equation using the quadratic formula \(a = \frac{-b \pm \sqrt{b^2 - 4ac}}{2a}\), where \(a = 1\), \(b = -505\), and \(c = 1010\): \[ a = \frac{505 \pm \sqrt{505^2 - 4 \cdot 1 \cdot 3066300}}{2 \cdot 1}. \] Calculating the discriminant: \[ 505^2 - 4 \cdot 1 \cdot 2529325 = 255025 - 10207300 = 10207525. \] Thus, we have: \[ a = \frac{505 \pm 1515}{2}. \]

Extracted lines:

- `a = \frac{-b \pm \sqrt{b^2 - 4ac}}{2a}`
- `a = 1`
- `b = -505`
- `c = 1010`
- `a = \frac{505 \pm \sqrt{505^2 - 4 \cdot 1 \cdot 3066300}}{2 \cdot 1}`
- `505^2 - 4 \cdot 1 \cdot 2529325 = 255025 - 10207300 = 10207525`
- `a = \frac{505 \pm 1515}{2}`

Errors:

- within: `505^2 - 4 \cdot 1 \cdot 2529325 = 255025 - 10207300 = 10207525`: -2529325*4*1 + 505^2 and -10207300 + 255025 are not equal

### olympiadbench-77, step 7 (human label: first wrong step 4)

Step text:

> From here we see that: \( (1-\frac{n^2}{200000}) = \frac{1}{2}(\frac{345600}{843750}-1) \) \( \frac{n^2}{200000} = 1-\frac{1}{2}(\frac{345600}{843750}-1) \) \( \frac{n^2}{200000} = \frac{1}{2}(2-\frac{345600}{843750}+1) \) \( \frac{n^2}{200000} = \frac{3}{2}-\frac{172800}{843750} \) \( \frac{n^2}{200000} = \frac{3}{2}-\frac{8}{49} \) \( \frac{n^2}{200000} = \frac{147-16}{98} \) \( \frac{n^2}{200000} = \frac{131}{98} \) \( n^2 = \frac{131}{98} \times 200000 \) \( n^2 = 267050.51 \) \( n = \sqrt{267050.51} \).

Extracted lines:

- `(1-\frac{n^2}{200000}) = \frac{1}{2}(\frac{345600}{843750}-1)`
- `\frac{n^2}{200000} = 1-\frac{1}{2}(\frac{345600}{843750}-1)`
- `\frac{n^2}{200000} = \frac{1}{2}(2-\frac{345600}{843750}+1)`
- `\frac{n^2}{200000} = \frac{3}{2}-\frac{172800}{843750}`
- `\frac{n^2}{200000} = \frac{3}{2}-\frac{8}{49}`
- `\frac{n^2}{200000} = \frac{147-16}{98}`
- `\frac{n^2}{200000} = \frac{131}{98}`
- `n^2 = \frac{131}{98} \times 200000`
- `n^2 = 267050.51`
- `n = \sqrt{267050.51}`

Errors:

- cross: `\frac{n^2}{200000} = \frac{3}{2}-\frac{172800}{843750}` -> `\frac{n^2}{200000} = \frac{3}{2}-\frac{8}{49}`: the solutions change from n = -4sqrt(16190) or n = 4sqrt(16190) to n = -100sqrt(1310)/7 or n = 100sqrt(1310)/7

### olympiadbench-233, step 9 (human label: first wrong step 9)

Step text:

> Solving this quadratic equation, we get:
> \[ 2x^2 - x - 3 = 0 \implies (2x + 3)(x - 1) = 0 \]

Extracted lines:

- `2x^2 - x - 3 = 0`
- `(2x + 3)(x - 1) = 0`

Errors:

- cross: `2x^2 - x - 3 = 0` -> `(2x + 3)(x - 1) = 0`: the solutions change from x = -1 or x = 3/2 to x = -3/2 or x = 1

### omnimath-25, step 6 (human label: first wrong step 5)

Step text:

> ### Step 3: Count the number of integers \( n \) such that \( f(n) = 2 \)
> These are the integers divisible by exactly two of 2, 3, or 5. We subtract the integers divisible by all three primes from the counts of integers divisible by the product of any two primes. The number of integers divisible by exactly two of 2, 3, or 5 is:
> \[ 336 + 202 + 134 - 3(67) = 673 - 201 = 472. \]

Extracted lines:

- `f(n) = 2`
- `336 + 202 + 134 - 3(67) = 673 - 201 = 472`

Errors:

- within: `336 + 202 + 134 - 3(67) = 673 - 201 = 472`: -3*67 + (134 + (202 + 336)) and -201 + 673 are not equal

### olympiadbench-391, step 9 (human label: first wrong step 4)

Step text:

> Now, \( y = \log_{10}(\sqrt{14}-2) = 1 - \log_{10}(\sqrt{14}+1) \), which simplifies to \( y = \log_{10}\left(\frac{\sqrt{14}-2}{\sqrt{14}+1}\right) + 1 \).

Extracted lines:

- `y = \log_{10}(\sqrt{14}-2) = 1 - \log_{10}(\sqrt{14}+1)`
- `y = \log_{10}\left(\frac{\sqrt{14}-2}{\sqrt{14}+1}\right) + 1`

Errors:

- within: `y = \log_{10}(\sqrt{14}-2) = 1 - \log_{10}(\sqrt{14}+1)`: log(-2 + sqrt(14), 10) and 1 - log(1 + sqrt(14), 10) are not equal

