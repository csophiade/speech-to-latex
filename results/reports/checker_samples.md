# Checker sample explanations

One caught injected error per kind from the derivation test split (text input).

## divide_by_variable (qfac-0005)

```latex
x \cdot (2x - 3) = x
2x^2 - 3x = x
2x - 3 = 1
2x = 4
x = 2
```

- line 2: `2x^2 - 3x = x` -> `2x - 3 = 1`
- status: **error**
- explanation: "dividing by x may lose the solution x = 0"

## square_both_sides (lin-0010)

```latex
2(x - 5) = 2
x - 5 = 1
(x - 5)^2 = 1
x^2 - 10x + 24 = 0
(x - 4)(x - 6) = 0
x = 4 \text{ or } x = 6
```

- line 2: `x - 5 = 1` -> `(x - 5)^2 = 1`
- status: **error**
- explanation: "squaring both sides can add solutions where L = -R: x = 4 solves the new line but not the previous one"

## missed_flip (ineq-0001)

```latex
5(x + 6) - 5 < 6x + 31
5x + 30 - 5 < 6x + 31
-x + 25 < 31
-x < 6
x < -6
```

- line 4: `-x < 6` -> `x < -6`
- status: **error**
- explanation: "multiplying or dividing by a negative number must flip the inequality"
