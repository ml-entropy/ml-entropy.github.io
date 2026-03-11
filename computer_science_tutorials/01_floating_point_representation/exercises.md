# Exercises

## 1. Decimal to Binary Float
Suppose you have a tiny 8-bit floating point format (1 sign bit, 3 exponent bits, bias=3, 4 mantissa bits). Convert the decimal number `-6.5` to this format.

## 2. Analyzing bfloat16 Limits
Explain why attempting to add `1.0 + 0.001` in pure bfloat16 arithmetic results in exactly `1.0`.