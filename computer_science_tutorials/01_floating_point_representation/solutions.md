# Solutions

## 1. Decimal to Binary Float

1. **Sign:** Negative, so S = 1.
2. **Binary conversion:** $6 = 110_2$, $0.5 = 1/2 = 0.1_2$. So $6.5 = 110.1$ in binary.
3. **Normalize:** $110.1 = 1.101 \times 2^2$.
4. **Exponent:** True exponent = 2. Stored exponent = True exponent + Bias = $2 + 3 = 5$. In 3 bits, 5 is `101`.
5. **Mantissa:** The fractional part is `101`. We pad with zeros to get 4 bits: `1010`.

**Final 8-bit representation:** `1 | 101 | 1010`

## 2. Analyzing bfloat16 Limits

bfloat16 has 7 bits of mantissa. Its machine epsilon is $2^{-7} \approx 0.00781$.

Machine epsilon is the smallest number $x$ such that $1.0 + x > 1.0$. Because $0.001$ is smaller than $0.00781$, the hardware shifts the mantissa of $0.001$ so far to the right to align the exponents that it completely falls off the edge of the 7-bit register. It rounds to 0. Thus, $1.0 + 0.001 = 1.0$.