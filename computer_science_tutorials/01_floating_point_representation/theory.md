# Floating-Point Representation

## How Computers Think About Numbers

In pure mathematics, the real number line is continuous. Between any two numbers (like $1.1$ and $1.2$), no matter how close, there are infinitely many other numbers. However, computers have finite memory. They must represent an infinite continuous number line using a fixed, finite number of boxes (bits).

How do we cram infinity into a fixed box of 32 bits? The answer is **Floating-Point Representation**, primarily defined by the IEEE 754 standard. Floating-point math is a clever trade-off: it sacrifices exact precision to provide an enormous *dynamic range* (the ability to represent both microscopically small numbers and astronomically large numbers).

> **The Scientific Notation Analogy:**
> Think of scientific notation from physics or chemistry class: $-1.234 \times 10^5$. 
> We have a **sign** ($-$), a **significand or mantissa** ($1.234$), and an **exponent** ($5$). 
> Floating-point uses the exact same concept, just in base-2 (binary) instead of base-10 (decimal).

## The Anatomy of a Floating-Point Number

A floating-point number is partitioned into three segments:
1. **Sign bit ($S$):** 1 bit. `0` for positive, `1` for negative.
2. **Exponent ($E$):** Determines the scale (magnitude) of the number. It's stored with a special *bias* to handle negative exponents gracefully.
3. **Mantissa / Fraction ($M$):** Determines the precision (the significant digits) of the number.

The formula to calculate the actual value is:
$$ \text{Value} = (-1)^S \times (1 + M) \times 2^{E - \text{bias}} $$

### The "Implicit 1" and Representing 2, 3, and 4
You might look at the formula `(1 + M)` and ask: *If the mantissa always starts with 1, how do we represent numbers that start with 2, 3, or 4?*

This is a common point of confusion when switching from decimal (base-10) to binary (base-2)! 
In decimal scientific notation, the leading digit can be anything from 1 to 9 (e.g., $4.56 \times 10^3$). 
But **in binary, the only possible digits are 0 and 1**. 

Any non-zero binary number, when normalized (written in scientific notation), **must** start with a 1. Let's look at examples:
*   **Decimal 2:** In binary, this is `10`. Normalized, it becomes $1.0_2 \times 2^1$.
*   **Decimal 3:** In binary, this is `11`. Normalized, it becomes $1.1_2 \times 2^1$.
*   **Decimal 4:** In binary, this is `100`. Normalized, it becomes $1.0_2 \times 2^2$.
*   **Decimal 6.5:** In binary, this is `110.1`. Normalized, it becomes $1.101_2 \times 2^2$.

Notice how *every single normalized binary number starts with `1.`*. Because it **always** starts with 1, the designers of the IEEE 754 standard realized they didn't actually need to store it! By assuming the 1 is always there (the "implicit 1"), they saved a bit, effectively giving us an extra bit of precision for free. The mantissa ($M$) only stores the fractional part *after* the decimal point.

### The Exponent Bias: Why Not Use Normal Negative Numbers?
The exponent needs to represent both large numbers (positive exponent, e.g., $2^{10}$) and tiny fractions (negative exponent, e.g., $2^{-10}$). Normally in computers, negative integers are represented using "Two's Complement." So why does floating-point use a **Bias** instead?

**The Goal:** Hardware engineers wanted floating-point numbers to be fast to compare. If you take two positive floating-point numbers and just compare their raw binary bits as if they were standard integers, the larger integer should correspond to the larger float.

**What happens without Bias (using Two's Complement)?**
In 8-bit Two's complement:
*   Exponent `+1` is `00000001`
*   Exponent `-1` is `11111111`

If a computer compares these raw bit patterns, it will see `11111111` as vastly larger than `00000001`. The hardware would incorrectly conclude that $1.0 \times 2^{-1}$ is larger than $1.0 \times 2^1$. To fix this, the CPU would need special, slower circuitry just to compare floats.

**What happens with Bias?**
Instead of Two's complement, we simply add a fixed positive number (the bias) to the true exponent. For an 8-bit exponent, the bias is 127.
*   True Exponent `+1`: Store as $1 + 127 = 128$ (`10000000` in binary)
*   True Exponent `-1`: Store as $-1 + 127 = 126$ (`01111110` in binary)

Now, `10000000` is naturally greater than `01111110`. By using a bias, the exponent becomes a strictly positive integer in binary. This perfectly preserves lexicographical ordering, allowing CPUs to use blazing-fast integer comparison circuits to compare floating-point numbers!

## Single Precision: float32

**float32** (IEEE 754 single precision) has been the gold standard for graphics and standard numerical computing for decades.

- **Sign:** 1 bit
- **Exponent:** 8 bits (Bias = 127)
- **Mantissa:** 23 bits (effectively 24 bits of precision due to the implicit 1)
- **Range:** $\approx 1.4 \times 10^{-45}$ to $\approx 3.4 \times 10^{38}$
- **Precision:** $\approx 7$ decimal digits

## The Deep Learning Revolution: float16 and bfloat16

For years, training neural networks relied on float32. But around 2017, the AI community realized a profound truth: *neural networks are surprisingly robust to noise and low precision.*

Moving from 32 bits to 16 bits halves the memory requirements, allowing for double the batch size or model size. It also dramatically speeds up matrix multiplications on modern hardware (like Tensor Cores). The problem is: *which 16 bits do you keep?*

### 1. IEEE float16 (Half Precision)
- **Sign:** 1 bit
- **Exponent:** 5 bits (Bias = 15)
- **Mantissa:** 10 bits
- **Range:** $\approx 5.96 \times 10^{-8}$ to $65,504$

**The Problem:** The maximum value is only $65,504$. In deep learning, gradients and activations can easily overflow this limit, leading to `NaN` (Not a Number) errors. Training with float16 requires careful *loss scaling* (multiplying gradients by a large number to keep them from underflowing, then dividing later) to keep values within this narrow window.

### 2. The Google Brain Solution: bfloat16
To fix the narrow range of float16, Google introduced **Brain Floating Point (bfloat16)** for its TPUs (now widely supported on GPUs too). The idea was brutally simple: take a standard float32, keep the exponent exactly the same, and just chop off the last 16 bits of the mantissa.

- **Sign:** 1 bit
- **Exponent:** 8 bits (Same as float32!)
- **Mantissa:** 7 bits
- **Range:** Same as float32 ($\approx 3.4 \times 10^{38}$)
- **Precision:** $\approx 2$ to 3 decimal digits

**Why bfloat16 wins for Deep Learning:**
In ML, having a large **dynamic range** (avoiding `NaN` overflow) is far more important than having high **precision**. Neural nets act as statistical engines—they don't care if a weight is $0.1234567$ instead of $0.123$. The noise from dropping precision actually acts as a mild regularizer, preventing overfitting. Because bfloat16 has the same exponent as float32, you can seamlessly swap float32 for bfloat16 with zero code changes and no loss scaling needed!

## Machine Epsilon and Catastrophic Cancellation

Because precision is finite, floating-point numbers have "gaps" between them. The gap between $1.0$ and the next representable number is called **Machine Epsilon ($\epsilon$)**.

- float32 $\epsilon \approx 1.19 \times 10^{-7}$
- float16 $\epsilon \approx 9.77 \times 10^{-4}$
- bfloat16 $\epsilon \approx 7.81 \times 10^{-3}$

This leads to a terrifying mathematical phenomenon called **Catastrophic Cancellation**. 

Imagine you have two very close numbers, $x = 1.234567$ and $y = 1.234566$, and your computer can only store 7 significant digits. If you subtract them:
$$ x - y = 0.000001 $$
You've just lost 6 digits of precision! The result has only 1 significant digit. In computer memory, the remaining bits of the mantissa will just be filled with computational "garbage" noise. 

In ML algorithms like variance calculation or Softmax, if you write the math naively (e.g., subtracting large exponential values), catastrophic cancellation will amplify this garbage noise and destroy your gradients. This is why we use mathematically equivalent but numerically stable alternatives (like the `logsumexp` trick) to avoid subtracting close, large numbers.