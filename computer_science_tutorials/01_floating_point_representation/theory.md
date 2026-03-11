# Floating-Point Representation

## How Computers Think About Numbers

In pure mathematics, the real number line is continuous. Between any two numbers, no matter how close, there are infinitely many other numbers. However, computers have finite memory. They must represent infinite continuous values using a finite number of bits.

How do we cram an infinite number line into a fixed box of 32 bits (or 16 bits)? The answer is **Floating-Point Representation**, primarily defined by the IEEE 754 standard. Floating-point math is a clever trade-off: it sacrifices exact precision to provide an enormous *dynamic range*.

> **The Scientific Notation Analogy:**
> Think of scientific notation: $-1.234 \times 10^5$. We have a **sign** ($-$), a **significand or mantissa** ($1.234$), and an **exponent** ($5$). Floating-point uses the exact same concept, just in base 2 (binary).

## The Anatomy of a Floating-Point Number

A floating-point number is partitioned into three segments:
- **Sign bit ($S$):** 1 bit. 0 for positive, 1 for negative.
- **Exponent ($E$):** Determines the scale (magnitude) of the number. It's stored with a *bias* to allow for negative exponents without using a sign bit just for the exponent.
- **Mantissa / Fraction ($M$):** Determines the precision (significant digits) of the number.

$$ \text{Value} = (-1)^S \times (1 + M) \times 2^{E - \text{bias}} $$

Notice the "$1 + M$". Because the leading bit of a binary number in scientific notation is almost always 1 (e.g., $1.\text{xxxxx} \times 2^E$), we don't even bother storing the leading 1. It is *implicit*.

## Single Precision: float32

**float32** (IEEE 754 single precision) has been the gold standard for graphics and standard numerical computing for decades.

- **Sign:** 1 bit
- **Exponent:** 8 bits (Bias = 127)
- **Mantissa:** 23 bits
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

**The Problem:** The maximum value is only $65,504$. In deep learning, gradients and activations can easily overflow this limit, leading to `NaN` (Not a Number) errors. Training with float16 requires careful *loss scaling* to keep values within this narrow window.

### 2. The Google Brain Solution: bfloat16
To fix the narrow range of float16, Google introduced **Brain Floating Point (bfloat16)** for its TPUs (now widely supported on GPUs too). The idea was brutally simple: take a standard float32, keep the exponent exactly the same, and just chop off the last 16 bits of the mantissa.

- **Sign:** 1 bit
- **Exponent:** 8 bits (Same as float32!)
- **Mantissa:** 7 bits
- **Range:** Same as float32 ($\approx 3.4 \times 10^{38}$)
- **Precision:** $\approx 2$ to 3 decimal digits

**Why bfloat16 wins for Deep Learning:**
In ML, having a large **dynamic range** (avoiding `NaN` overflow) is far more important than having high **precision**. Neural nets act as statistical engines—they don't care if a weight is $0.1234567$ instead of $0.123$. The noise from dropping precision actually acts as a mild regularizer. Because bfloat16 has the same exponent as float32, you can seamlessly swap float32 for bfloat16 with zero code changes and no loss scaling needed!

## Machine Epsilon and Catastrophic Cancellation

Because precision is finite, floating-point numbers have "gaps" between them. The gap between $1.0$ and the next representable number is called **Machine Epsilon ($\epsilon$)**.

- float32 $\epsilon \approx 1.19 \times 10^{-7}$
- float16 $\epsilon \approx 9.77 \times 10^{-4}$
- bfloat16 $\epsilon \approx 7.81 \times 10^{-3}$

This leads to a terrifying phenomenon called **Catastrophic Cancellation**. If you subtract two very large, nearly identical numbers, the significant digits cancel out, leaving only the "garbage" noise at the end of the mantissa. In algorithms like variance calculation or softmax, if you write the math naively, catastrophic cancellation will destroy your gradients. This is why we use mathematically equivalent but numerically stable alternatives (like `logsumexp`).