# Tutorial 12: Convolutional Neural Networks — Learning Visual Hierarchies

## Overview
CNNs exploit the structure of images: local patterns, translation invariance, and hierarchical features. We derive the convolution operation, understand receptive fields, and see how convolutions learn feature detectors.

---

## Part 1: Why Not Fully Connected for Images?

### The Problem

A 256×256 RGB image = 196,608 inputs.

Fully connected to 1000 hidden units = **196 million parameters** in ONE layer!

**Problems**:
1. Too many parameters → overfitting
2. Ignores spatial structure
3. Not translation invariant

### The Solution: Local Connectivity

**Key insight**: Visual patterns are:
- **Local**: Edges, textures defined by nearby pixels
- **Translation invariant**: An edge is an edge anywhere in the image
- **Hierarchical**: Edges → shapes → parts → objects

CNNs exploit all three properties.

---

## Part 2: The Convolution Operation

### 1D Convolution (for intuition)

Input signal $x[n]$, filter $w[k]$ of size $K$:
$$(x * w)[n] = \sum_{k=0}^{K-1} x[n+k] \cdot w[k]$$

**Sliding the filter** across the input.

### 2D Convolution

Input $X$ (H × W), filter $W$ (K × K):
$$(X * W)[i, j] = \sum_{m=0}^{K-1}\sum_{n=0}^{K-1} X[i+m, j+n] \cdot W[m, n]$$

### With Multiple Channels

Input: $X \in \mathbb{R}^{H \times W \times C_{in}}$
Filter: $W \in \mathbb{R}^{K \times K \times C_{in}}$

$$(X * W)[i, j] = \sum_{c=0}^{C_{in}-1}\sum_{m=0}^{K-1}\sum_{n=0}^{K-1} X[i+m, j+n, c] \cdot W[m, n, c]$$

### Multiple Output Channels

Use $C_{out}$ different filters, each produces one channel:
$$Y[:, :, f] = X * W_f + b_f$$

**Total parameters**: $K \times K \times C_{in} \times C_{out} + C_{out}$

---

## Part 3: Deriving the Backward Pass

### Forward Pass (simplified)

For output pixel $y_{ij}$:
$$y_{ij} = \sum_{m,n} x_{i+m, j+n} \cdot w_{mn}$$

### Backward Pass: Gradient w.r.t. Weights

Given upstream gradient $\frac{\partial L}{\partial y_{ij}}$:

$$\frac{\partial L}{\partial w_{mn}} = \sum_{i,j} \frac{\partial L}{\partial y_{ij}} \cdot \frac{\partial y_{ij}}{\partial w_{mn}} = \sum_{i,j} \frac{\partial L}{\partial y_{ij}} \cdot x_{i+m, j+n}$$

**Key insight**: Weight gradient = **convolution of input with upstream gradient**!

### Backward Pass: Gradient w.r.t. Input

$$\frac{\partial L}{\partial x_{pq}} = \sum_{i,j} \frac{\partial L}{\partial y_{ij}} \cdot \frac{\partial y_{ij}}{\partial x_{pq}}$$

For which $(i,j)$ does $x_{pq}$ appear? When $p = i+m$, $q = j+n$.

$$\frac{\partial L}{\partial x_{pq}} = \sum_{m,n} \frac{\partial L}{\partial y_{p-m, q-n}} \cdot w_{mn}$$

**Key insight**: Input gradient = **full convolution with flipped filter**!

---

## Part 4: Padding, Stride, and Output Size

### Output Size Formula

Input: $H_{in} \times W_{in}$
Filter: $K \times K$
Padding: $P$
Stride: $S$

$$H_{out} = \left\lfloor\frac{H_{in} + 2P - K}{S}\right\rfloor + 1$$

### Padding Types

**Valid** (no padding): $P = 0$ → output shrinks
**Same** (preserve size): $P = \lfloor K/2 \rfloor$ → output same as input (for stride 1)
**Full**: $P = K - 1$ → output grows

### Stride

Skip positions when sliding filter:
- Stride 1: Dense output
- Stride 2: Output half the size (downsampling)

### Dilation

Spread filter elements apart:
$$y_{ij} = \sum_{m,n} x_{i+d \cdot m, j+d \cdot n} \cdot w_{mn}$$

where $d$ is dilation rate.

**Effect**: Larger receptive field without more parameters.

---

## Part 5: Receptive Field

### Definition

The **receptive field** of a neuron is the region of input that affects its value.

### Computing Receptive Field

For single layer: RF = K (filter size)

For stacked layers:
$$RF_l = RF_{l-1} + (K_l - 1) \times \prod_{i=1}^{l-1} S_i$$

### Example: VGG-style

Three 3×3 conv layers (stride 1):
- Layer 1: RF = 3
- Layer 2: RF = 3 + (3-1) × 1 = 5
- Layer 3: RF = 5 + (3-1) × 1 = 7

**Insight**: Three 3×3 layers have same RF as one 7×7, but fewer parameters!
- 7×7: 49 params per channel
- 3×(3×3): 27 params per channel

---

## Part 6: Pooling Operations

### Max Pooling

Take maximum in each window:
$$y_{ij} = \max_{m,n \in \text{window}} x_{i \cdot S + m, j \cdot S + n}$$

**Properties**:
- Provides translation invariance
- Reduces spatial size
- Keeps strongest activation

### Average Pooling

Take mean in each window:
$$y_{ij} = \frac{1}{K^2}\sum_{m,n \in \text{window}} x_{i \cdot S + m, j \cdot S + n}$$

### Global Average Pooling (GAP)

Average over entire spatial dimension:
$$y_c = \frac{1}{HW}\sum_{i,j} x_{ijc}$$

Converts feature maps to vector. Used before final classifier.

### Pooling Backward Pass

**Max pooling**: Gradient goes only to the max element
$$\frac{\partial L}{\partial x_{mn}} = \begin{cases} \frac{\partial L}{\partial y} & \text{if } x_{mn} \text{ was max} \\ 0 & \text{otherwise} \end{cases}$$

**Average pooling**: Gradient distributed equally
$$\frac{\partial L}{\partial x_{mn}} = \frac{1}{K^2}\frac{\partial L}{\partial y}$$

---

## Part 7: What Do Convolutions Learn?

### Layer 1: Edge Detectors

First layer filters learn **Gabor-like** patterns:
- Horizontal edges
- Vertical edges
- Diagonal edges
- Color blobs

### Layer 2-3: Textures and Parts

- Combinations of edges
- Texture patterns
- Simple shapes

### Deeper Layers: Object Parts

- Eyes, wheels, windows
- Semantic parts

### Final Layers: Full Objects

- Entire faces, cars, animals
- Category-specific patterns

### The Hierarchy

```
Pixels → Edges → Textures → Parts → Objects
```

This mirrors the **visual cortex** (V1 → V2 → V4 → IT).

---

## Part 8: Classic CNN Architectures

### LeNet-5 (1998)
```
Input(32×32) → Conv → Pool → Conv → Pool → FC → FC → Output
```
~60K parameters. First successful CNN.

### AlexNet (2012)
```
Conv(96) → Pool → Conv(256) → Pool → Conv(384) → Conv(384) → Conv(256) → Pool → FC → FC → Output
```
~60M parameters. Started deep learning revolution.

### VGG (2014)
**Philosophy**: Small (3×3) filters, deep network
```
[Conv3×3]×2 → Pool → [Conv3×3]×2 → Pool → [Conv3×3]×3 → Pool → [Conv3×3]×3 → Pool → [Conv3×3]×3 → Pool → FC
```
~138M parameters. Very regular structure.

### ResNet (2015)
**Key innovation**: Skip connections
$$y = F(x) + x$$

Allows training 100+ layer networks.

---

## Part 9: The Information-Theoretic View

### Convolution as Feature Detection

Each filter computes **mutual information** between local patch and learned pattern.

High response = input patch **matches** filter = high MI.

### Hierarchical Abstraction

Each layer compresses information:
$$H(\text{Layer}_{l+1}) < H(\text{Layer}_l)$$

But keeps task-relevant information:
$$I(\text{Layer}_{l+1}; \text{Label}) \approx I(\text{Layer}_l; \text{Label})$$

This is the **information bottleneck**!

### Translation Invariance = Compression

Same filter everywhere = **weight sharing** = compression.

Instead of learning position-specific features (wasteful), learn position-invariant features (efficient).

### Pooling as Entropy Reduction

Max/average pooling reduces entropy:
$$H(\text{pooled}) < H(\text{input})$$

Keeps the "essence" (max) or "average character" of each region.

---

## Part 10: Modern Developments

### Depthwise Separable Convolutions

Factor standard conv into:
1. **Depthwise**: One filter per input channel
2. **Pointwise**: 1×1 conv to mix channels

**Savings**: $(K^2 + C_{out})$ vs $K^2 \cdot C_{out}$ per position.

Used in MobileNet, EfficientNet.

### Attention in CNNs

**Squeeze-and-Excitation**: Learn channel weights
**Self-Attention**: Learn spatial relationships

Combines CNN efficiency with attention flexibility.

### Vision Transformers (ViT)

Replace convolutions with attention entirely.
But recent work shows convolutions + attention work best together.

---

## Summary

| Concept | Formula/Insight |
|---------|----------------|
| Convolution | $y_{ij} = \sum_{m,n} x_{i+m,j+n} \cdot w_{mn}$ |
| Output Size | $\lfloor(H + 2P - K)/S\rfloor + 1$ |
| Receptive Field | Grows with depth |
| Weight Sharing | Same filter everywhere = compression |
| Pooling | Reduces size, adds invariance |
| Hierarchy | Edges → Textures → Parts → Objects |

**The key insight**: CNNs exploit image structure (locality, translation invariance, hierarchy) to learn compressed, hierarchical representations with far fewer parameters than fully connected networks.
