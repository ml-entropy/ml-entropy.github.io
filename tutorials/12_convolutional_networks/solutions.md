# Tutorial 12: Convolutional Neural Networks - Solutions

## Part A: Theory Solutions

### Solution A1: Output Size Formula

**Step 1: Kernel positions**
With padding $P$, the effective input size becomes $H + 2P$.
The kernel of size $K$ can be placed at positions $0, S, 2S, ...$

The last valid position is where the kernel still fits:
$$\text{last position} + K \leq H + 2P$$
$$\text{last position} \leq H + 2P - K$$

With stride $S$, number of positions = $\lfloor\frac{H + 2P - K}{S}\rfloor + 1$

**Formula:**
$$\boxed{H_{out} = \left\lfloor\frac{H + 2P - K}{S}\right\rfloor + 1}$$

---

### Solution A2: Parameter Counting

**Weight parameters:**
Each output channel has one kernel for each input channel:
$$\text{Weights} = C_{out} \times C_{in} \times K \times K$$

**Bias parameters:**
$$\text{Biases} = C_{out}$$

**Total:**
$$\boxed{\text{Total} = C_{out} \times C_{in} \times K \times K + C_{out}}$$

**Comparison to FC:**
For FC layer with same input ($C_{in} \times H \times W$) and output ($C_{out} \times H \times W$):
$$\text{FC params} = (C_{in} \times H \times W) \times (C_{out} \times H \times W)$$

For a 32×32 image with 64 input and 128 output channels, 3×3 conv:
- Conv: $128 \times 64 \times 9 + 128 = 73,856$
- FC: $(64 \times 32 \times 32) \times (128 \times 32 \times 32) = 8.6$ billion!

CNNs use dramatically fewer parameters due to weight sharing.

---

### Solution A3: Receptive Field

**Step 1: Recursive formula**
Each layer with kernel $K$ and stride 1 adds $K-1$ to the receptive field:
$$RF_l = RF_{l-1} + (K - 1)$$

Base case: $RF_0 = 1$ (single pixel)

**Step 2: For $L$ layers of 3×3**
$$RF_L = 1 + L \times (3 - 1) = 1 + 2L$$

Examples:
- 1 layer: RF = 3
- 2 layers: RF = 5
- 3 layers: RF = 7

**Step 3: Two 3×3 vs one 5×5**
- Two 3×3: $RF = 1 + 2(2) = 5$
- One 5×5: $RF = 1 + 1(4) = 5$ ✓

Same receptive field, but parameters:
- Two 3×3: $2 \times 9 = 18$
- One 5×5: $25$

Two 3×3 is more efficient!

---

### Solution A4: Backward Pass

Let $y_{ij} = \sum_{m,n} x_{i+m, j+n} \cdot w_{mn}$

**Weight gradient:**
$$\frac{\partial \mathcal{L}}{\partial w_{mn}} = \sum_{i,j} \frac{\partial \mathcal{L}}{\partial y_{ij}} \cdot x_{i+m, j+n}$$

This is a convolution of the upstream gradient with the input:
$$\boxed{\frac{\partial \mathcal{L}}{\partial W} = X * \frac{\partial \mathcal{L}}{\partial Y}}$$

**Input gradient:**
$$\frac{\partial \mathcal{L}}{\partial x_{pq}} = \sum_{i,j} \frac{\partial \mathcal{L}}{\partial y_{ij}} \cdot \frac{\partial y_{ij}}{\partial x_{pq}}$$

After working through indices:
$$\boxed{\frac{\partial \mathcal{L}}{\partial X} = \text{flip}(W) *_{full} \frac{\partial \mathcal{L}}{\partial Y}}$$

Where $*_{full}$ is "full" convolution (with appropriate padding) and flip rotates the kernel 180°.

---

### Solution A5: Translation Equivariance

**Part 1: Convolution is equivariant**

Let $T_t$ shift image by $t$ pixels. For convolution $(f * x)$:

$$(f * T_t(x))[i, j] = \sum_{m,n} f[m,n] \cdot T_t(x)[i-m, j-n]$$
$$= \sum_{m,n} f[m,n] \cdot x[i-m-t_x, j-n-t_y]$$
$$= (f * x)[i-t_x, j-t_y]$$
$$= T_t((f * x))[i, j]$$

Therefore: $f * T_t(x) = T_t(f * x)$ ✓

**Part 2: Max pooling**

Max pooling is NOT strictly equivariant.

Counterexample: 1D with pool size 2
- Input: [1, 3, 2, 4] → Pool → [3, 4]
- Shift by 1: [0, 1, 3, 2, 4] → Pool → [1, 3, 4]

The output shifts differently than input due to alignment with pool boundaries.

**Part 3: Implication**
The network will produce similar (but not identical) responses regardless of where an object appears. This is why CNNs can recognize objects without needing to see them in every possible position during training.

---

## Part B: Coding Solutions

### Solution B1: 2D Convolution

```python
import numpy as np

def conv2d(image, kernel, stride=1, padding=0):
    # Pad image
    if padding > 0:
        image = np.pad(image, padding, mode='constant')
    
    H, W = image.shape
    K = kernel.shape[0]
    
    # Output dimensions
    H_out = (H - K) // stride + 1
    W_out = (W - K) // stride + 1
    
    output = np.zeros((H_out, W_out))
    
    for i in range(H_out):
        for j in range(W_out):
            # Extract patch
            patch = image[i*stride:i*stride+K, j*stride:j*stride+K]
            # Element-wise multiply and sum
            output[i, j] = np.sum(patch * kernel)
    
    return output
```

---

### Solution B2: Multi-channel Convolution

```python
def conv2d_multichannel(x, w, b, stride=1, padding=0):
    C_in, H, W = x.shape
    C_out, _, K, _ = w.shape
    
    # Pad input
    if padding > 0:
        x_pad = np.pad(x, ((0, 0), (padding, padding), (padding, padding)), mode='constant')
    else:
        x_pad = x
    
    # Output dimensions
    H_out = (H + 2*padding - K) // stride + 1
    W_out = (W + 2*padding - K) // stride + 1
    
    output = np.zeros((C_out, H_out, W_out))
    
    for c_out in range(C_out):
        for i in range(H_out):
            for j in range(W_out):
                # Extract patch from all input channels
                patch = x_pad[:, i*stride:i*stride+K, j*stride:j*stride+K]
                # Convolve with filter for this output channel
                output[c_out, i, j] = np.sum(patch * w[c_out]) + b[c_out]
    
    return output
```

---

### Solution B3: Max Pooling

```python
class MaxPool2d:
    def __init__(self, kernel_size, stride=None):
        self.kernel_size = kernel_size
        self.stride = stride if stride else kernel_size
        self.max_indices = None
        self.input_shape = None
    
    def forward(self, x):
        self.input_shape = x.shape
        C, H, W = x.shape
        K = self.kernel_size
        S = self.stride
        
        H_out = (H - K) // S + 1
        W_out = (W - K) // S + 1
        
        output = np.zeros((C, H_out, W_out))
        self.max_indices = np.zeros((C, H_out, W_out, 2), dtype=int)
        
        for c in range(C):
            for i in range(H_out):
                for j in range(W_out):
                    patch = x[c, i*S:i*S+K, j*S:j*S+K]
                    max_idx = np.unravel_index(np.argmax(patch), patch.shape)
                    output[c, i, j] = patch[max_idx]
                    # Store absolute indices
                    self.max_indices[c, i, j] = [i*S + max_idx[0], j*S + max_idx[1]]
        
        return output
    
    def backward(self, grad_output):
        C, H, W = self.input_shape
        grad_input = np.zeros(self.input_shape)
        
        _, H_out, W_out = grad_output.shape
        
        for c in range(C):
            for i in range(H_out):
                for j in range(W_out):
                    max_i, max_j = self.max_indices[c, i, j]
                    grad_input[c, max_i, max_j] += grad_output[c, i, j]
        
        return grad_input
```

---

### Solution B4: Visualize Filter (Gradient Ascent)

```python
import torch
import torch.nn as nn

def visualize_filter(model, layer_idx, filter_idx, iterations=100, lr=1):
    model.eval()
    
    # Get the target layer
    layers = list(model.children())
    target_layer = layers[layer_idx]
    
    # Random starting image
    img = torch.randn(1, 3, 224, 224, requires_grad=True)
    
    for _ in range(iterations):
        # Forward pass up to target layer
        x = img
        for i, layer in enumerate(layers):
            x = layer(x)
            if i == layer_idx:
                break
        
        # Activation of target filter
        activation = x[0, filter_idx].mean()
        
        # Backward to input
        activation.backward()
        
        # Gradient ascent
        img.data += lr * img.grad.data
        img.grad.data.zero_()
        
        # Regularization: clip values
        img.data = torch.clamp(img.data, -1, 1)
    
    # Normalize for visualization
    result = img.detach().squeeze().permute(1, 2, 0).numpy()
    result = (result - result.min()) / (result.max() - result.min())
    
    return result
```

---

### Solution B5: Depthwise Separable Convolution

```python
import numpy as np

class DepthwiseSeparableConv:
    def __init__(self, in_channels, out_channels, kernel_size):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        
        # Depthwise: one filter per input channel
        self.depthwise_weights = np.random.randn(in_channels, kernel_size, kernel_size) * 0.1
        self.depthwise_bias = np.zeros(in_channels)
        
        # Pointwise: 1x1 convolution
        self.pointwise_weights = np.random.randn(out_channels, in_channels) * 0.1
        self.pointwise_bias = np.zeros(out_channels)
    
    def forward(self, x):
        C, H, W = x.shape
        K = self.kernel_size
        
        # Depthwise convolution
        H_out = H - K + 1
        W_out = W - K + 1
        depthwise_out = np.zeros((C, H_out, W_out))
        
        for c in range(C):
            for i in range(H_out):
                for j in range(W_out):
                    patch = x[c, i:i+K, j:j+K]
                    depthwise_out[c, i, j] = np.sum(patch * self.depthwise_weights[c]) + self.depthwise_bias[c]
        
        # Pointwise convolution (1x1)
        pointwise_out = np.zeros((self.out_channels, H_out, W_out))
        
        for c_out in range(self.out_channels):
            pointwise_out[c_out] = np.sum(
                depthwise_out * self.pointwise_weights[c_out, :, np.newaxis, np.newaxis],
                axis=0
            ) + self.pointwise_bias[c_out]
        
        return pointwise_out
    
    def count_parameters(self):
        depthwise = self.in_channels * self.kernel_size * self.kernel_size
        pointwise = self.in_channels * self.out_channels
        total = depthwise + pointwise + self.in_channels + self.out_channels
        
        standard = self.in_channels * self.out_channels * self.kernel_size * self.kernel_size + self.out_channels
        
        print(f"Depthwise separable: {total}")
        print(f"Standard convolution: {standard}")
        print(f"Reduction factor: {standard / total:.1f}x")
        
        return total

# Example
dsc = DepthwiseSeparableConv(64, 128, 3)
dsc.count_parameters()
```

---

## Part C: Conceptual Answers

### C1: Benefits of Weight Sharing
1. **Parameter efficiency**: Same filter used everywhere, far fewer parameters than FC
2. **Translation equivariance**: Features detected regardless of position in image
3. **Faster training**: Fewer parameters = less computation = faster convergence
4. **Better generalization**: Fewer parameters = less overfitting
5. **Inductive bias**: Encodes prior that local patterns matter, which suits images

### C2: Feature Hierarchy
- **Early layers**: Low-level features - edges, colors, textures, gradients
- **Middle layers**: Mid-level features - corners, simple shapes, textures, object parts
- **Late layers**: High-level features - object parts, complete objects, semantic concepts

This hierarchy mimics the visual cortex (V1 → V2 → V4 → IT).

### C3: Small vs Large Filters
1. **Same receptive field, fewer parameters**: Two 3×3 = 18 params, one 5×5 = 25 params
2. **More non-linearities**: Each layer adds a ReLU, more expressive
3. **Easier to train**: Smaller filters have better gradient flow
4. **More flexible**: Can learn more complex patterns with multiple layers

### C4: Convolution Types
a) **Valid**: No padding, output smaller than input. $H_{out} = H - K + 1$
b) **Same**: Padding to keep output size = input size. $P = (K-1)/2$
c) **Full**: Enough padding that each input pixel affects each output position. $P = K - 1$

### C5: Information-Theoretic View
A convolutional filter can be seen as computing mutual information between the filter pattern and local image patches:
- High activation = high MI = patch similar to filter pattern
- The filter learns to maximize MI with patterns that help the task
- Weight sharing means the same "information detector" scans the entire image

### C6: Strided Conv vs Pooling
**Strided convolution:**
- ✓ Learnable downsampling
- ✓ Combines feature extraction and downsampling
- ✗ More parameters
- ✗ Can create checkerboard artifacts

**Pooling (max/average):**
- ✓ No parameters (efficient)
- ✓ Built-in invariance (max pool)
- ✗ Fixed operation, not learned
- ✗ Loses spatial information

Modern trend: prefer strided convolutions for better gradient flow and learned downsampling.
