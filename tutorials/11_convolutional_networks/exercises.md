# Tutorial 12: Convolutional Neural Networks - Exercises

## Part A: Theory Derivations

### Exercise A1 🟢 (Easy)
**Output size formula**

Derive the output size formula for a convolution:
- Input: $H \times W$
- Kernel: $K \times K$
- Stride: $S$
- Padding: $P$

1. How many positions can the kernel be placed horizontally?
2. Write the general formula: $H_{out} = ?$

---

### Exercise A2 🟢 (Easy)
**Parameter counting**

For a convolutional layer:
- Input channels: $C_{in}$
- Output channels: $C_{out}$
- Kernel size: $K \times K$

1. How many parameters in the weight tensor?
2. How many bias parameters?
3. Compare to a fully connected layer with same input/output sizes

---

### Exercise A3 🟡 (Medium)
**Receptive field derivation**

For a network with $L$ convolutional layers, each with kernel size $K$ and stride $S=1$:

1. Show that the receptive field after layer $l$ is $RF_l = RF_{l-1} + (K-1)$
2. For $L$ layers of $3 \times 3$ convolutions, what is the total receptive field?
3. Prove that two $3 \times 3$ layers have the same receptive field as one $5 \times 5$ layer

---

### Exercise A4 🟡 (Medium)
**Backward pass through convolution**

Given forward pass: $y = x * w$ (convolution), derive:
1. $\frac{\partial \mathcal{L}}{\partial w}$ (weight gradient)
2. $\frac{\partial \mathcal{L}}{\partial x}$ (input gradient)

Hint: The gradient w.r.t. weights is also a convolution. The gradient w.r.t. input is a "full" convolution with flipped kernel.

---

### Exercise A5 🔴 (Hard)
**Prove translation equivariance**

A function $f$ is translation equivariant if: $f(T_t(x)) = T_t(f(x))$
where $T_t$ is translation by $t$ pixels.

1. Prove that convolution is translation equivariant
2. Is max pooling translation equivariant? Prove or give counterexample
3. What does this mean for object detection in different image locations?

---

## Part B: Coding Exercises

### Exercise B1 🟢 (Easy)
**Implement 2D convolution**

```python
def conv2d(image, kernel, stride=1, padding=0):
    """
    2D convolution (single channel).
    
    Args:
        image: Input image, shape (H, W)
        kernel: Convolution kernel, shape (K, K)
        stride: Stride of convolution
        padding: Zero-padding size
    
    Returns:
        Output feature map, shape (H_out, W_out)
    """
    # YOUR CODE HERE
    pass
```

---

### Exercise B2 🟡 (Medium)
**Implement multi-channel convolution**

```python
def conv2d_multichannel(x, w, b, stride=1, padding=0):
    """
    Multi-channel 2D convolution.
    
    Args:
        x: Input, shape (C_in, H, W)
        w: Weights, shape (C_out, C_in, K, K)
        b: Bias, shape (C_out,)
        stride: Stride
        padding: Padding
    
    Returns:
        Output, shape (C_out, H_out, W_out)
    """
    # YOUR CODE HERE
    pass
```

---

### Exercise B3 🟡 (Medium)
**Implement max pooling with backward pass**

```python
class MaxPool2d:
    def __init__(self, kernel_size, stride=None):
        """
        Max pooling layer.
        
        Args:
            kernel_size: Size of pooling window
            stride: Stride (default = kernel_size)
        """
        # YOUR CODE HERE
        pass
    
    def forward(self, x):
        """
        Forward pass. Save indices of max values for backward.
        """
        # YOUR CODE HERE
        pass
    
    def backward(self, grad_output):
        """
        Backward pass. Route gradients only to max positions.
        """
        # YOUR CODE HERE
        pass
```

---

### Exercise B4 🔴 (Hard)
**Visualize learned features with gradient ascent**

```python
def visualize_filter(model, layer_idx, filter_idx, iterations=100, lr=1):
    """
    Generate image that maximally activates a specific filter.
    
    1. Start with random noise image
    2. Forward pass to get activation of target filter
    3. Compute gradient of activation w.r.t. input
    4. Update input to increase activation
    5. Return optimized image
    
    Args:
        model: Trained CNN
        layer_idx: Which layer to visualize
        filter_idx: Which filter in that layer
        iterations: Optimization steps
        lr: Learning rate for gradient ascent
    
    Returns:
        Optimized input image that maximizes filter activation
    """
    # YOUR CODE HERE
    pass
```

---

### Exercise B5 🔴 (Hard)
**Implement depthwise separable convolution**

```python
class DepthwiseSeparableConv:
    """
    Depthwise separable convolution (used in MobileNet).
    
    Two steps:
    1. Depthwise: Apply one filter per input channel
    2. Pointwise: 1x1 convolution to combine channels
    
    Reduces parameters from C_in * C_out * K * K to:
    C_in * K * K + C_in * C_out
    """
    
    def __init__(self, in_channels, out_channels, kernel_size):
        # YOUR CODE HERE
        pass
    
    def forward(self, x):
        # YOUR CODE HERE
        pass
    
    def count_parameters(self):
        """Compare to standard convolution."""
        # YOUR CODE HERE
        pass
```

---

## Part C: Conceptual Questions

### C1 🟢
Why is weight sharing in CNNs beneficial? List at least 3 reasons.

### C2 🟢
Explain the hierarchy of features learned by deep CNNs:
- Early layers learn: ?
- Middle layers learn: ?
- Late layers learn: ?

### C3 🟡
Why use multiple small filters (e.g., 3×3) instead of one large filter (e.g., 7×7)?

### C4 🟡
Explain the difference between:
a) Valid convolution
b) Same convolution
c) Full convolution

### C5 🔴
From an information-theoretic perspective, what does a convolutional filter compute? (Hint: Think about mutual information between the filter and local image patches)

### C6 🔴
Modern architectures often use strided convolutions instead of pooling. What are the pros and cons of each approach?
