# Tutorial 25: Amortized Inference Exercises

## Easy Problems

1. Define amortized inference.
2. What is the main computational motivation for amortized inference?
3. What role does an encoder network play in amortized inference?
4. Write the mapping implemented by amortized inference in a latent-variable model.
5. What is meant by shared parameters across datapoints?
6. Give one model family that uses amortized inference.
7. Why is amortized inference useful at test time?
8. What is the amortization gap?
9. What is the difference between local variational parameters and shared encoder parameters?
10. Why can amortized inference be faster than per-datapoint optimization?

## Middle Problems

1. Compare classical local variational inference with amortized inference.
2. Explain how amortization changes the computational profile of inference on large datasets.
3. Describe a setting where amortized inference may perform poorly on rare examples.
4. Explain the distinction between approximation gap and amortization gap.
5. Give an example of iterative refinement improving an amortized posterior approximation.
6. Explain why an expressive encoder alone may not eliminate all inference error.
7. Discuss how shared parameters can transfer knowledge across datapoints.
8. Compare amortized inference to caching or memoization conceptually.
9. Explain why amortized inference is especially useful in deployed probabilistic systems.
10. Describe how training objectives indirectly teach the encoder to perform inference.

## Hard Problems

1. Formalize the amortization gap in terms of optimal local variational parameters versus encoder outputs.
2. Explain why amortized inference may underfit difficult posterior geometries.
3. Compare fully amortized, semi-amortized, and non-amortized inference.
4. Discuss how distribution shift at test time affects amortized inference quality.
5. Explain how encoder architecture constrains the family of posterior approximations actually realized.
6. Derive how shared encoder parameters couple the optimization problems of different datapoints.
7. Argue why amortized inference can be viewed as learning an algorithm.
8. Compare amortized inference in VAEs to learned proposal distributions in sequential Monte Carlo.
9. Discuss when the speed gain of amortization is worth the bias it introduces.
10. Write a short essay on amortized inference as a tradeoff between specialization and reuse.
