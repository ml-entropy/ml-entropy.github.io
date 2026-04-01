# Tutorial 27: Amortized Variational Inference Exercises

## Easy Problems

1. Define amortized variational inference.
2. What two ideas are combined in AVI?
3. Write the standard ELBO used in amortized variational inference.
4. What is the role of the encoder in AVI?
5. What is the role of the decoder in AVI?
6. Why is AVI central to VAEs?
7. What is the reparameterization trick used for?
8. What is one major benefit of AVI over per-datapoint VI?
9. What is one major source of error introduced by AVI?
10. Why can a powerful decoder cause problems in AVI?

## Middle Problems

1. Compare amortized VI with classical variational inference using local variational parameters.
2. Explain how AVI learns an approximate posterior map from data to latent distributions.
3. Describe how the ELBO trains both the generative model and the inference model.
4. Explain why reparameterization is useful for gradient-based AVI.
5. Distinguish posterior collapse from the amortization gap.
6. Explain why AVI is especially attractive in large datasets and online settings.
7. Compare AVI with MCMC for test-time latent inference.
8. Describe one reason a simple encoder may fail even if the decoder is strong.
9. Explain how KL annealing can help AVI-based models.
10. Describe the role of shared parameters in generalization across datapoints.

## Hard Problems

1. Derive the dataset-level AVI objective as a sum of per-example ELBO terms.
2. Explain why AVI can be interpreted as learning a fast approximation to Bayes' rule.
3. Compare approximation gap, optimization gap, and amortization gap in a single framework.
4. Discuss why expressive decoders can cause latent variables to be ignored.
5. Explain how semi-amortized inference attempts to reduce encoder bias.
6. Compare AVI with expectation-maximization in latent-variable models.
7. Analyze how posterior family choice and encoder architecture jointly determine inference quality.
8. Describe how AVI behaves under test-time distribution shift.
9. Explain why fast amortized inference can still produce poorly calibrated uncertainty.
10. Write a short essay on AVI as the computational engine behind modern deep probabilistic modeling.

