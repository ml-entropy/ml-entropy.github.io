# Tutorial 26: Variational Inference Exercises

## Easy Problems

1. Define variational inference.
2. What optimization problem does variational inference solve?
3. Why is the ELBO used in variational inference?
4. What is a variational family?
5. What is mean-field factorization?
6. What does reverse KL mean here?
7. Give one reason VI is often faster than MCMC.
8. What is coordinate-ascent variational inference?
9. What is one downside of mean-field assumptions?
10. What is black-box variational inference?

## Middle Problems

1. Explain in words how VI turns integration into optimization.
2. Describe the difference between approximation error and optimization error in VI.
3. Explain why reverse KL can produce mode-seeking behavior.
4. Compare mean-field VI and full-covariance approximations.
5. Explain why VI often underestimates posterior variance.
6. Describe when coordinate-ascent updates are available.
7. Compare VI and MCMC in terms of speed, bias, and scalability.
8. Explain how stochastic gradient methods enable large-scale VI.
9. Give an example where VI is a practical choice despite approximation bias.
10. Explain why the choice of variational family is as important as the optimizer.

## Hard Problems

1. Derive the mean-field coordinate update formula up to an additive constant.
2. Explain carefully why reverse KL and forward KL behave differently near missing modes.
3. Show how VI can be derived either from KL minimization or ELBO maximization.
4. Compare deterministic coordinate-ascent VI with stochastic black-box VI.
5. Discuss how richer variational families trade off flexibility against computational cost.
6. Explain why a poor ELBO landscape can create local-optimum issues in VI.
7. Describe how reparameterization gradients reduce variance compared to score-function estimators.
8. Analyze a case where VI gives good posterior means but poor uncertainty calibration.
9. Compare VI for conjugate exponential-family models with VI for deep generative models.
10. Write a short essay on variational inference as approximate Bayesian computation by optimization.

