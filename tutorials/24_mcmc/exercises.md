# Tutorial 24: MCMC Exercises

## Easy Problems

1. What does MCMC stand for?
2. What is the goal of MCMC in Bayesian inference?
3. What is a Markov chain?
4. What is meant by the stationary distribution of a chain?
5. Write the Metropolis-Hastings acceptance probability.
6. What is burn-in?
7. What is mixing?
8. What is autocorrelation in an MCMC chain?
9. What is effective sample size?
10. Name one advantage of Gibbs sampling.

## Middle Problems

1. Explain why MCMC can be used when direct sampling from the posterior is impossible.
2. Describe in words how Metropolis-Hastings works.
3. Explain why rejected proposals are still part of the Markov chain.
4. Compare random-walk Metropolis and Gibbs sampling.
5. Explain why high autocorrelation reduces Monte Carlo efficiency.
6. Describe how proposal variance affects acceptance rate and exploration.
7. Explain why multimodal targets are hard for local proposals.
8. Describe two practical convergence diagnostics.
9. Compare MCMC samples to i.i.d. samples.
10. Explain why finite-chain MCMC output is only approximately distributed as the target.

## Hard Problems

1. Derive the Metropolis-Hastings acceptance rule from detailed balance.
2. Show that if the proposal is symmetric, the acceptance ratio simplifies.
3. Explain why irreducibility and aperiodicity matter for convergence.
4. Derive the Gibbs sampler for a two-variable joint distribution with tractable conditionals.
5. Discuss the bias-variance tradeoff involved in discarding burn-in samples.
6. Show how autocorrelation enters the variance of an MCMC estimator.
7. Explain why thinning usually does not improve statistical efficiency per computation.
8. Compare MCMC and importance sampling for high-dimensional posterior inference.
9. Describe how Hamiltonian Monte Carlo improves on random-walk behavior conceptually.
10. Write a short essay on why MCMC diagnostics can suggest but never prove convergence.

