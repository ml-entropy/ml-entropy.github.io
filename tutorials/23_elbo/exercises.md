# Tutorial 23: ELBO Exercises

## Easy Problems

1. What does ELBO stand for?
2. Why is the ELBO called a lower bound?
3. Write the definition of the ELBO using $p(x,z)$ and $q(z \mid x)$.
4. What inequality is used to derive the ELBO?
5. What is the common decomposition of the ELBO in latent-variable models?
6. Which KL divergence appears in the gap between log evidence and ELBO?
7. Why can the ELBO be optimized even when $\log p(x)$ is hard to compute?
8. What role does the prior play in the ELBO decomposition?
9. What does the reconstruction term encourage?
10. What does the KL regularizer encourage?

## Middle Problems

1. Derive the ELBO starting from $\log p(x)$.
2. Explain in words why Jensen's inequality produces a lower bound.
3. Show how the ELBO becomes $E_q[\log p(x \mid z)] - D_{KL}(q\|p(z))$.
4. Explain why maximizing the ELBO improves the variational approximation.
5. Describe how the ELBO balances data fit and regularization.
6. Explain why a restricted variational family can leave a nonzero gap even at optimum.
7. Compare optimizing the ELBO with directly minimizing $D_{KL}(q\|p(z \mid x))$.
8. Give an example of a model where the evidence integral is intractable.
9. Explain the connection between the ELBO and representation learning in VAEs.
10. Describe one failure mode caused by overemphasizing the KL term.

## Hard Problems

1. Prove the identity $\log p(x)=\mathcal{L}(q)+D_{KL}(q\|p(z \mid x))$.
2. Explain why optimizing reverse KL often leads to mode-seeking approximations.
3. Derive the ELBO for a Gaussian latent-variable model with linear decoder.
4. Compare the ELBO to importance sampling lower bounds conceptually.
5. Explain how posterior collapse can be understood directly from the ELBO terms.
6. Discuss how the ELBO changes when the prior itself has learnable parameters.
7. Show how a beta-VAE modifies the ELBO and interpret the effect of $\beta$.
8. Give an example where a high ELBO still corresponds to a poor posterior approximation for a downstream task.
9. Explain the difference between maximizing average ELBO over data and tightening the bound per observation.
10. Write a short essay on the ELBO as a compromise between statistical fidelity and computational tractability.

