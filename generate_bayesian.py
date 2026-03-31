import re

with open("docs/tutorials/ml/16-bayesian-inference/index.html", "r") as f:
    content = f.read()

# Find everything before <main class="tutorial-main">
pre_main = content.split("<main class=\"tutorial-main\">")[0] + "<main class=\"tutorial-main\">\n"

# Find everything after </main>
post_main = "\n        </main>" + content.split("</main>")[1]

# Now we need to update the title, description, and breadcrumbs in pre_main
pre_main = re.sub(r"<title>.*?</title>", "<title>Bayesian Inference | ML Fundamentals</title>", pre_main)
pre_main = re.sub(r'<meta name="description" content=".*?">', '<meta name="description" content="Learn Bayesian Inference from first principles: Bayes theorem, priors, posteriors, conjugate priors, and MAP estimation.">', pre_main)
pre_main = re.sub(r"<span>Autoencoders</span>", "<span>Bayesian Inference</span>", pre_main)

# We also need to remove the autoencoders sidebar active link, but update_sidebars.py handles the sidebar.
new_main_content = """
            <div id="theory" class="tab-content active">
                <header class="tutorial-header">
                    <h1>16. Bayesian Inference</h1>
                    <p class="tutorial-subtitle">
                        Moving from point estimates to reasoning with uncertainty using Bayes theorem.
                    </p>
                </header>

                <div class="content-section">
                    <h2 id="why-bayesian">Why Bayesian Inference?</h2>
                    <p>In traditional frequentist statistics and maximum likelihood estimation, we treat parameters as fixed but unknown quantities. The goal is to find the single best set of parameters that explain the data. But what if the data is scarce, noisy, or ambiguous?</p>
                    <p>Bayesian inference flips this paradigm. Instead of a single "best" parameter, we maintain a <em>distribution</em> over all possible parameters. We start with prior beliefs, and as we observe data, we update these beliefs according to the laws of probability. This allows models to express <em>uncertainty</em>, knowing when they don't know the answer.</p>
                    <p>This is crucial in domains like medicine, autonomous driving, and active learning, where overconfidence can be fatal, and knowing the limits of your knowledge is as important as the prediction itself.</p>

                    <h2 id="bayes-theorem">Bayes Theorem for Parameters</h2>
                    <p>The core of Bayesian inference is Bayes' Theorem, applied to a set of parameters $\\theta$ and data $\\mathcal{D}$:</p>
                    <div class="math-derivation">
                        <p>Starting from the joint probability $P(\\theta, \\mathcal{D})$, we can write it in two ways using the product rule:</p>
                        $$P(\\theta, \\mathcal{D}) = P(\\mathcal{D} | \\theta)P(\\theta) = P(\\theta | \\mathcal{D})P(\\mathcal{D})$$
                        <p>Solving for the posterior $P(\\theta | \\mathcal{D})$ gives Bayes theorem:</p>
                        $$P(\\theta | \\mathcal{D}) = \\frac{P(\\mathcal{D} | \\theta)P(\\theta)}{P(\\mathcal{D})}$$
                    </div>
                    <p>Each term has a distinct and intuitive meaning:</p>
                    <ul>
                        <li><strong>Prior $P(\\theta)$</strong>: Our belief about the parameters before seeing any data.</li>
                        <li><strong>Likelihood $P(\\mathcal{D} | \\theta)$</strong>: How probable is the data given these parameters?</li>
                        <li><strong>Posterior $P(\\theta | \\mathcal{D})$</strong>: Our updated belief about the parameters after seeing the data.</li>
                        <li><strong>Evidence $P(\\mathcal{D})$</strong>: The total probability of the data under all possible parameters. It acts as a normalizing constant.</li>
                    </ul>

                    <h2 id="map-estimation">Maximum A Posteriori (MAP)</h2>
                    <p>Full Bayesian inference requires computing the full posterior, which involves the intractable integral $P(\\mathcal{D}) = \\int P(\\mathcal{D}|\\theta)P(\\theta)d\\theta$. When this is too hard, we can settle for a point estimate: the mode of the posterior distribution.</p>
                    <p>This is called Maximum A Posteriori (MAP) estimation:</p>
                    <div class="math-derivation">
                        $$\\theta_{MAP} = \\arg\\max_\\theta P(\\theta | \\mathcal{D})$$
                        <p>Taking the logarithm (which doesn't change the argmax) and ignoring $P(\\mathcal{D})$ since it doesn't depend on $\\theta$:</p>
                        $$\\theta_{MAP} = \\arg\\max_\\theta \\left[ \\log P(\\mathcal{D} | \\theta) + \\log P(\\theta) \\right]$$
                    </div>
                    <p>Notice how MAP connects to Maximum Likelihood Estimation (MLE). MLE is simply MAP with a uniform prior $P(\\theta) \propto 1$. Thus, regularization (like L2 weight decay) can be viewed as MAP estimation with a Gaussian prior on the weights!</p>
                </div>
                
                <div class="content-section">
                    <h2 id="predictive-distribution">Posterior Predictive Distribution</h2>
                    <p>The real power of Bayesian inference lies in making predictions for new, unseen data points, $x^*$. Instead of using a single set of parameters $\\theta$ to make this prediction, we integrate over all possible parameters, weighted by their posterior probability. This gives us the posterior predictive distribution:</p>
                    <div class="math-derivation">
                        $$P(y^* | x^*, \\mathcal{D}) = \\int P(y^* | x^*, \\theta) P(\\theta | \\mathcal{D}) d\\theta$$
                    </div>
                    <p>This distribution naturally incorporates our uncertainty about the parameters into our predictions. If the posterior $P(\\theta | \\mathcal{D})$ is sharp (we are very confident about the parameters), the predictive distribution will be similar to a point estimate prediction. However, if the posterior is broad (we are uncertain about the parameters), the predictive distribution will also be broader, reflecting this uncertainty.</p>
                    <p>In the context of machine learning, this is why Bayesian Neural Networks are often considered more robust to out-of-distribution inputs than standard networks. Standard networks make overconfident predictions on unfamiliar inputs, while Bayesian networks output a broad distribution of predictions, signaling that they don't know the answer.</p>
                </div>
                
                <div class="content-section">
                    <h2 id="conjugate-priors">Conjugate Priors</h2>
                    <p>Historically, computing the posterior was analytically impossible for most prior and likelihood pairs. However, a special class of priors called conjugate priors yield closed-form posterior distributions. A prior $P(\\theta)$ is conjugate to a likelihood $P(\\mathcal{D} | \\theta)$ if the resulting posterior $P(\\theta | \\mathcal{D})$ belongs to the same family of distributions as the prior.</p>
                    <p>For example, if the likelihood is a Binomial distribution, a Beta prior will result in a Beta posterior. This makes the Bayesian update step incredibly simple: we just add the observed successes and failures to the parameters of our Beta distribution.</p>
                    <div class="math-derivation">
                        <p>Let the likelihood be Binomial: $\\mathcal{D} \\sim \\text{Binomial}(N, \\theta)$</p>
                        <p>Let the prior be Beta: $\\theta \\sim \\text{Beta}(\\alpha, \\beta)$</p>
                        <p>Then the posterior is also Beta: $\\theta | \\mathcal{D} \\sim \\text{Beta}(\\alpha + N_{\\text{successes}}, \\beta + N_{\\text{failures}})$</p>
                    </div>
                    <p>While elegant, conjugate priors are often too restrictive for complex models. The rise of modern computational methods, like Markov Chain Monte Carlo (MCMC) and Variational Inference, has allowed us to use more expressive priors and likelihoods, leading to the current resurgence of Bayesian methods in machine learning.</p>
                </div>
            </div>
            
            <div id="exercises" class="tab-content">
                <div class="exercises-header">
                    <h2>Exercises & Progress</h2>
                    <p>Test your understanding of Bayesian inference concepts.</p>
                </div>
                <div class="exercises-container">
                    <div class="exercise-item easy" data-exercise="1">
                        <div class="exercise-header">
                            <span class="exercise-badge easy">🟢 Easy</span>
                            <span class="exercise-title">1. Prior, Likelihood, Posterior</span>
                        </div>
                        <div class="exercise-content">
                            <p>Describe in your own words the role of the prior, likelihood, and posterior in the context of a coin flip experiment.</p>
                        </div>
                    </div>
                    <div class="exercise-item medium" data-exercise="2">
                        <div class="exercise-header">
                            <span class="exercise-badge medium">🟡 Medium</span>
                            <span class="exercise-title">2. MAP and Regularization</span>
                        </div>
                        <div class="exercise-content">
                            <p>Show that Maximum A Posteriori (MAP) estimation with a zero-mean Gaussian prior on the parameters is equivalent to Maximum Likelihood Estimation (MLE) with L2 regularization.</p>
                        </div>
                    </div>
                </div>
            </div>
"""

# Now write it out
with open("docs/tutorials/ml/16-bayesian-inference/index.html", "w") as f:
    f.write(pre_main + new_main_content + post_main)
print("Created 16-bayesian-inference/index.html")