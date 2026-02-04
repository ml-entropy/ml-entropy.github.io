def generate_exercise_html(id, title, difficulty, question, solution):
    diff_class = f"diff-{difficulty.lower()}"
    return f"""
                <div class="exercise-card">
                    <div class="exercise-header">
                        <h3>{id}. {title}</h3>
                        <span class="difficulty-badge {diff_class}">{difficulty}</span>
                    </div>
                    {question}
                    <details>
                        <summary>Show Solution</summary>
                        <div class="solution-content">
                            {solution}
                        </div>
                    </details>
                </div>"""

exercises = []

# --- EASY EXERCISES ---

exercises.append(("E1", "Entropy of Uniform Distribution", "Easy",
    r"<p>Calculate the entropy (in bits) of a uniform distribution over 16 outcomes.</p>",
    r"<p>$H(X) = \log_2(N) = \log_2(16) = 4$ bits.</p>"
))

exercises.append(("E2", "Surprise Calculation", "Easy",
    r"<p>An event has a probability of $p=0.01$. What is the 'surprise' or information content (in bits) of observing this event?</p>",
    r"<p>$I(x) = -\log_2(0.01) \approx 6.64$ bits.</p>"
))

exercises.append(("E3", "Deterministic Entropy", "Easy",
    r"<p>What is the entropy of a random variable that always takes the value 5?</p>",
    r"<p>$H(X) = 0$. There is no uncertainty.</p>"
))

exercises.append(("E4", "Bits vs Nats", "Easy",
    r"<p>If an event has 1 bit of information, how many 'nats' of information does it have? (Hint: $\ln 2 \approx 0.693$)</p>",
    r"<p>1 bit = $\ln 2$ nats $\approx 0.693$ nats.</p>"
))

exercises.append(("E5", "Independent Events", "Easy",
    r"<p>You flip a fair coin 3 times. What is the total entropy of the sequence of outcomes?</p>",
    r"<p>Since flips are independent, $H(X_1, X_2, X_3) = H(X_1) + H(X_2) + H(X_3) = 1 + 1 + 1 = 3$ bits.</p>"
))

exercises.append(("E6", "Specific Distribution", "Easy",
    r"<p>Calculate the entropy of $P = [0.5, 0.25, 0.25]$.</p>",
    r"""<p>
    $$H(P) = -(0.5 \log_2 0.5 + 0.25 \log_2 0.25 + 0.25 \log_2 0.25)$$
    $$= - (0.5(-1) + 0.25(-2) + 0.25(-2))$$
    $$= 0.5 + 0.5 + 0.5 = 1.5 \text{ bits}$$
    </p>"""
))

exercises.append(("E7", "Max Entropy Binary", "Easy",
    r"<p>For a binary random variable with $P(X=1) = p$, for what value of $p$ is entropy maximized?</p>",
    r"<p>$p=0.5$. The uniform distribution maximizes entropy.</p>"
))

exercises.append(("E8", "Cross-Entropy Identity", "Easy",
    r"<p>If the predicted distribution $Q$ is exactly equal to the true distribution $P$, what is the cross-entropy $H(P, Q)$ equal to?</p>",
    r"<p>It is equal to the entropy $H(P)$. Since $D_{KL}(P||Q) = 0$.</p>"
))

exercises.append(("E9", "KL Divergence Minimum", "Easy",
    r"<p>What is the minimum possible value for KL Divergence $D_{KL}(P||Q)$?</p>",
    r"<p>0. It is always non-negative (Gibbs' inequality).</p>"
))

exercises.append(("E10", "Negative Entropy?", "Easy",
    r"<p>True or False: The entropy of a discrete random variable can be negative.</p>",
    r"<p>False. Since $0 \le p \le 1$, $\log p \le 0$, so $-\sum p \log p \ge 0$. (Note: Differential entropy for continuous variables <em>can</em> be negative).</p>"
))

# --- MEDIUM EXERCISES ---

exercises.append(("M1", "Joint Entropy", "Medium",
    r"<p>Let $X$ be the outcome of a fair coin flip (H/T). Let $Y=X$ (copy of X). What is the joint entropy $H(X, Y)$?</p>",
    r"""<p>Since $X$ and $Y$ are perfectly correlated (dependent):</p>
    <p>$H(X, Y) = H(X) + H(Y|X) = H(X) + 0 = 1$ bit.</p>
    <p>Alternatively, there are only 2 possible joint outcomes: (H,H) and (T,T), each with prob 0.5.</p>"""
))

exercises.append(("M2", "Conditional Entropy", "Medium",
    r"<p>If $Y = f(X)$ is a deterministic function of $X$, what is $H(Y|X)$?</p>",
    r"<p>$H(Y|X) = 0$. If you know $X$, you know $Y$ exactly, so there is no remaining uncertainty.</p>"
))

exercises.append(("M3", "Huffman Lower Bound", "Medium",
    r"<p>Can the average length of a Huffman code be strictly less than the entropy $H(X)$?</p>",
    r"<p>No. Shannon's Source Coding Theorem states $L \ge H(X)$. It can only be equal if all probabilities are powers of 2.</p>"
))

exercises.append(("M4", "KL Asymmetry", "Medium",
    r"<p>Let $P=[1, 0]$ and $Q=[0.5, 0.5]$. Calculate $D_{KL}(P||Q)$ and $D_{KL}(Q||P)$. What does this show?</p>",
    r"""<p>
    $D_{KL}(P||Q) = 1 \log_2(1/0.5) + 0 = 1$ bit.<br>
    $D_{KL}(Q||P) = 0.5 \log_2(0.5/1) + 0.5 \log_2(0.5/0) = \infty$ (undefined/infinite).<br>
    This shows KL Divergence is <strong>not symmetric</strong>.
    </p>"""
))

exercises.append(("M5", "Cross-Entropy vs Entropy", "Medium",
    r"<p>Prove that Cross-Entropy $H(P, Q)$ is always greater than or equal to Entropy $H(P)$.</p>",
    r"<p>$H(P, Q) = H(P) + D_{KL}(P||Q)$. Since $D_{KL} \ge 0$, it follows that $H(P, Q) \ge H(P)$.</p>"
))

exercises.append(("M6", "Entropy of Sum", "Medium",
    r"<p>Let $X_1, X_2$ be independent fair binary variables (0 or 1). Let $Y = X_1 + X_2$. What is $H(Y)$?</p>",
    r"""<p>Possible values for Y: 0, 1, 2.<br>
    P(0) = 0.25 (0+0)<br>
    P(1) = 0.50 (0+1 or 1+0)<br>
    P(2) = 0.25 (1+1)<br>
    $H(Y) = -(0.25 \log 0.25 + 0.5 \log 0.5 + 0.25 \log 0.25) = -(-0.5 - 0.5 - 0.5) = 1.5$ bits.</p>"""
))

exercises.append(("M7", "Mutual Information", "Medium",
    r"<p>Define Mutual Information $I(X; Y)$ in terms of Entropy $H(X)$ and Conditional Entropy $H(X|Y)$. Explain intuitively.</p>",
    r"<p>$I(X; Y) = H(X) - H(X|Y)$. It represents the reduction in uncertainty about $X$ gained by observing $Y$.</p>"
))

exercises.append(("M8", "Perplexity", "Medium",
    r"<p>In NLP, models are evaluated on 'Perplexity', defined as $2^{H(P)}$. If a model has a perplexity of 8, what does this intuitively mean?</p>",
    r"<p>It means the model is as confused as if it were choosing uniformly at random from 8 equally likely words.</p>"
))

exercises.append(("M9", "Chain Rule", "Medium",
    r"<p>Write the Chain Rule for Entropy for three variables $H(X, Y, Z)$.</p>",
    r"<p>$H(X, Y, Z) = H(X) + H(Y|X) + H(Z|X, Y)$. Uncertainty sums up: uncertainty of X, plus uncertainty of Y given X, etc.</p>"
))

exercises.append(("M10", "Concavity", "Medium",
    r"<p>Is the entropy function $H(p)$ concave or convex? Why is this important for optimization?</p>",
    r"<p>It is <strong>concave</strong>. This guarantees that a local maximum (uniform distribution) is the global maximum.</p>"
))

# --- HARD EXERCISES ---

exercises.append(("H1", "Geometric Distribution", "Hard",
    r"<p>Derive the entropy of a geometric distribution $P(k) = (1-p)^{k-1}p$ for $k=1, 2, \dots$.</p>",
    r"""<p>
    $H(X) = -\sum p(k) \log p(k) = -\sum (1-p)^{k-1}p [\log p + (k-1)\log(1-p)]$
    After simplifications using expected value $E[X] = 1/p$:
    $$H(X) = \frac{-(1-p)\log_2(1-p) - p\log_2 p}{p} \text{ bits}$$
    </p>"""
))

exercises.append(("H2", "Independence Bound", "Hard",
    r"<p>Prove that $H(X, Y) \le H(X) + H(Y)$. When does equality hold?</p>",
    r"<p>This follows from $I(X; Y) \ge 0$. Since $I(X; Y) = H(X) + H(Y) - H(X, Y)$, non-negativity implies $H(X, Y) \le H(X) + H(Y)$. Equality holds iff X and Y are independent.</p>"
))

exercises.append(("H3", "Differential Entropy Uniform", "Hard",
    r"<p>Calculate the differential entropy of a Continuous Uniform distribution on $[0, a]$.</p>",
    r"""<p>
    PDF $p(x) = 1/a$ for $0 \le x \le a$.
    $$h(X) = -\int_0^a \frac{1}{a} \log \frac{1}{a} dx = - \log \frac{1}{a} \int_0^a \frac{1}{a} dx = \log a$$
    Note: If $a < 1$, entropy is negative!
    </p>"""
))

exercises.append(("H4", "Exponential Entropy", "Hard",
    r"<p>Calculate the differential entropy of an Exponential distribution $\lambda e^{-\lambda x}$.</p>",
    r"""<p>
    $h(X) = 1 - \ln \lambda$ (nats).
    Derivation involves $\ln p(x) = \ln \lambda - \lambda x$ and taking expectation.
    </p>"""
))

exercises.append(("H5", "Gaussian KL", "Hard",
    r"<p>Write the formula for KL Divergence between two univariate Gaussians $P \sim \mathcal{N}(\mu_1, \sigma_1^2)$ and $Q \sim \mathcal{N}(\mu_2, \sigma_2^2)$.</p>",
    r"""<p>
    $$D_{KL}(P||Q) = \ln\frac{\sigma_2}{\sigma_1} + \frac{\sigma_1^2 + (\mu_1 - \mu_2)^2}{2\sigma_2^2} - \frac{1}{2}$$
    This is crucial for Variational Autoencoders (VAEs).
    </p>"""
))

exercises.append(("H6", "Max Entropy Variance", "Hard",
    r"<p>Why does the Gaussian distribution have the maximum entropy among all distributions with a fixed mean and variance?</p>",
    r"<p>Using Lagrange multipliers with constraints $\int p(x)dx=1$, $\int xp(x)dx=\mu$, $\int (x-\mu)^2 p(x)dx=\sigma^2$ yields the form of the Gaussian PDF.</p>"
))

exercises.append(("H7", "MLE vs KL", "Hard",
    r"<p>Formal proof: Show that maximizing Likelihood is equivalent to minimizing KL divergence from the empirical distribution to the model.</p>",
    r"""<p>
    Let $P_{data}$ be the empirical distribution. Max $\sum \log Q(x_i)$ is equivalent to Max $E_{P_{data}}[\log Q(x)]$.
    Min $D_{KL}(P_{data}||Q) = \sum P_{data} \log P_{data} - \sum P_{data} \log Q$.
    The first term is constant wrt Q. So Min KL $\iff$ Max $\sum P_{data} \log Q$.
    </p>"""
))

exercises.append(("H8", "Entropy Rate", "Hard",
    r"<p>For a stochastic process (like a Markov chain), what is the Entropy Rate?</p>",
    r"<p>$\lim_{n \to \infty} \frac{1}{n} H(X_1, \dots, X_n)$. It measures the average new information per step in the long run.</p>"
))

exercises.append(("H9", "Data Processing Inequality", "Hard",
    r"<p>State the Data Processing Inequality for a Markov chain $X \to Y \to Z$.</p>",
    r"<p>$I(X; Z) \le I(X; Y)$. Processing data (Y to Z) cannot create new information about the source X.</p>"
))

exercises.append(("H10", "Fano's Inequality", "Hard",
    r"<p>What does Fano's Inequality relate?</p>",
    r"<p>It relates the probability of error $P_e$ in estimating $X$ from $Y$ to the conditional entropy $H(X|Y)$. It sets a lower bound on error probability based on uncertainty.</p>"
))

# Generate HTML
html_output = '''            <!-- Exercises Section -->
            <section id="exercises" style="padding-top: 2rem; margin-top: 2rem; border-top: 1px solid #e5e7eb;">
                <h2 style="margin-bottom: 1.5rem;">30 Practice Exercises</h2>'''

for ex in exercises:
    html_output += generate_exercise_html(*ex)

html_output += '\n            </section>'

print(html_output)