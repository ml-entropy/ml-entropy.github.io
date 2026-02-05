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
    r"""<p>For a uniform distribution with $N$ outcomes, the probability of each outcome is $p = 1/N$.</p>
    <p>The entropy is given by:</p>
    $$H(X) = -\sum_{i=1}^N p_i \log_2(p_i) = -\sum_{i=1}^N \frac{1}{N} \log_2\left(\frac{1}{N}\right)$$
    $$= - \left( N \cdot \frac{1}{N} \cdot (-\log_2 N) \right) = \log_2 N$$
    <p>Plugging in $N=16$:</p>
    $$H(X) = \log_2(16) = \log_2(2^4) = 4 \text{ bits}$$"""
))

exercises.append(("E2", "Surprise Calculation", "Easy",
    r"<p>An event has a probability of $p=0.01$. What is the 'surprise' or information content (in bits) of observing this event?</p>",
    r"""<p>The surprise or self-information is defined as:</p>
    $$I(x) = -\log_2(p(x))$$
    <p>Substituting $p=0.01$:</p>
    $$I(x) = -\log_2(0.01) = \log_2(100)$$
    <p>Since $2^6 = 64$ and $2^7 = 128$, the value is between 6 and 7.</p>
    $$I(x) \approx 6.644 \text{ bits}$$"""
))

exercises.append(("E3", "Deterministic Entropy", "Easy",
    r"<p>What is the entropy of a random variable that always takes the value 5?</p>",
    r"""<p>If the variable always takes the value 5, then:</p>
    <ul>
        <li>$P(X=5) = 1$</li>
        <li>$P(X \neq 5) = 0$</li>
    </ul>
    <p>The entropy calculation is:</p>
    $$H(X) = - \sum p(x) \log_2 p(x) = - (1 \cdot \log_2 1 + 0) = - (1 \cdot 0) = 0 \text{ bits}$$
    <p><strong>Intuition:</strong> There is zero uncertainty. You know exactly what will happen, so there is no surprise.</p>"""
))

exercises.append(("E4", "Bits vs Nats", "Easy",
    r"<p>If an event has 1 bit of information, how many 'nats' of information does it have? (Hint: $\ln 2 \approx 0.693$)</p>",
    r"""<p>Information units depend on the base of the logarithm:</p>
    <ul>
        <li><strong>Bits:</strong> Base 2 ($I = -\log_2 p$)</li>
        <li><strong>Nats:</strong> Base $e$ ($I = -\ln p$)</li>
    </ul>
    <p>To convert from bits to nats, we use the change of base formula:</p>
    $$\ln x = \frac{\log_2 x}{\log_2 e} = \ln 2 \cdot \log_2 x$$
    <p>So, 1 bit corresponds to:</p>
    $$1 \text{ bit} \times \ln 2 \approx 0.693 \text{ nats}$$"""
))

exercises.append(("E5", "Independent Events", "Easy",
    r"<p>You flip a fair coin 3 times. What is the total entropy of the sequence of outcomes?</p>",
    r"""<p>Let $X_1, X_2, X_3$ be the outcomes of the three flips.</p>
    <p>For a fair coin, the entropy of a single flip is:</p>
    $$H(X_i) = -(0.5 \log_2 0.5 + 0.5 \log_2 0.5) = 1 \text{ bit}$$
    <p>Since the flips are <strong>independent</strong>, the joint entropy is the sum of individual entropies:</p>
    $$H(X_1, X_2, X_3) = H(X_1) + H(X_2) + H(X_3) = 1 + 1 + 1 = 3 \text{ bits}$$
    <p>Alternatively, there are $2^3=8$ equally likely sequences (HHH, HHT, ...). Entropy of uniform distribution over 8 items is $\log_2 8 = 3$ bits.</p>"""
))

exercises.append(("E6", "Specific Distribution", "Easy",
    r"<p>Calculate the entropy of $P = [0.5, 0.25, 0.25]$.</p>",
    r"""<p>Applying Shannon's formula $H(X) = -\sum p_i \log_2 p_i$:</p>
    <ul>
        <li>Term 1: $0.5 \log_2(0.5) = 0.5 \times (-1) = -0.5$</li>
        <li>Term 2: $0.25 \log_2(0.25) = 0.25 \times (-2) = -0.5$</li>
        <li>Term 3: $0.25 \log_2(0.25) = 0.25 \times (-2) = -0.5$</li>
    </ul>
    <p>Summing them up:</p>
    $$H(P) = -(-0.5 - 0.5 - 0.5) = 1.5 \text{ bits}$$"""
))

exercises.append(("E7", "Max Entropy Binary", "Easy",
    r"<p>For a binary random variable with $P(X=1) = p$, for what value of $p$ is entropy maximized?</p>",
    r"""<p>The binary entropy function is $H(p) = -p \log_2 p - (1-p) \log_2(1-p)$.</p>
    <p>To find the maximum, we can take the derivative with respect to $p$ and set it to 0, or rely on the property that entropy is maximized when the distribution is most uniform.</p>
    <p>For a binary variable, the most uniform distribution is $p = 1-p = 0.5$.</p>
    <p>At $p=0.5$, $H(p) = 1$ bit (maximum).</p>
    <p>At $p=0$ or $p=1$, $H(p) = 0$ bits (minimum).</p>"""
))

exercises.append(("E8", "Cross-Entropy Identity", "Easy",
    r"<p>If the predicted distribution $Q$ is exactly equal to the true distribution $P$, what is the cross-entropy $H(P, Q)$ equal to?</p>",
    r"""<p>The definition of Cross-Entropy is:</p>
    $$H(P, Q) = -\sum p(x) \log_2 q(x)$$
    <p>If $Q = P$, then $q(x) = p(x)$ for all $x$. Substituting this in:</p>
    $$H(P, P) = -\sum p(x) \log_2 p(x)$$
    <p>This is exactly the definition of <strong>Entropy $H(P)$</strong>.</p>
    <p>Also, recalling that $H(P, Q) = H(P) + D_{KL}(P||Q)$, if $P=Q$ then $D_{KL}=0$, leaving just $H(P)$.</p>"""
))

exercises.append(("E9", "KL Divergence Minimum", "Easy",
    r"<p>What is the minimum possible value for KL Divergence $D_{KL}(P||Q)$?</p>",
    r"""<p>The minimum value is <strong>0</strong>.</p>
    <p>This is known as <strong>Gibbs' Inequality</strong>.</p>
    <p>$D_{KL}(P||Q) \geq 0$ for any valid probability distributions $P$ and $Q$.</p>
    <p>Equality ($D_{KL} = 0$) holds if and only if $P = Q$ almost everywhere.</p>"""
))

exercises.append(("E10", "Negative Entropy?", "Easy",
    r"<p>True or False: The entropy of a discrete random variable can be negative.</p>",
    r"""<p><strong>False.</strong></p>
    <p>For discrete variables, probabilities $p(x)$ satisfy $0 \le p(x) \le 1$.</p>
    <p>This means $\log_2 p(x) \le 0$ (logarithm of a number $\le 1$ is negative or zero).</p>
    <p>Therefore, $-p(x) \log_2 p(x) \ge 0$.</p>
    <p>Since the sum of non-negative terms is non-negative, $H(X) \ge 0$.</p>
    <p><em>Note: Differential entropy (for continuous variables) CAN be negative, as probability density can be greater than 1.</em></p>"""
))

# --- MEDIUM EXERCISES ---

exercises.append(("M1", "Joint Entropy", "Medium",
    r"<p>Let $X$ be the outcome of a fair coin flip (H/T). Let $Y=X$ (copy of X). What is the joint entropy $H(X, Y)$?</p>",
    r"""<p>We need to define the joint probability distribution $P(X, Y)$:</p>
    <ul>
        <li>$P(H, H) = 0.5$ (since X=H implies Y=H)</li>
        <li>$P(T, T) = 0.5$</li>
        <li>$P(H, T) = 0$ (impossible)</li>
        <li>$P(T, H) = 0$ (impossible)</li>
    </ul>
    <p>Now calculate entropy:</p>
    $$H(X, Y) = -\sum_{x,y} p(x,y) \log_2 p(x,y)$$
    $$= -(0.5 \log_2 0.5 + 0.5 \log_2 0.5 + 0 + 0)$$
    $$= -(-0.5 - 0.5) = 1 \text{ bit}$$
    <p><strong>Intuition:</strong> Since $Y$ provides no <em>new</em> information (it's just a copy), the total uncertainty is just the uncertainty of $X$.</p>"""
))

exercises.append(("M2", "Conditional Entropy", "Medium",
    r"<p>If $Y = f(X)$ is a deterministic function of $X$, what is $H(Y|X)$?</p>",
    r"""<p>Conditional entropy $H(Y|X)$ measures the uncertainty remaining in $Y$ if we know $X$.</p>
    <p>If $Y$ is a deterministic function of $X$ (e.g., $Y = X^2$), then knowing $X$ tells us $Y$ with 100% certainty.</p>
    <p>The conditional probability $P(y|x)$ will be 1 for the correct $y$ and 0 otherwise.</p>
    $$H(Y|X) = \sum_x p(x) H(Y|X=x)$$
    <p>Since $H(Y|X=x) = 0$ (entropy of a deterministic outcome), the sum is 0.</p>
    <p>Therefore, $\boxed{H(Y|X) = 0}$.</p>"""
))

exercises.append(("M3", "Huffman Lower Bound", "Medium",
    r"<p>Can the average length of a Huffman code be strictly less than the entropy $H(X)$?</p>",
    r"""<p><strong>No.</strong></p>
    <p>Shannon's Source Coding Theorem states that for any uniquely decodable code (which includes Huffman codes), the expected code length $L$ satisfies:</p>
    $$L \ge H(X)$$
    <p>Huffman coding constructs an optimal prefix code, so it gets very close to $H(X)$. Specifically:</p>
    $$H(X) \le L_{Huffman} < H(X) + 1$$
    <p>Equality $L = H(X)$ is achieved only if all symbol probabilities are negative powers of 2 (e.g., $1/2, 1/4, 1/8 \dots$).</p>"""
))

exercises.append(("M4", "KL Asymmetry", "Medium",
    r"<p>Let $P=[1, 0]$ and $Q=[0.5, 0.5]$. Calculate $D_{KL}(P||Q)$ and $D_{KL}(Q||P)$. What does this show?</p>",
    r"""<p>Formula: $D_{KL}(A||B) = \sum a_i \log_2 \frac{a_i}{b_i}$</p>
    
    <p><strong>1. $D_{KL}(P||Q)$:</strong></p>
    $$= 1 \cdot \log_2 \frac{1}{0.5} + 0 \cdot \log_2 \frac{0}{0.5}$$
    $$= 1 \cdot \log_2 2 + 0 = 1 \text{ bit}$$
    
    <p><strong>2. $D_{KL}(Q||P)$:</strong></p>
    $$= 0.5 \cdot \log_2 \frac{0.5}{1} + 0.5 \cdot \log_2 \frac{0.5}{0}$$
    <p>The second term contains division by zero ($\log \infty$). Thus:</p>
    $$D_{KL}(Q||P) = \infty$$
    
    <p><strong>Conclusion:</strong> KL Divergence is <strong>not symmetric</strong>. $D_{KL}(P||Q) \neq D_{KL}(Q||P)$. This is why it's a "divergence," not a "distance."</p>"""
))

exercises.append(("M5", "Cross-Entropy vs Entropy", "Medium",
    r"<p>Prove that Cross-Entropy $H(P, Q)$ is always greater than or equal to Entropy $H(P)$.</p>",
    r"""<p>We use the fundamental identity relating Cross-Entropy, Entropy, and KL Divergence:</p>
    $$H(P, Q) = H(P) + D_{KL}(P||Q)$$
    <p>We know from Gibbs' Inequality that KL Divergence is always non-negative:</p>
    $$D_{KL}(P||Q) \ge 0$$
    <p>Therefore:</p>
    $$H(P, Q) \ge H(P) + 0$$
    $$H(P, Q) \ge H(P)$$
    <p>Equality holds only when $D_{KL}(P||Q) = 0$, which implies $P = Q$.</p>"""
))

exercises.append(("M6", "Entropy of Sum", "Medium",
    r"<p>Let $X_1, X_2$ be independent fair binary variables (0 or 1). Let $Y = X_1 + X_2$. What is $H(Y)$?</p>",
    r"""<p>Let's derive the distribution of $Y$. Possible values are $\{0, 1, 2\}$.</p>
    <ul>
        <li>$Y=0$: Requires $X_1=0, X_2=0$. Prob = $0.5 \times 0.5 = 0.25$.</li>
        <li>$Y=1$: Requires $(0,1)$ or $(1,0)$. Prob = $0.25 + 0.25 = 0.5$.</li>
        <li>$Y=2$: Requires $X_1=1, X_2=1$. Prob = $0.5 \times 0.5 = 0.25$.</li>
    </ul>
    <p>Distribution $P_Y = [0.25, 0.5, 0.25]$.</p>
    <p>Calculate Entropy:</p>
    $$H(Y) = -(0.25 \log 0.25 + 0.5 \log 0.5 + 0.25 \log 0.25)$$
    $$= -(-0.5 - 0.5 - 0.5) = 1.5 \text{ bits}$$"""
))

exercises.append(("M7", "Mutual Information", "Medium",
    r"<p>Define Mutual Information $I(X; Y)$ in terms of Entropy $H(X)$ and Conditional Entropy $H(X|Y)$. Explain intuitively.</p>",
    r"""<p><strong>Definition:</strong></p>
    $$I(X; Y) = H(X) - H(X|Y)$$
    
    <p><strong>Intuitive Explanation:</strong></p>
    <ul>
        <li>$H(X)$ is your initial uncertainty about $X$.</li>
        <li>$H(X|Y)$ is your remaining uncertainty about $X$ after knowing $Y$.</li>
        <li>The difference is the <strong>information gained</strong> about $X$ by learning $Y$.</li>
    </ul>
    <p>It represents how much knowing one variable reduces uncertainty about the other.</p>"""
))

exercises.append(("M8", "Perplexity", "Medium",
    r"<p>In NLP, models are evaluated on 'Perplexity', defined as $2^{H(P)}$. If a model has a perplexity of 8, what does this intuitively mean?</p>",
    r"""<p>Perplexity is the "branching factor" or the effective number of choices.</p>
    <p>If Perplexity = 8, then Entropy $H(P) = \log_2(8) = 3$ bits.</p>
    <p><strong>Intuition:</strong> A perplexity of 8 means the model is as unsure about the next word as if it were choosing uniformly at random from <strong>8 equally likely possibilities</strong>.</p>
    <p>Lower perplexity indicates better prediction (less surprise).</p>"""
))

exercises.append(("M9", "Chain Rule", "Medium",
    r"<p>Write the Chain Rule for Entropy for three variables $H(X, Y, Z)$.</p>",
    r"""<p>The Chain Rule for Entropy states:</p>
    $$H(X_1, \dots, X_n) = \sum_{i=1}^n H(X_i | X_1, \dots, X_{i-1})$$
    <p>For three variables $X, Y, Z$:</p>
    $$H(X, Y, Z) = H(X) + H(Y|X) + H(Z|X, Y)$$
    <p><strong>Intuition:</strong> The total information in the set $(X, Y, Z)$ is:</p>
    <ol>
        <li>Information in $X$ alone, plus</li>
        <li>Information in $Y$ that wasn't already in $X$, plus</li>
        <li>Information in $Z$ that wasn't in $X$ or $Y$.</li>
    </ol>"""
))

exercises.append(("M10", "Concavity", "Medium",
    r"<p>Is the entropy function $H(p)$ concave or convex? Why is this important for optimization?</p>",
    r"""<p>Entropy $H(p) = -\sum p_i \log p_i$ is a <strong>concave</strong> function.</p>
    <ul>
        <li>Geometrically, the chord connecting any two points on the curve lies <em>below</em> the curve.</li>
        <li>The second derivative $H''(p) = -1/p$ (assuming natural log base) is negative for $p>0$, confirming concavity.</li>
    </ul>
    <p><strong>Importance:</strong> Concavity guarantees that any local maximum is a global maximum. This implies that there is a unique probability distribution (the uniform distribution, subject to constraints) that maximizes entropy.</p>"""
))

# --- HARD EXERCISES ---

exercises.append(("H1", "Geometric Distribution", "Hard",
    r"<p>Derive the entropy of a geometric distribution $P(k) = (1-p)^{k-1}p$ for $k=1, 2, \dots$.</p>",
    r"""<p>Let $q = 1-p$. The distribution is $P(k) = q^{k-1}p$.</p>
    $$H(X) = -\sum_{k=1}^{\infty} q^{k-1}p \log_2(q^{k-1}p)$$
    $$= -\sum_{k=1}^{\infty} q^{k-1}p [(k-1)\log_2 q + \log_2 p]$$
    $$= -p \log_2 q \sum_{k=1}^{\infty} (k-1)q^{k-1} - p \log_2 p \sum_{k=1}^{\infty} q^{k-1}$$
    
    <p>We use sum identities:</p>
    <ul>
        <li>$\sum_{k=1}^{\infty} q^{k-1} = \frac{1}{1-q} = \frac{1}{p}$</li>
        <li>$\sum_{k=1}^{\infty} (k-1)q^{k-1} = \frac{q}{(1-q)^2} = \frac{q}{p^2}$ (related to Expected Value)</li>
    </ul>
    
    <p>Substituting back:</p>
    $$H(X) = -p \log_2 q \left(\frac{q}{p^2}\right) - p \log_2 p \left(\frac{1}{p}\right)$$
    $$H(X) = -\frac{q}{p} \log_2 q - \log_2 p$$
    $$\boxed{H(X) = \frac{-(1-p)\log_2(1-p) - p\log_2 p}{p}}$$"""
))

exercises.append(("H2", "Independence Bound", "Hard",
    r"<p>Prove that $H(X, Y) \le H(X) + H(Y)$. When does equality hold?</p>",
    r"""<p>We start with the definition of Mutual Information:</p>
    $$I(X; Y) = H(X) + H(Y) - H(X, Y)$$
    <p>Mutual Information is actually a KL Divergence between the joint distribution and the product of marginals:</p>
    $$I(X; Y) = D_{KL}(P(X,Y) || P(X)P(Y))$$
    <p>Since KL divergence is always non-negative ($D_{KL} \ge 0$):</p>
    $$H(X) + H(Y) - H(X, Y) \ge 0$$
    $$H(X, Y) \le H(X) + H(Y)$$
    <p><strong>Equality:</strong> Holds if and only if $X$ and $Y$ are independent (i.e., $P(X,Y) = P(X)P(Y)$), making mutual information zero.</p>"""
))

exercises.append(("H3", "Differential Entropy Uniform", "Hard",
    r"<p>Calculate the differential entropy of a Continuous Uniform distribution on $[0, a]$.</p>",
    r"""<p>The Probability Density Function (PDF) is $p(x) = \frac{1}{a}$ for $0 \le x \le a$, and 0 otherwise.</p>
    <p>Differential entropy definition:</p>
    $$h(X) = -\int_{-\infty}^{\infty} p(x) \ln p(x) dx$$
    $$= -\int_{0}^{a} \frac{1}{a} \ln\left(\frac{1}{a}\right) dx$$
    <p>The term $\frac{1}{a} \ln(1/a)$ is constant wrt $x$.</p>
    $$= -\ln\left(\frac{1}{a}\right) \int_{0}^{a} \frac{1}{a} dx$$
    $$= -(-\ln a) \cdot 1$$
    $$\boxed{h(X) = \ln a}$$
    <p><strong>Insight:</strong> If $a < 1$, $\ln a < 0$. Differential entropy can be negative!</p>"""
))

exercises.append(("H4", "Exponential Entropy", "Hard",
    r"<p>Calculate the differential entropy of an Exponential distribution $\lambda e^{-\lambda x}$.</p>",
    r"""<p>PDF: $p(x) = \lambda e^{-\lambda x}$ for $x \ge 0$.</p>
    $$h(X) = - \int_0^{\infty} p(x) \ln(\lambda e^{-\lambda x}) dx$$
    $$= - \int_0^{\infty} p(x) [\ln \lambda - \lambda x] dx$$
    $$= - \left( \ln \lambda \int p(x) dx - \lambda \int x p(x) dx \right)$$
    <p>Using $\int p(x) dx = 1$ and $\int x p(x) dx = E[X] = \frac{1}{\lambda}$:</p>
    $$= - \left( \ln \lambda (1) - \lambda \left(\frac{1}{\lambda}\right) \right)$$
    $$= - (\ln \lambda - 1)$$
    $$\boxed{h(X) = 1 - \ln \lambda} \text{ (nats)}$$"""
))

exercises.append(("H5", "Gaussian KL", "Hard",
    r"<p>Write the formula for KL Divergence between two univariate Gaussians $P \sim \mathcal{N}(\mu_1, \sigma_1^2)$ and $Q \sim \mathcal{N}(\mu_2, \sigma_2^2)$.</p>",
    r"""<p>The closed-form solution is widely used in VAEs:</p>
    $$D_{KL}(P||Q) = \int p(x) \ln \frac{p(x)}{q(x)} dx$$
    <p>Substituting the Gaussian PDFs and integrating (skipping complex algebra steps) yields:</p>
    $$\boxed{D_{KL}(P||Q) = \ln\frac{\sigma_2}{\sigma_1} + \frac{\sigma_1^2 + (\mu_1 - \mu_2)^2}{2\sigma_2^2} - \frac{1}{2}}$$
    <p><strong>Special Case (VAE):</strong> If $Q$ is standard normal $\mathcal{N}(0, 1)$:</p>
    $$D_{KL} = -\frac{1}{2}\left(1 + \ln \sigma_1^2 - \mu_1^2 - \sigma_1^2\right)$$"""
))

exercises.append(("H6", "Max Entropy Variance", "Hard",
    r"<p>Why does the Gaussian distribution have the maximum entropy among all distributions with a fixed mean and variance?</p>",
    r"""<p>This is a constrained optimization problem.</p>
    <p>We want to maximize $h(p) = -\int p(x) \ln p(x) dx$ subject to:</p>
    <ol>
        <li>$\int p(x) dx = 1$ (Normalization)</li>
        <li>$\int x p(x) dx = \mu$ (Fixed Mean)</li>
        <li>$\int (x-\mu)^2 p(x) dx = \sigma^2$ (Fixed Variance)</li>
    </ol>
    <p>Using <strong>Lagrange Multipliers</strong>, the functional derivative leads to the form:</p>
    $$p(x) = e^{\lambda_0 + \lambda_1 x + \lambda_2 x^2}$$
    <p>This implies the log-density is quadratic, which is the defining characteristic of a Gaussian distribution ($e^{-x^2}$).</p>"""
))

exercises.append(("H7", "MLE vs KL", "Hard",
    r"<p>Formal proof: Show that maximizing Likelihood is equivalent to minimizing KL divergence from the empirical distribution to the model.</p>",
    r"""<p>Let $P_{data}$ be the empirical data distribution (point masses at data points) and $Q_\theta$ be our model.</p>
    <p>Minimizing KL:</p>
    $$\text{argmin}_\theta D_{KL}(P_{data} || Q_\theta) = \text{argmin}_\theta \sum P_{data}(x) \log \frac{P_{data}(x)}{Q_\theta(x)}$$
    $$= \text{argmin}_\theta \left( \sum P_{data}(x) \log P_{data}(x) - \sum P_{data}(x) \log Q_\theta(x) \right)$$
    <p>The first term depends only on data (constant). Minimizing the whole expression is equivalent to minimizing the second term:</p>
    $$= \text{argmin}_\theta \left( - \sum P_{data}(x) \log Q_\theta(x) \right)$$
    $$= \text{argmax}_\theta \mathbb{E}_{x \sim data} [\log Q_\theta(x)]$$
    <p>This expectation is exactly the <strong>Log-Likelihood</strong>. Thus, Min KL $\iff$ Max Likelihood.</p>"""
))

exercises.append(("H8", "Entropy Rate", "Hard",
    r"<p>For a stochastic process (like a Markov chain), what is the Entropy Rate?</p>",
    r"""<p>The Entropy Rate $\mathcal{H}(X)$ measures the asymptotic average information content per symbol.</p>
    <p><strong>Definition:</strong></p>
    $$\mathcal{H}(X) = \lim_{n \to \infty} \frac{1}{n} H(X_1, X_2, \dots, X_n)$$
    <p>For a stationary Markov chain, this simplifies to the conditional entropy of the next state given the current state:</p>
    $$\mathcal{H}(X) = - \sum_{i} \pi_i \sum_{j} P_{ij} \log P_{ij}$$
    <p>Where $\pi$ is the stationary distribution and $P_{ij}$ is the transition matrix.</p>"""
))

exercises.append(("H9", "Data Processing Inequality", "Hard",
    r"<p>State the Data Processing Inequality for a Markov chain $X \to Y \to Z$.</p>",
    r"""<p><strong>Theorem:</strong> If $X \to Y \to Z$ forms a Markov chain (meaning $Z$ depends on $X$ only through $Y$), then:</p>
    $$I(X; Z) \le I(X; Y)$$
    <p><strong>Interpretation:</strong></p>
    <p>You cannot increase the information about $X$ by processing $Y$ to get $Z$. "Processing can only destroy information, never create it."</p>
    <p>This is fundamental in Deep Learning: layers of a network can effectively lose information about the input if not carefully designed.</p>"""
))

exercises.append(("H10", "Fano's Inequality", "Hard",
    r"<p>What does Fano's Inequality relate?</p>",
    r"""<p>Fano's Inequality provides a lower bound on the probability of error $P_e$ for any estimator $\hat{X}$ of $X$, based on the conditional entropy $H(X|Y)$.</p>
    <p><strong>Formula:</strong></p>
    $$H(P_e) + P_e \log(|\mathcal{X}| - 1) \ge H(X|Y)$$
    <p>Where $|\mathcal{X}|$ is the number of possible classes.</p>
    <p><strong>Meaning:</strong> If the conditional entropy $H(X|Y)$ is high (high uncertainty given observation), the error probability $P_e$ <em>must</em> be high. You cannot predict accurately if the information isn't there.</p>"""
))

# --- SPECIAL TOPIC: FREE ENERGY ---

exercises.append(("F1", "Free Energy Bound", "Hard",
    r"<p>Prove that Variational Free Energy $F$ is an upper bound on surprise $-\log P(x)$.</p>",
    r"""<p>We start with the definition of Free Energy (negative ELBO):</p>
    $$F = D_{KL}(Q(z) || P(z|x)) - \log P(x)$$
    <p>Rearranging for log-evidence:</p>
    $$-\log P(x) = F - D_{KL}(Q(z) || P(z|x))$$
    <p>We know that KL Divergence is always non-negative ($D_{KL} \ge 0$). Therefore, subtracting a positive number from $F$ gives $-\log P(x)$.</p>
    <p>This implies:</p>
    $$F \ge -\log P(x)$$
    <p><strong>Conclusion:</strong> $F$ is an upper bound on the surprise. Minimizing $F$ pushes it down towards the true surprise, implicitly maximizing the model evidence $P(x)$.</p>"""
))

exercises.append(("F2", "The Dark Room Problem", "Medium",
    r"<p>If agents minimize the entropy of their sensory states, why don't they just stay in a dark, silent room where inputs are perfectly predictable (low entropy)?</p>",
    r"""<p>This is a famous critique of the entropy minimization principle known as the "Dark Room Problem".</p>
    <p><strong>The Resolution:</strong> Agents minimize <strong>Variational Free Energy</strong>, not just raw sensory entropy. Free Energy is measured relative to a <em>generative model</em> (prior beliefs).</p>
    $$F \approx -\ln P(sensory\_input | internal\_model)$$
    <p>If an agent's internal model (evolved by natural selection) expects it to be fed, warm, and social, then sitting in a dark, cold, empty room generates a massive <strong>prediction error</strong> (Surprise).</p>
    <p>The sensory input (darkness) contradicts the prior expectation (light/food), resulting in HIGH Free Energy. To minimize $F$, the agent must leave the room to find the states it expects to occupy.</p>"""
))

exercises.append(("F3", "Accuracy-Complexity Tradeoff", "Medium",
    r"<p>How does minimizing Free Energy $F = \text{Complexity} - \text{Accuracy}$ prevent overfitting?</p>",
    r"""<p>Recall the decomposition of Free Energy (or negative ELBO):</p>
    $$F = \underbrace{D_{KL}(Q(z|x) \| P(z))}_{\text{Complexity}} - \underbrace{\mathbb{E}_{Q}[\log P(x|z)]}_{\text{Accuracy}}$$
    <ul>
        <li><strong>Accuracy Term:</strong> Wants the posterior $Q$ to explain the data $x$ perfectly. This drives the model to fit data closely (potentially overfitting).</li>
        <li><strong>Complexity Term:</strong> Wants the posterior $Q$ to stay close to the simple prior $P(z)$. This acts as a <strong>regularizer</strong>.</li>
    </ul>
    <p><strong>Mechanism:</strong> To overfit, the model would need to encode every noisy detail of $x$ into $z$, creating a complex, spiky posterior $Q$ that diverges far from the smooth prior $P$. This would explode the Complexity cost.</p>
    <p>Minimizing $F$ finds the optimal balance: explaining the data as well as possible without creating an overly complex internal representation.</p>"""
))


# Generate HTML
html_output = '''            <!-- Exercises Section -->
            <section id="exercises" style="padding-top: 2rem; margin-top: 2rem; border-top: 1px solid #e5e7eb;">
                <h2 style="margin-bottom: 1.5rem;">30 Practice Exercises</h2>'''

for ex in exercises:
    html_output += generate_exercise_html(*ex)

html_output += """
            </section>"""

print(html_output)