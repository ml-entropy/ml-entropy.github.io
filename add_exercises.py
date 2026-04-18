#!/usr/bin/env python3
"""Add 60 new exercises (20 Easy, 20 Medium, 20 Hard) to the RVQ tutorial."""

import re

FILE = "docs/tutorials/tts/14-residual-vector-quantization/index.html"

EASY = [
    ("21. Residual Definition", "What is a residual in the context of RVQ?", "A residual is the difference between the original vector (or previous residual) and its quantized approximation at the current level."),
    ("22. Number of Codebooks", "If an RVQ system uses 8 levels, how many separate codebooks does it maintain?", "It maintains 8 separate codebooks, one per quantization level."),
    ("23. Reconstruction Formula", "How do you reconstruct the final approximation from all RVQ levels?", "You sum the codeword vectors selected at every level: x&#x302; = c₁ + c₂ + ... + cQ."),
    ("24. First-Level Role", "What does the first RVQ level typically capture?", "The first level captures the coarsest, most energy-dominant structure of the input, such as the fundamental frequency and spectral envelope in audio."),
    ("25. Codebook Size Terminology", "If each codebook has 1024 entries, how many bits does one token from that codebook represent?", "Each token represents log₂(1024) = 10 bits."),
    ("26. RVQ vs. Product Quantization", "Name one structural difference between RVQ and Product Quantization (PQ).", "RVQ quantizes the same full-dimensional vector repeatedly in sequence, whereas PQ splits the vector into disjoint subspaces and quantizes each subspace independently."),
    ("27. Successive Approximation", "Why is RVQ sometimes called successive approximation?", "Because each additional level successively refines the approximation by quantizing the error left over from all previous levels."),
    ("28. Token Sequence Length", "For an RVQ with Q levels and a single input frame, how many discrete tokens are produced?", "Exactly Q tokens are produced, one from each level's codebook."),
    ("29. Codebook Index Range", "If a codebook has K entries, what is the valid index range for tokens from that codebook?", "Valid indices range from 0 to K − 1 (or equivalently 1 to K)."),
    ("30. Zero Residual", "What does it mean if the residual at some level is exactly the zero vector?", "It means the previous levels already perfectly reconstructed the input, so no further refinement is needed at this or subsequent levels."),
    ("31. Commitment Loss Purpose", "What is the purpose of the commitment loss term in VQ/RVQ training?", "It encourages the encoder output to stay close to the selected codebook entry, preventing the encoder from changing too fast for the codebook to follow."),
    ("32. Straight-Through Estimator", "Why does RVQ training typically use a straight-through estimator?", "Because the argmin codebook lookup is non-differentiable, so gradients are copied from the quantized output directly to the encoder output."),
    ("33. Bitrate Unit", "If an RVQ produces 8 tokens per frame at 50 frames per second with 10-bit codebooks, what is the nominal bitrate?", "The nominal bitrate is 8 × 10 × 50 = 4000 bits per second (4 kbps)."),
    ("34. Codebook Utilization", "What does it mean when codebook utilization is low?", "It means many codebook entries are rarely or never selected, wasting representational capacity."),
    ("35. EMA Update", "What does EMA stand for in the context of codebook training, and what does it do?", "EMA stands for Exponential Moving Average; it updates codebook vectors as a running average of assigned encoder outputs instead of using gradient descent."),
    ("36. Quantization Error Direction", "After the first RVQ level, in which direction does the residual vector point?", "It points in the direction of the error between the original input and the first-level codeword, which could be any direction in the vector space."),
    ("37. Flat VQ Equivalent", "How large would a single flat codebook need to be to match the nominal capacity of an 8-level RVQ with 1024 entries per level?", "It would need 1024⁸ = 2⁸⁰ entries, which is astronomically large and infeasible."),
    ("38. Frame Independence", "In a basic RVQ setup, are tokens at different time frames dependent on each other?", "No, basic RVQ quantizes each frame independently; temporal dependencies are only introduced by a downstream model such as a language model."),
    ("39. Depth vs. Width", "In RVQ terminology, what do depth and width refer to?", "Depth refers to the number of quantization levels (Q), while width refers to the codebook size (K) at each level."),
    ("40. Decoding Simplicity", "Why is RVQ decoding computationally simple?", "Decoding only requires looking up Q codeword vectors by index and summing them, which involves no learned neural network computation."),
]

MEDIUM = [
    ("21. Bitrate Calculation with Mixed Codebooks", "An RVQ uses 4 levels with codebook sizes 2048, 1024, 512, and 256. At 75 frames/s, what is the nominal bitrate?", "Bits per frame = 11 + 10 + 9 + 8 = 38. Nominal bitrate = 38 × 75 = 2850 bps (2.85 kbps)."),
    ("22. Entropy Gap Analysis", "Level 3 of an RVQ has a 1024-entry codebook but empirical entropy of 7.2 bits. What fraction of capacity is wasted?", "Capacity is 10 bits, so 10 − 7.2 = 2.8 bits are wasted per token, meaning 28% of that level's nominal capacity is unused."),
    ("23. Dropout Probability Design", "Why might you set quantizer dropout probability higher for later levels than earlier ones?", "Later levels contribute less perceptually important detail, so dropping them more often teaches the model to produce acceptable output at lower bitrates without harming coarse structure."),
    ("24. Residual Magnitude Trend", "If RVQ is well-trained, how should the L2 norm of residuals change across levels?", "Residual norms should decrease monotonically because each level removes a portion of the remaining error, leaving less for subsequent levels."),
    ("25. Comparing Two Configurations", "Compare a 4-level RVQ with K=4096 per level vs. an 8-level RVQ with K=64 per level in terms of nominal bits per frame.", "4-level: 4 × 12 = 48 bits. 8-level: 8 × 6 = 48 bits. Both have the same nominal capacity, but they distribute it differently across depth and width."),
    ("26. Codebook Collapse Diagnosis", "You find that level 5 of your RVQ uses only 12 of its 1024 entries. What are two likely causes?", "Likely causes: (1) the residual at level 5 is too small or low-variance for 1024 entries to be distinguished, and (2) the learning rate or EMA decay may be poorly tuned, causing dead codes."),
    ("27. Entropy Coding Savings", "An 8-level RVQ has uniform 10-bit codebooks. Empirical per-level entropies are [9.8, 9.1, 8.3, 7.4, 6.5, 5.8, 5.2, 4.6]. What is the effective bitrate as a fraction of nominal?", "Effective = sum of entropies = 56.7 bits. Nominal = 80 bits. Fraction = 56.7/80 ≈ 70.9%, so entropy coding saves about 29% of the nominal bitrate."),
    ("28. Why Not Jointly Optimize?", "Why don't most RVQ implementations jointly optimize all codebooks simultaneously from the start?", "Joint optimization of all levels creates complex gradient interactions; training greedily (or with stop-gradient on earlier levels) is more stable and lets each level clearly learn its own residual."),
    ("29. Rate-Distortion Slope", "In a rate-distortion curve for RVQ, what does a flattening slope at higher levels indicate?", "It indicates diminishing returns: each additional level adds bits but reduces distortion by a smaller amount, suggesting the model is approaching the limit of useful refinement."),
    ("30. Language Model Token Order", "Why do autoregressive language models for RVQ tokens typically predict coarse levels before fine levels?", "Coarse levels carry the most mutual information with the signal's structure, so predicting them first gives the model a strong conditioning signal for predicting the less certain fine-level tokens."),
    ("31. Bandwidth Extension Analogy", "How is dropping the last few RVQ levels analogous to audio bandwidth extension?", "Both involve discarding high-frequency or fine detail and reconstructing it from coarser information, relying on learned priors to fill in plausible missing content."),
    ("32. Initialization Strategy", "Why is k-means initialization of codebooks beneficial for RVQ?", "K-means pre-positions codewords near high-density regions of the data distribution, giving each level a head start and reducing early training instability and dead codes."),
    ("33. Residual Whitening", "If residuals at level q become highly correlated across dimensions, what problem does this create for the next codebook?", "Correlated residuals concentrate in a low-dimensional subspace, making it hard for a codebook in the full space to efficiently cover the distribution, which wastes entries and hurts utilization."),
    ("34. Effective Bits vs. Perceptual Quality", "Two RVQ systems both use 6 kbps nominal. System A has higher entropy per level but lower perceptual quality. What might explain this?", "System A's codebooks may be poorly trained (high entropy but poor codeword placement), meaning tokens carry information that doesn't align well with perceptually relevant features."),
    ("35. Multi-Scale Dropout Schedule", "Describe a training schedule where quantizer dropout probability changes over training.", "Start with high dropout (e.g., 0.5) to force early levels to be self-sufficient, then anneal dropout toward a lower value (e.g., 0.1) so later levels learn fine detail without the model ignoring them."),
    ("36. Cross-Level Token Correlation", "If tokens at levels q and q+1 are highly correlated, what does that suggest about training?", "It suggests level q+1 is partially re-encoding information already captured by level q rather than focusing on the true residual, indicating a possible training or architecture issue."),
    ("37. Streaming RVQ Constraints", "What constraint does real-time streaming impose on RVQ that offline processing does not?", "Streaming requires causal processing: each frame's tokens must be produced without access to future frames, preventing the use of bidirectional context for better quantization."),
    ("38. Distortion Metric Choice", "Why might using a perceptual loss instead of MSE change which RVQ levels matter most?", "Perceptual losses weight frequency bands differently; early levels might focus on perceptually critical bands (e.g., 1-4 kHz for speech), and later levels on less important bands, changing the rate-distortion trade-off."),
    ("39. Finite-State Entropy by Level", "How would you measure whether tokens at level q are predictable from level q−1 tokens at the same frame?", "Compute the conditional entropy H(Tq | Tq−1): if it is much lower than H(Tq), then level q tokens are substantially predictable from level q−1, indicating cross-level redundancy."),
    ("40. Batch Size Effect on Codebook Updates", "Why can very small batch sizes hurt RVQ codebook training with EMA updates?", "Small batches produce noisy assignment statistics, causing EMA centroids to jitter instead of converging, which is especially harmful for low-frequency codewords that appear rarely in small batches."),
]

HARD = [
    ("21. Optimal Bit Allocation Across Levels", "Derive a principle for allocating different codebook sizes across RVQ levels to minimize total distortion at a fixed total bitrate.", "By analogy with reverse water-filling, levels whose residual variance is higher deserve larger codebooks (more bits). Formally, one should allocate bits proportional to log of the residual variance at each level, giving more capacity where the quantization error is largest."),
    ("22. RVQ as Successive Refinement Source Coding", "Relate RVQ to the information-theoretic concept of successive refinement. Under what source conditions is successive refinement optimal?", "Successive refinement achieves the rate-distortion bound at every stage for Gaussian sources under MSE distortion. RVQ approximates this by iteratively quantizing residuals; for non-Gaussian sources, there may be a gap between RVQ performance and the theoretical successive refinement bound."),
    ("23. Codebook Size Decay Schedule", "Propose and justify a schedule where codebook sizes decrease geometrically across levels (e.g., 2048, 1024, 512, ...).", "If residual variance drops roughly by a constant factor per level (as in well-trained RVQ), the entropy of each level's residual also decreases. Matching codebook size to this entropy avoids wasting entries at deeper levels while maintaining near-optimal distortion reduction at each stage."),
    ("24. Failure Mode: Codebook Shadowing", "Describe the failure mode where a later RVQ level learns to partially undo the work of an earlier level, and propose a mitigation.", "This occurs when gradients from the reconstruction loss push a later codebook to shift centroids that compensate for a misplaced earlier centroid, creating co-adapted pairs. Mitigation: use stop-gradient on earlier levels during later-level updates, or add an orthogonality regularizer between successive codebook update directions."),
    ("25. Information-Theoretic Bound on RVQ Levels", "Given a source with differential entropy h(X) and a target distortion D, derive a lower bound on the number of RVQ levels needed if each level uses a K-entry codebook.", "The rate-distortion function R(D) gives the minimum bits needed. Each level contributes at most log₂(K) bits. Therefore Q ≥ R(D) / log₂(K). For a Gaussian source, R(D) = ½ log₂(σ²/D), so Q ≥ log₂(σ²/D) / (2 log₂(K))."),
    ("26. Experiment: Shared vs. Independent Codebooks", "Design an experiment to test whether sharing a single codebook across all RVQ levels hurts performance, and predict the outcome.", "Train two systems on the same data: one with independent codebooks per level and one with a shared codebook. Measure distortion and codebook utilization at each level. Prediction: shared codebooks will underperform because the distribution of residuals changes across levels, and a single codebook cannot be optimal for all of them."),
    ("27. Entropy Bottleneck Interaction", "If you add an entropy bottleneck penalty (minimizing token entropy) during RVQ training, how does it interact with the residual chain?", "The penalty encourages each level to use fewer effective codewords, concentrating probability mass. Early levels resist compression because their tokens carry essential structure, but later levels collapse more aggressively, effectively performing automatic depth selection and variable-rate coding."),
    ("28. Analyzing RVQ with Non-Euclidean Distances", "What happens if you replace the Euclidean nearest-neighbor lookup in RVQ with cosine similarity? Analyze the effect on the residual chain.", "Cosine similarity ignores vector magnitude, so residuals no longer simply subtract out the selected codeword's contribution in all dimensions. The residual chain loses its additive reconstruction property, requiring either magnitude-aware corrections or a fundamentally different reconstruction formula."),
    ("29. Multi-Rate Architecture Design", "Design an RVQ system that can serve three quality tiers (low/medium/high) from the same encoder without retraining.", "Use Q levels with quantizer dropout during training. Low quality uses levels 1-2, medium uses 1-5, high uses all 1-Q. The dropout schedule during training must ensure that partial prefixes produce valid outputs. A hierarchical language model conditions on the chosen tier to predict only the relevant token subset."),
    ("30. Theoretical Comparison: RVQ vs. Tree-Structured VQ", "Compare the computational complexity and rate-distortion efficiency of RVQ with Q levels and K entries vs. tree-structured VQ with the same total codewords.", "RVQ: Q × K distance computations, K^Q effective codewords. Tree-structured VQ with branching factor K and depth Q: also Q × K computations, K^Q leaves. However, RVQ's additive structure means its effective Voronoi regions are sums of individual codebooks, which can be more flexible than a tree's rigid hierarchical partitioning for many source distributions."),
    ("31. Gradient Analysis of Deep RVQ", "Analyze why gradients for early RVQ levels can become noisy when the number of levels Q is large.", "With straight-through estimation, the reconstruction loss gradient flows through all Q levels. Early levels receive gradients that are the sum of direct error plus indirect effects from all later levels' quantization decisions. As Q grows, this sum accumulates quantization noise from each level's non-differentiable lookup, making early-level gradients increasingly noisy."),
    ("32. Adaptive Depth Selection at Inference", "Propose a method to dynamically choose the number of RVQ levels per frame at inference time based on signal characteristics.", "Compute the residual norm after each level. If it falls below a threshold (calibrated to a target distortion), stop quantizing that frame. This produces variable-length token sequences per frame, requiring a special padding or delimiter token for downstream models, but saves bitrate on easy-to-encode frames like silence."),
    ("33. Cross-Codebook Regularization", "Propose a regularization loss that prevents different RVQ levels from developing redundant codebook entries.", "Add a penalty term that maximizes the average pairwise distance between codewords across adjacent levels: L_reg = −λ Σ_{i,j} ||c_q^i − c_{q+1}^j||². This encourages each level's codebook to occupy different regions of the vector space, reducing redundancy in the additive reconstruction."),
    ("34. RVQ for Non-Stationary Sources", "How should RVQ codebook training be modified for highly non-stationary signals like music with abrupt transitions?", "Use online or sliding-window codebook adaptation where EMA statistics are computed over recent frames rather than the full dataset. Alternatively, condition codebook selection on a learned segment classifier, effectively maintaining multiple codebook sets for different signal regimes."),
    ("35. Proving Distortion Monotonicity", "Prove that adding an RVQ level can never increase reconstruction distortion (under MSE with optimal codebook).", "Let D_Q be distortion with Q levels. The (Q+1)-th level minimizes ||r_Q − c||² over codebook entries. The zero vector is implicitly achievable (or the best entry is at least as good), so D_{Q+1} = D_Q − ||r_Q − c*||² + ||r_Q − c* − r_{Q+1}||² ≤ D_Q since the optimal codeword reduces the residual norm. Formally, D_{Q+1} ≤ D_Q always holds."),
    ("36. Language Model Complexity vs. RVQ Depth", "Argue why a deeper RVQ (more levels) can make autoregressive language modeling harder even though it provides better reconstruction.", "More levels mean more tokens per frame, increasing sequence length and requiring the LM to model complex cross-level dependencies. The conditional distribution P(t_{q+1}|t_1,...,t_q) at deeper levels involves subtle residual patterns that are harder to predict, potentially requiring larger LM capacity and more training data to model effectively."),
    ("37. Codebook Pruning Strategy", "Design a post-training codebook pruning algorithm for RVQ that removes unused entries and analyze its effect on bitrate.", "Track codeword usage frequencies over a validation set. Remove entries used less than a threshold (e.g., &lt;0.01% of assignments). Re-index remaining entries with a smaller codebook. This reduces log₂(K) for affected levels, directly lowering nominal bitrate. The distortion increase should be minimal if pruned entries were truly unused."),
    ("38. RVQ in Latent Space vs. Waveform Space", "Compare training RVQ on raw waveform samples vs. on a learned encoder's latent space. Which should achieve better rate-distortion?", "Latent-space RVQ should achieve better rate-distortion because a learned encoder can remove irrelevant variation and align the representation with perceptually meaningful axes before quantization. Raw waveform RVQ must spend codebook capacity on perceptually irrelevant phase and amplitude variations."),
    ("39. Theoretical Limit of Identical Codebooks", "If all Q levels of an RVQ share identical codebooks and the source is Gaussian, characterize the distortion reduction per level as Q grows.", "With identical codebooks, each level reduces distortion by quantizing a residual that is the original minus a sum of quantized copies. For a Gaussian source, the residual remains approximately Gaussian but with reduced variance. Distortion drops geometrically: D_q ≈ D_1 × (1 − 1/K)^(q−1), but convergence slows because the codebook is not adapted to the changing residual distribution."),
    ("40. End-to-End RVQ Fine-Tuning Stability", "Explain why fine-tuning all RVQ levels end-to-end can cause training instability and propose a stabilization strategy.", "End-to-end gradients create circular dependencies: changing an early codebook shifts all subsequent residuals, invalidating later codebooks' learned distributions. Strategy: use a layerwise learning rate schedule where early levels have much smaller learning rates than later levels, or alternate between freezing early and late levels in successive training phases."),
]

def make_exercise_html(num, title, question, solution):
    return f'                    <div class="exercise-item"><div class="exercise-header"><span class="exercise-title">{num}. {title}</span><span class="exercise-toggle">&darr;</span></div><div class="exercise-body"><p>{question}</p><button class="btn btn-sm solution-toggle">Show Solution</button><div class="solution-content"><p><strong>Solution:</strong> {solution}</p></div></div></div>\n'

def build_block(exercises):
    lines = []
    for (label, question, solution) in exercises:
        num, title = label.split(". ", 1)
        lines.append(make_exercise_html(num, title, question, solution))
    return "\n" + "".join(lines)

with open(FILE, "r") as f:
    content = f.read()

lines = content.split("\n")

# Find the line numbers for the section headers
easy_header = None
medium_header = None
hard_header = None
for i, line in enumerate(lines):
    if "Easy</h3>" in line:
        easy_header = i
    elif "Medium</h3>" in line:
        medium_header = i
    elif "Hard</h3>" in line:
        hard_header = i

print(f"Easy header: {easy_header}, Medium header: {medium_header}, Hard header: {hard_header}")

# Insert before Medium header for easy exercises (after easy #20)
# Insert before Hard header for medium exercises (after medium #20)
# Insert before closing </div></article> for hard exercises (after hard #20)

# Find the last hard exercise closing </div> before </article>
# Work backwards: find </div>\n            </article>
# Actually let's find the last exercise-item closing in each section

# Strategy: insert easy block just before the Medium header line
# insert medium block just before the Hard header line
# insert hard block after the last exercise item in the hard section

easy_block = build_block(EASY)
medium_block = build_block(MEDIUM)
hard_block = build_block(HARD)

# Insert in reverse order of line numbers to preserve indices
# Hard: insert after the last exercise-item's closing </div> in the hard section
# Find last </div> of the last hard exercise (which is before </div>\n</article>)

# Find the closing </div> that ends the exercises container (the one right before </article>)
article_close = None
for i, line in enumerate(lines):
    if "</article>" in line:
        article_close = i
        break

# The line before </article> should be the closing </div> of the exercises-container
# Find the last exercise-item end before article_close
last_hard_exercise_end = None
for i in range(article_close - 1, hard_header, -1):
    if "</div>" in lines[i] and "exercise-item" not in lines[i]:
        # Look for the pattern of the last exercise-item's final </div>
        pass
    # Let's find the last line containing '</div>' that closes an exercise-item
    # Actually, let's just find the blank line after the last hard exercise

# Simpler approach: insert text right before the medium/hard headers and before closing div
# For easy: insert before medium_header line
# For medium: insert before hard_header line  
# For hard: insert before the </div> that closes the exercise container (line before </article>)

# Find the </div> + </article> pattern
exercises_container_close = None
for i in range(article_close - 1, -1, -1):
    stripped = lines[i].strip()
    if stripped == "</div>":
        exercises_container_close = i
        break

print(f"Exercises container close: {exercises_container_close}, Article close: {article_close}")

# Build new content by inserting blocks
new_lines = []
for i, line in enumerate(lines):
    if i == medium_header:
        # Insert easy exercises before medium header
        new_lines.append("")
        for ex in EASY:
            num, title = ex[0].split(". ", 1)
            new_lines.append(make_exercise_html(num, title, ex[1], ex[2]).rstrip())
        new_lines.append("")
        new_lines.append(line)
    elif i == hard_header:
        # Insert medium exercises before hard header
        new_lines.append("")
        for ex in MEDIUM:
            num, title = ex[0].split(". ", 1)
            new_lines.append(make_exercise_html(num, title, ex[1], ex[2]).rstrip())
        new_lines.append("")
        new_lines.append(line)
    elif i == exercises_container_close:
        # Insert hard exercises before closing div
        new_lines.append("")
        for ex in HARD:
            num, title = ex[0].split(". ", 1)
            new_lines.append(make_exercise_html(num, title, ex[1], ex[2]).rstrip())
        new_lines.append("")
        new_lines.append(line)
    else:
        new_lines.append(line)

with open(FILE, "w") as f:
    f.write("\n".join(new_lines))

print("Done! Exercises added.")
