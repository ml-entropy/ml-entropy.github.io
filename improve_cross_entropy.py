
import re

INDEX_PATH = "docs/tutorials/ml/02-cross-entropy/index.html"
EXERCISES_PATH = "docs/tutorials/ml/02-cross-entropy/exercises.html"

def update_index_content():
    print(f"Updating {INDEX_PATH}...")
    with open(INDEX_PATH, "r", encoding="utf-8") as f:
        content = f.read()

    # 1. Add Communication Channel Diagram to Intuition
    # We'll insert it after the definition box in Section 1.
    
    diagram_svg = """
    <div class="figure-container" style="margin: 2rem 0; text-align: center;">
        <svg width="100%" height="220" viewBox="0 0 600 220" style="background: #f8f9fa; border-radius: 8px;">
            <!-- Alice (Source) -->
            <rect x="50" y="60" width="100" height="80" rx="8" fill="#e0e7ff" stroke="#4f46e5" stroke-width="2"/>
            <text x="100" y="95" text-anchor="middle" font-family="Inter" font-weight="bold" fill="#3730a3">True Dist P</text>
            <text x="100" y="115" text-anchor="middle" font-family="Inter" font-size="12" fill="#3730a3">(Reality)</text>
            
            <!-- Arrow 1 -->
            <line x1="150" y1="100" x2="230" y2="100" stroke="#9ca3af" stroke-width="2" marker-end="url(#arrowhead)"/>
            
            <!-- Codebook Q -->
            <rect x="230" y="40" width="140" height="120" rx="8" fill="#fef3c7" stroke="#d97706" stroke-width="2"/>
            <text x="300" y="70" text-anchor="middle" font-family="Inter" font-weight="bold" fill="#92400e">Model Q</text>
            <text x="300" y="90" text-anchor="middle" font-family="Inter" font-size="12" fill="#92400e">"The Wrong Codebook"</text>
            <text x="300" y="110" text-anchor="middle" font-family="Inter" font-size="11" fill="#92400e">Code Length = -log Q(x)</text>
            <text x="300" y="130" text-anchor="middle" font-family="Inter" font-size="11" fill="#92400e" font-style="italic">Optimized for Q, not P</text>
            
            <!-- Arrow 2 -->
            <line x1="370" y1="100" x2="450" y2="100" stroke="#9ca3af" stroke-width="2" marker-end="url(#arrowhead)"/>
            
            <!-- Result -->
            <rect x="450" y="60" width="120" height="80" rx="8" fill="#fee2e2" stroke="#ef4444" stroke-width="2"/>
            <text x="510" y="95" text-anchor="middle" font-family="Inter" font-weight="bold" fill="#991b1b">Avg Cost</text>
            <text x="510" y="115" text-anchor="middle" font-family="Inter" font-size="12" fill="#991b1b">H(P, Q)</text>
            
            <!-- Definitions -->
            <defs>
                <marker id="arrowhead" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto">
                    <polygon points="0 0, 10 3.5, 0 7" fill="#9ca3af" />
                </marker>
            </defs>
        </svg>
        <p class="caption" style="margin-top: 0.5rem; font-size: 0.9rem; color: #666;">
            <strong>The Communication Analogy:</strong> Events occur according to Reality ($P$), but we encode them using a scheme based on our Model ($Q$). 
            Since $Q 
eq P$, our codes are longer than necessary. Cross-Entropy measures this average length.
        </p>
    </div>
    """
    
    # Locate insertion point: After the definition box in the Introduction
    # <div class="definition-box"> ... </div>
    # We look for the closing div of the definition box inside the introduction section
    
    # We search for the unique text inside that box to be safe
    target_str = "optimized for a different distribution $Q$."
    if target_str in content:
        # Find the closing div tag after this text
        parts = content.split(target_str)
        # parts[1] starts with 
                </div>...
        sub_parts = parts[1].split("</div>", 1)
        
        # Reconstruct
        new_content = parts[0] + target_str + sub_parts[0] + "</div>" + "
" + diagram_svg + sub_parts[1]
        content = new_content
    else:
        print("Warning: Could not find insertion point for Diagram.")

    # 2. Add PyTorch Code
    # Look for the Python Implementation section
    
    pytorch_code = """
<h3>PyTorch Implementation</h3>
<p>In practice, we rarely implement this manually. PyTorch's <code>nn.CrossEntropyLoss</code> combines the Softmax and Log-Likelihood steps for numerical stability.</p>

<pre><code class="language-python">import torch
import torch.nn as nn

# 1. Inputs
# PyTorch expects raw logits (scores), not probabilities!
# Shape: (Batch Size, Number of Classes)
logits = torch.tensor([[1.5, 2.0, 0.5]])  # Raw scores for [Cat, Dog, Bird]

# 2. True Labels
# Class indices (not one-hot vectors)
target = torch.tensor([1])  # Class 1 is "Dog"

# 3. Define and Compute Loss
criterion = nn.CrossEntropyLoss()
loss = criterion(logits, target)

print(f"PyTorch Loss: {loss.item():.4f}")

# Verification manually:
# Softmax([1.5, 2.0, 0.5]) -> [0.32, 0.53, 0.12] (approx)
# -log(0.53) -> ~0.63
</code></pre>
"""
    
    # Insert before the closing </section> of #code
    if '<section id="code">' in content:
        content = content.replace('</section>', pytorch_code + '
            </section>', 1) 
        # Using count=1 might be risky if there are multiple sections but I'll target the specific one via regex if needed.
        # Actually, replace is global by default in Python string (wait, count is optional arg).
        # Let's be more specific.
        
        # Find the #code section content
        code_section = re.search(r'(<section id="code">.*?</section>)', content, re.DOTALL)
        if code_section:
            old_block = code_section.group(1)
            # Insert before the last </div> (closing the code-preview or similar?) No, inside the section.
            # Just append to the end of the section content
            new_block = old_block.replace('</section>', pytorch_code + '
            </section>')
            content = content.replace(old_block, new_block)
    
    with open(INDEX_PATH, "w", encoding="utf-8") as f:
        f.write(content)
    print("✅ Updated index.html")

def fix_exercises_content():
    print(f"Updating {EXERCISES_PATH}...")
    with open(EXERCISES_PATH, "r", encoding="utf-8") as f:
        content = f.read()
        
    # 1. Remove Markdown Artifacts
    # e.g., <p>## Part B: Coding</p>
    content = re.sub(r'<p>## .*?</p>', '', content)
    
    # 2. Fix Code Blocks
    # <pre><code>python -> <pre><code class="language-python">
    content = content.replace('<pre><code>python', '<pre><code class="language-python">')
    
    # 3. Fix bolding artifacts inside code blocks if any (sometimes happen with markdown conversion)
    # Check for <em></em> inside code? Unlikely to handle robustly here, but let's check for the specific artifact mentioned
    # "<em></em>2" -> "**2" or "^2"?
    # The artifact `(1 - p)<em></em>2` looks like it was meant to be `(1-p)^2` or `(1-p)**2`.
    # Markdown `*` usually italicizes. `**` bolds.
    # If it was `(1-p)**2`, markdown might have parsed `**` as empty bold? No.
    # If it was `(1-p)*2`, it becomes `(1-p)<em></em>2`? No.
    # Let's just fix the specific case if found.
    content = content.replace('<em></em>', '**') 
    
    with open(EXERCISES_PATH, "w", encoding="utf-8") as f:
        f.write(content)
    print("✅ Updated exercises.html")

if __name__ == "__main__":
    update_index_content()
    fix_exercises_content()
