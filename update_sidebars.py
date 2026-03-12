import os
import re

TUTORIALS_ROOT = "docs/tutorials"
CSS_LINK = '<link rel="stylesheet" href="../../../css/sidebar.css">'

# Updated to match actual file system structure
NAV_LINKS = {
    "tts": [
        ("01-f5-tts", "01. F5-TTS: Flow Matching"),
        ("02-qwen-tts", "02. Qwen-TTS: LLM-based Audio"),
    ],
    "cs": [
        ("01-floating-point", "01. Floating-Point Representation"),
    ],
    "ml": [
        ("00-probability", "00. Probability Foundations"),
        ("01-logarithms", "01. Why Logarithms?"),
        ("02-combinatorics", "02. Combinatorics"),
        ("03-distributions", "03. Normal Distributions"),
        ("04-entropy", "04. Entropy Fundamentals"),
        ("05-cross-entropy", "05. Cross-Entropy"),
        ("06-kl-divergence", "06. KL Divergence"),
        ("07-entropy-connections", "07. Entropy Connections"),
        ("08-backpropagation", "08. Backpropagation"),
        ("09-regularization", "09. Regularization"),
        ("10-batch-normalization", "10. Batch Normalization"),
        ("11-learning-rate", "11. Learning Rate"),
        ("12-cnn", "12. CNNs"),
        ("13-rnn", "13. RNNs"),
        ("14-rate-distortion", "14. Rate-Distortion Theory"),
        ("15-autoencoder", "15. Autoencoders"),
        ("16-bayesian-inference", "16. Bayesian Inference"),
        ("17-variational-inference", "17. Variational Inference"),
        ("18-vae", "18. VAE"),
        ("19-inductive-bias", "19. Inductive Bias"),
        ("20-architectural-biases", "20. Architectural Biases"),
        ("21-designing-biases", "21. Designing Biases"),
        ("22-fst-fundamentals", "22. FST Fundamentals"),
        ("23-weighted-fsts", "23. Weighted FSTs"),
        ("24-fst-libraries", "24. FST Libraries"),
        ("25-fst-applications", "25. FST Applications"),
        ("26-neural-symbolic", "26. Neural-Symbolic Hybrids"),
        ("27-sequence-alignment", "27. Sequence Alignment"),
        ("28-mas-algorithm", "28. MAS Algorithm"),
        ("29-forced-alignment", "29. Forced Alignment & MFA"),
        ("30-tts-fundamentals", "30. TTS Fundamentals"),
        ("31-neural-vocoders", "31. Neural Vocoders"),
        ("32-tacotron", "32. Tacotron & Attention TTS"),
        ("33-fastspeech", "33. FastSpeech & Non-AR TTS"),
        ("34-glow-tts", "34. Glow-TTS & Flows"),
        ("35-vits", "35. VITS: End-to-End TTS"),
        ("36-bilingual-tts", "36. Bilingual TTS: RU+KY"),
        ("37-gans", "37. GANs"),
        ("38-normalizing-flows", "38. Normalizing Flows"),
        ("39-flow-matching", "39. Flow Matching"),
        ("40-diffusion-models", "40. Diffusion Models"),
    ],
    "linear-algebra": [
        ("01-vectors", "01. Vectors and Spaces"),
        ("02-matrices", "02. Matrices"),
        ("03-systems", "03. Systems of Equations"),
        ("04-determinants", "04. Determinants"),
        ("05-eigenvalues", "05. Eigenvalues"),
        ("06-svd", "06. SVD"),
        ("07-orthogonality", "07. Orthogonality"),
        ("08-positive-definite", "08. Positive Definite"),
    ],
    "calculus": [
        ("01-derivatives", "01. Derivatives"),
        ("02-multivariable", "02. Multivariable"),
        ("03-directional", "03. Directional Derivatives"),
        ("04-matrix-calculus", "04. Matrix Calculus"),
    ],
    "physics": [
        ("01-entropy", "01. Entropy"),
        ("02-carnot", "02. Carnot Cycle"),
        ("03-differentials", "03. Differentials"),
    ]
}

TITLES = {
    "tts": "Text-to-Speech",
    "cs": "Computer Science",
    "ml": "Machine Learning",
    "linear-algebra": "Linear Algebra",
    "calculus": "Calculus",
    "physics": "Physics"
}

def generate_sidebar(category, current_folder):
    nav_items = NAV_LINKS.get(category, [])
    html = f'''
        <!-- Sidebar Navigation -->
        <aside class="tutorial-sidebar">
            <div class="sidebar-section">
                <h3 class="sidebar-section-title">{TITLES.get(category, category.title())}</h3>
                <nav class="sidebar-nav">
    '''
    
    for folder, title in nav_items:
        active_class = ' active' if folder == current_folder else ''
        link = f'../{folder}/index.html'
        html += f'                    <a href="{link}" class="sidebar-link{active_class}">{title}</a>\n'
    
    html += '''                </nav>
            </div>
            
            <div class="sidebar-section" style="margin-top: 2rem;">
                <h3 class="sidebar-section-title">Related Subjects</h3>
                <nav class="sidebar-nav">
    '''
    
    # Add cross links
    if category != 'tts':
        html += '                    <a href="../../tts/index.html" class="sidebar-link">Text-to-Speech</a>\n'
    if category != 'cs':
        html += '                    <a href="../../cs/index.html" class="sidebar-link">Computer Science</a>\n'
    if category != 'ml':
        html += '                    <a href="../../ml/index.html" class="sidebar-link">Machine Learning</a>\n'
    if category != 'linear-algebra':
        html += '                    <a href="../../linear-algebra/index.html" class="sidebar-link">Linear Algebra</a>\n'
    if category != 'calculus':
        html += '                    <a href="../../calculus/index.html" class="sidebar-link">Calculus</a>\n'
    if category != 'physics':
        html += '                    <a href="../../physics/index.html" class="sidebar-link">Physics</a>\n'
        
    html += '''                </nav>
            </div>
        </aside>'''
    return html

def update_file(filepath):
    print(f"Processing {filepath}...")
    with open(filepath, 'r') as f:
        content = f.read()

    # Determine category
    parts = filepath.split('/')
    if len(parts) < 4:
        return # Skip non-tutorial depth
    category = parts[-3] # docs/tutorials/[category]/[folder]/index.html
    current_folder = parts[-2]
    
    if category not in NAV_LINKS:
        print(f"Skipping unknown category: {category}")
        return

    # Check if already updated (look for tutorial-wrapper)
    # Even if updated, we MUST regenerate the sidebar to fix the broken links!
    # So we extract the sidebar part and replace it.
    
    wrapper_pattern = re.compile(r'<div class="tutorial-wrapper">.*?<aside class="tutorial-sidebar">(.*?)</aside>.*?<main class="tutorial-main">', re.DOTALL)
    
    sidebar_html = generate_sidebar(category, current_folder)
    # Extract the inner part of sidebar_html (remove the outer aside tags because the pattern matches inner)
    # Wait, the generate_sidebar returns the whole aside block.
    
    # New strategy: If wrapper exists, find the sidebar block and replace it.
    if "tutorial-wrapper" in content:
        # Regex to find the sidebar aside
        sidebar_pattern = re.compile(r'<aside class="tutorial-sidebar">.*?</aside>', re.DOTALL)
        if sidebar_pattern.search(content):
            content = sidebar_pattern.sub(sidebar_html.strip(), content)
            print(f"  Refreshed sidebar in {filepath}")
        else:
             print(f"  Warning: wrapper found but sidebar not found in {filepath}")
        
        # Also ensure CSS link is present
        if CSS_LINK not in content:
             content = content.replace('</head>', f'{CSS_LINK}\n</head>')
             
        with open(filepath, 'w') as f:
            f.write(content)
        return

    # 1. Inject CSS Link
    if CSS_LINK not in content:
        content = content.replace('</head>', f'{CSS_LINK}\n</head>')

    # 2. Extract Main Content and TOC
    main_pattern = re.compile(r'<main class="tutorial-article">\s*<div class="container">(.*?)</div>\s*</main>', re.DOTALL)
    toc_pattern = re.compile(r'<aside class="toc-container">(.*?)</aside>', re.DOTALL)
    
    main_match = main_pattern.search(content)
    toc_match = toc_pattern.search(content)
    
    if not main_match:
        print(f"Could not find main content in {filepath}")
        return

    article_content = main_match.group(1)
    
    toc_html = ""
    if toc_match:
        toc_html = f'<aside class="toc-container">{toc_match.group(1)}</aside>'
    
    # 3. Build New Structure
    new_html = f'''
    <!-- Main Content -->
    <div class="tutorial-wrapper">
        {sidebar_html}

        <!-- Main Article -->
        <main class="tutorial-main">
            {article_content}
        </main>

        <!-- TOC (Right Side) -->
        {toc_html}
    </div>
    '''
    
    # 4. Replace
    if toc_match:
        content = content.replace(toc_match.group(0), "")
    content = content.replace(main_match.group(0), new_html)
    
    with open(filepath, 'w') as f:
        f.write(content)
    print(f"Successfully updated {filepath}")

def main():
    for root, dirs, files in os.walk(TUTORIALS_ROOT):
        for file in files:
            if file == "index.html":
                filepath = os.path.join(root, file)
                rel_path = os.path.relpath(filepath, TUTORIALS_ROOT)
                parts = rel_path.split('/')
                
                if len(parts) == 3:
                    update_file(filepath)

if __name__ == "__main__":
    main()
