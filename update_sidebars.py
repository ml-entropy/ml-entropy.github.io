import os
import re

TUTORIALS_ROOT = "docs/tutorials"
CSS_LINK = '<link rel="stylesheet" href="../../../css/sidebar.css">'

NAV_LINKS = {
    "ml": [
        ("00-probability", "00. Probability Foundations"),
        ("01-entropy", "01. Entropy Fundamentals"),
        ("02-kl-divergence", "02. KL Divergence"),
        ("03-distributions", "03. Normal Distributions"),
        ("04-vae", "04. VAE & Variational Inference"),
        ("05-logarithms", "05. Why Logarithms?"),
        ("06-probability-concepts", "06. Probability Concepts"),
        ("07-combinatorics", "07. Combinatorics"),
        ("08-backpropagation", "08. Backpropagation"),
        ("09-regularization", "09. Regularization"),
        ("10-batch-normalization", "10. Batch Normalization"),
        ("11-learning-rate", "11. Learning Rate"),
        ("12-cnn", "12. CNNs"),
        ("13-rnn", "13. RNNs"),
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
    ]
}

TITLES = {
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
        active_class = ' active' if folder in current_folder else ''
        # We assume we are in docs/tutorials/category/folder/index.html
        # So link to ../folder/index.html
        link = f'../{folder}/index.html'
        html += f'                    <a href="{link}" class="sidebar-link{active_class}">{title}</a>\n'
    
    html += '''                </nav>
            </div>
            
            <div class="sidebar-section" style="margin-top: 2rem;">
                <h3 class="sidebar-section-title">Related Subjects</h3>
                <nav class="sidebar-nav">
    '''
    
    # Add cross links
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
    if "tutorial-wrapper" in content:
        print(f"Already updated: {filepath}")
        if "01-entropy/index.html" in filepath:
             # Strip inline style
             content = re.sub(r'<style>.*?</style>', '', content, flags=re.DOTALL)
             if CSS_LINK not in content:
                 content = content.replace('</head>', f'{CSS_LINK}\n</head>')
             with open(filepath, 'w') as f:
                 f.write(content)
             print("Updated 01-entropy to use external CSS")
        return

    # 1. Inject CSS Link
    if CSS_LINK not in content:
        content = content.replace('</head>', f'{CSS_LINK}\n</head>')

    # 2. Extract Main Content and TOC
    # Pattern: <main class="tutorial-article"> ... </main> ... <aside class="toc-container"> ... </aside>
    # Note: whitespace might vary.
    
    main_pattern = re.compile(r'<main class="tutorial-article">\s*<div class="container">(.*?)</div>\s*</main>', re.DOTALL)
    toc_pattern = re.compile(r'<aside class="toc-container">(.*?)</aside>', re.DOTALL)
    
    main_match = main_pattern.search(content)
    toc_match = toc_pattern.search(content)
    
    if not main_match:
        print(f"Could not find main content in {filepath}")
        return

    article_content = main_match.group(1) # This contains <article class="article-content" ...
    
    toc_html = ""
    if toc_match:
        toc_html = f'<aside class="toc-container">{toc_match.group(1)}</aside>'
    
    # 3. Build New Structure
    sidebar_html = generate_sidebar(category, current_folder)
    
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
    # We remove the old main and old toc.
    # Note: The TOC might be after main.
    
    # Remove TOC first if it exists
    if toc_match:
        content = content.replace(toc_match.group(0), "")
        
    # Replace Main with New Structure
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