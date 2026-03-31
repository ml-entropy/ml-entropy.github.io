"""Common HTML scaffolding for decipherment tutorials."""

CHAPTERS = [
    ("00-information-theory-of-scripts", "Information Theory of Scripts"),
    ("01-egyptian-hieroglyphs", "Egyptian Hieroglyphs & the Rosetta Stone"),
    ("02-cuneiform", "Cuneiform Decipherment"),
    ("03-linear-b", "Linear B & the Ventris Grid"),
    ("04-maya-glyphs", "Maya Glyphs"),
    ("05-undeciphered-scripts", "Undeciphered Scripts"),
    ("06-computational-methods", "Computational Decipherment"),
]

def sidebar_links(active_idx):
    lines = []
    for i, (slug, title) in enumerate(CHAPTERS):
        cls = ' active' if i == active_idx else ''
        num = f"{i:02d}"
        lines.append(f'            <a href="../{slug}/index.html" class="sidebar-link{cls}">{num}. {title}</a>')
    return "\n".join(lines)

def nav_links(active_idx):
    parts = []
    prev_idx = active_idx - 1
    next_idx = active_idx + 1
    if prev_idx >= 0:
        slug, title = CHAPTERS[prev_idx]
        parts.append(f'''                    <a href="../{slug}/index.html" class="tutorial-nav-link prev">
                        <span class="nav-label">Previous</span>
                        <span class="nav-title">&larr; {title}</span>
                    </a>''')
    if next_idx < len(CHAPTERS):
        slug, title = CHAPTERS[next_idx]
        parts.append(f'''                    <a href="../{slug}/index.html" class="tutorial-nav-link next">
                        <span class="nav-label">Next</span>
                        <span class="nav-title">{title} &rarr;</span>
                    </a>''')
    return "\n".join(parts)

def header(title, description, breadcrumb_short, active_idx):
    return f'''<!DOCTYPE html>
<html lang="en" data-theme="light">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title} | Decipherment</title>
    <meta name="description" content="{description}">
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&family=Playfair+Display:wght@400;500;600;700&display=swap" rel="stylesheet">
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.css">
    <script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.js"></script>
    <script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/contrib/auto-render.min.js" onload="renderMathInElement(document.body, {{delimiters: [{{left: '$$', right: '$$', display: true}}, {{left: '$', right: '$', display: false}}], throwOnError: false}});"></script>
    <link rel="stylesheet" href="../../../css/main.css">
    <link rel="stylesheet" href="../../../css/components.css">
    <link rel="stylesheet" href="../../../css/sidebar.css">
    <link rel="icon" type="image/svg+xml" href="data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'><text y='.9em' font-size='90'>&#x1F4DC;</text></svg>">
</head>
<body>
    <nav class="navbar" id="navbar">
        <div class="nav-container">
            <a href="../../../index.html" class="nav-logo">
                <span class="logo-symbol">&nabla;</span>
                <span class="logo-text">ML Fundamentals</span>
            </a>
            <button class="nav-toggle" id="navToggle" aria-label="Toggle navigation">
                <span></span><span></span><span></span>
            </button>
            <div class="nav-menu" id="navMenu">
                <div class="nav-links">
                    <a href="../../../tutorials/ml/index.html" class="nav-link">Machine Learning</a>
                    <a href="../../../tutorials/decipherment/index.html" class="nav-link active">Decipherment</a>
                    <a href="../../../tutorials/linguistics/index.html" class="nav-link">Linguistics</a>
                    <a href="../../../tutorials/writing-systems/index.html" class="nav-link">Writing Systems</a>
                    <a href="../../../tutorials/probability/index.html" class="nav-link">Probability</a>
                    <a href="../../../tutorials/cs/index.html" class="nav-link">CS</a>
                    <a href="../../../index.html" class="nav-link">Home</a>
                </div>
                <button class="theme-toggle" id="themeToggle" aria-label="Toggle theme">
                    <svg class="sun-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                        <circle cx="12" cy="12" r="5"/>
                        <path d="M12 1v2M12 21v2M4.22 4.22l1.42 1.42M18.36 18.36l1.42 1.42M1 12h2M21 12h2M4.22 19.78l1.42-1.42M18.36 5.64l1.42-1.42"/>
                    </svg>
                    <svg class="moon-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                        <path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"/>
                    </svg>
                </button>
            </div>
        </div>
    </nav>
    <header class="tutorial-content-header">
        <div class="container">
            <nav class="breadcrumb">
                <a href="../../../index.html">Home</a>
                <span class="breadcrumb-separator">&rarr;</span>
                <a href="../index.html">Decipherment</a>
                <span class="breadcrumb-separator">&rarr;</span>
                <span>{breadcrumb_short}</span>
            </nav>
            <div class="tutorial-tabs">
                <a href="#theory" class="tutorial-tab active">Theory</a>
                <a href="#exercises" class="tutorial-tab">Exercises</a>
            </div>
        </div>
    </header>
    <div class="tutorial-wrapper">
        <aside class="tutorial-sidebar">
    <div class="sidebar-section">
        <h3 class="sidebar-section-title">Decipherment</h3>
        <nav class="sidebar-nav">
{sidebar_links(active_idx)}
        </nav>
    </div>
    <div class="sidebar-section" style="margin-top: 2rem;">
        <h3 class="sidebar-section-title">Related Subjects</h3>
        <nav class="sidebar-nav">
            <a href="../../writing-systems/index.html" class="sidebar-link">Writing Systems</a>
            <a href="../../linguistics/index.html" class="sidebar-link">Linguistics</a>
            <a href="../../ml/index.html" class="sidebar-link">Machine Learning</a>
            <a href="../../probability/index.html" class="sidebar-link">Probability</a>
            <a href="../../cs/index.html" class="sidebar-link">Computer Science</a>
        </nav>
    </div>
</aside>
        <main class="tutorial-main">
            <article class="article-content" id="theory">
'''

def footer(active_idx):
    return f'''
            </article>
            <div class="tutorial-nav">
{nav_links(active_idx)}
            </div>
        </main>
    </div>
    <footer class="footer">
        <div class="container">
            <div class="footer-content">
                <div class="footer-brand">
                    <span class="logo-symbol">&nabla;</span>
                    <span>ML Fundamentals</span>
                </div>
                <p class="footer-tagline">Deep understanding through first principles.</p>
            </div>
            <div class="footer-links">
                <a href="../../../index.html">Home</a>
                <a href="https://github.com/ml-entropy/ml-entropy.github.io" target="_blank">GitHub</a>
            </div>
        </div>
    </footer>
    <script src="../../../js/main.js"></script>
</body>
</html>'''

def exercise_block(exercises):
    """exercises: list of (difficulty_emoji, difficulty_label, title, question_html, solution_html)"""
    lines = ['            <section class="article-content" id="exercises" style="display:none;">']
    lines.append('                <h2>Exercises</h2>')
    lines.append('                <p>30 exercises spanning three difficulty levels. Work through them in order or jump to your level.</p>')
    for i, (emoji, label, title, question, solution) in enumerate(exercises, 1):
        lines.append(f'                <div class="exercise-item">')
        lines.append(f'                    <div class="exercise-header">')
        lines.append(f'                        <span class="exercise-difficulty">{emoji}</span>')
        lines.append(f'                        <span class="exercise-title">{i}. {title}</span>')
        lines.append(f'                        <span class="exercise-level">{label}</span>')
        lines.append(f'                    </div>')
        lines.append(f'                    <div class="exercise-body">')
        lines.append(f'                        <div class="exercise-question">{question}</div>')
        lines.append(f'                        <details class="exercise-solution">')
        lines.append(f'                            <summary>Show Solution</summary>')
        lines.append(f'                            <div class="solution-content">{solution}</div>')
        lines.append(f'                        </details>')
        lines.append(f'                    </div>')
        lines.append(f'                </div>')
    lines.append('            </section>')
    return "\n".join(lines)

def write_chapter(idx, title, description, breadcrumb, theory_html, exercises):
    slug = CHAPTERS[idx][0]
    path = f"docs/tutorials/decipherment/{slug}/index.html"
    content = header(title, description, breadcrumb, idx)
    content += theory_html
    content += "\n"
    content += exercise_block(exercises)
    content += footer(idx)
    with open(path, 'w') as f:
        f.write(content)
    lines = content.count('\n') + 1
    print(f"  Wrote {path}: {lines} lines")
    return lines
