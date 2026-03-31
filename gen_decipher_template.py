"""Shared HTML template for decipherment tutorial chapters."""
from pathlib import Path

ROOT = Path('/home/k_arzymatov/PycharmProjects/opus-tutorials/docs/tutorials/decipherment')

SIDEBAR_ITEMS = [
    ('00-information-theory', '00. Information Theory of Scripts'),
    ('01-egyptian-hieroglyphs', '01. Egyptian Hieroglyphs'),
    ('02-cuneiform', '02. Cuneiform & Behistun'),
    ('03-linear-b', '03. Linear B'),
    ('04-maya-glyphs', '04. Maya Glyphs'),
    ('05-undeciphered-scripts', '05. Undeciphered Scripts'),
    ('06-computational-methods', '06. Computational Methods'),
]

PREV_NEXT = {
    '00-information-theory': (None, '01-egyptian-hieroglyphs'),
    '01-egyptian-hieroglyphs': ('00-information-theory', '02-cuneiform'),
    '02-cuneiform': ('01-egyptian-hieroglyphs', '03-linear-b'),
    '03-linear-b': ('02-cuneiform', '04-maya-glyphs'),
    '04-maya-glyphs': ('03-linear-b', '05-undeciphered-scripts'),
    '05-undeciphered-scripts': ('04-maya-glyphs', '06-computational-methods'),
    '06-computational-methods': ('05-undeciphered-scripts', None),
}

TITLES = {
    '00-information-theory': 'Information Theory of Writing Systems',
    '01-egyptian-hieroglyphs': 'Egyptian Hieroglyphs &amp; the Rosetta Stone',
    '02-cuneiform': 'Cuneiform &amp; the Behistun Inscription',
    '03-linear-b': 'Linear B &mdash; Kober &amp; Ventris',
    '04-maya-glyphs': 'Maya Glyphs &mdash; Knorozov',
    '05-undeciphered-scripts': 'Undeciphered Scripts',
    '06-computational-methods': 'Computational Decipherment',
}


def sidebar_html(active_slug):
    links = []
    for slug, label in SIDEBAR_ITEMS:
        cls = ' active' if slug == active_slug else ''
        links.append(f'            <a href="../{slug}/index.html" class="sidebar-link{cls}">{label}</a>')
    return '\n'.join(links)


def prev_next_html(slug):
    prev_slug, next_slug = PREV_NEXT[slug]
    parts = []
    if prev_slug:
        parts.append(f'        <a href="../{prev_slug}/index.html" class="tutorial-nav-prev">← {TITLES[prev_slug]}</a>')
    else:
        parts.append('        <span></span>')
    if next_slug:
        parts.append(f'        <a href="../{next_slug}/index.html" class="tutorial-nav-next">{TITLES[next_slug]} →</a>')
    else:
        parts.append('        <span></span>')
    return '\n'.join(parts)


def build_page(slug, title, description, theory_html, exercises_html, code_html=''):
    """Build a complete chapter page.
    
    Args:
        slug: Directory name like '00-information-theory'
        title: Page title (plain text, no HTML entities)
        description: Meta description
        theory_html: The theory tab content (raw HTML)
        exercises_html: The exercises tab content (raw HTML)
        code_html: Optional code tab content
    """
    breadcrumb_title = TITLES[slug].replace('&amp;', '&').replace('&mdash;', '—')
    
    if not code_html:
        code_html = '''<div class="placeholder-content">
                    <p>Code examples for this chapter are integrated into the theory section above.
                    Select the <strong>Theory</strong> tab to see worked examples with step-by-step analysis.</p>
                </div>'''

    return f'''<!DOCTYPE html>
<html lang="en" data-theme="light">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title} | Deciphering Ancient Scripts</title>
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
    <link rel="icon" type="image/svg+xml" href="data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'><text y='.9em' font-size='90'>&#x221E;</text></svg>">
</head>
<body>
    <nav class="navbar" id="navbar">
        <div class="nav-container">
            <a href="../../../index.html" class="nav-logo">
                <span class="logo-symbol">&#x2207;</span>
                <span class="logo-text">ML Fundamentals</span>
            </a>
            <button class="nav-toggle" id="navToggle" aria-label="Toggle navigation">
                <span></span><span></span><span></span>
            </button>
            <div class="nav-menu" id="navMenu">
                <div class="nav-links">
                    <a href="../../../tutorials/ml/index.html" class="nav-link">Machine Learning</a>
                    <a href="../../../tutorials/diffusion/index.html" class="nav-link">Diffusion Models</a>
                    <a href="../../../tutorials/decipherment/index.html" class="nav-link active">Decipherment</a>
                    <a href="../../../tutorials/linear-algebra/index.html" class="nav-link">Linear Algebra</a>
                    <a href="../../../tutorials/calculus/index.html" class="nav-link">Calculus</a>
                    <a href="../../../tutorials/physics/index.html" class="nav-link">Physics</a>
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
                <span class="breadcrumb-separator">&#x2192;</span>
                <a href="../index.html">Deciphering Ancient Scripts</a>
                <span class="breadcrumb-separator">&#x2192;</span>
                <span>{breadcrumb_title}</span>
            </nav>
            <div class="tutorial-tabs">
                <a href="#theory" class="tutorial-tab active">Theory</a>
                <a href="#code" class="tutorial-tab">Code</a>
                <a href="#exercises" class="tutorial-tab">Exercises</a>
            </div>
        </div>
    </header>
    <div class="tutorial-wrapper">
        <aside class="tutorial-sidebar">
            <div class="sidebar-section">
                <h3 class="sidebar-section-title">Deciphering Scripts</h3>
                <nav class="sidebar-nav">
{sidebar_html(slug)}
                </nav>
            </div>
        </aside>
        <main class="tutorial-main">
            <article class="article-content" id="theory">
{theory_html}
            </article>
            <article class="article-content" id="code" style="display:none;">
                {code_html}
            </article>
            <article class="article-content" id="exercises" style="display:none;">
                <h2>Exercises</h2>
                <p>Test your understanding with 30 problems across three difficulty levels.</p>
{exercises_html}
            </article>
            <div class="tutorial-nav">
{prev_next_html(slug)}
            </div>
        </main>
        <aside class="toc-container">
            <h4 class="toc-title">Contents</h4>
            <nav class="toc-list"></nav>
        </aside>
    </div>
    <footer class="footer">
        <div class="container">
            <div class="footer-content">
                <div class="footer-brand">
                    <span class="logo-symbol">&#x2207;</span>
                    <span>ML Fundamentals</span>
                </div>
                <p class="footer-tagline">Deep understanding through first principles.</p>
            </div>
            <div class="footer-links">
                <a href="../../../index.html">Home</a>
            </div>
        </div>
    </footer>
    <script src="../../../js/main.js"></script>
    <script>
        document.addEventListener("DOMContentLoaded", function() {{
            if (typeof renderMathInElement === 'function') {{
                renderMathInElement(document.body, {{
                    delimiters: [
                        {{left: '$$', right: '$$', display: true}},
                        {{left: '$', right: '$', display: false}}
                    ],
                    throwOnError: false
                }});
            }}
        }});
    </script>
</body>
</html>'''


def write_chapter(slug, title, description, theory_html, exercises_html, code_html=''):
    """Write a chapter HTML file to disk."""
    outdir = ROOT / slug
    outdir.mkdir(parents=True, exist_ok=True)
    page = build_page(slug, title, description, theory_html, exercises_html, code_html)
    outpath = outdir / 'index.html'
    outpath.write_text(page)
    lines = page.count('\n') + 1
    print(f"  Wrote {outpath} ({lines} lines, {len(page)} bytes)")
    return outpath


def exercise_block(number, difficulty, title, question_html, solution_html):
    """Generate a single exercise item.
    
    difficulty: 'easy' | 'medium' | 'hard'
    """
    emoji = {'easy': '&#x1F7E2;', 'medium': '&#x1F7E1;', 'hard': '&#x1F534;'}[difficulty]
    return f'''                <div class="exercise-item">
                    <div class="exercise-header" onclick="this.parentElement.classList.toggle('open')">
                        <span class="exercise-difficulty">{emoji}</span>
                        <span class="exercise-title">{number}. {title}</span>
                        <span class="exercise-toggle">+</span>
                    </div>
                    <div class="exercise-solution">
                        {solution_html}
                    </div>
                </div>'''


def exercises_section(exercises):
    """Build the full exercises HTML from a list of (title, question_html, solution_html, difficulty) tuples.
    
    exercises should be a list of 30 items: 10 easy, 10 medium, 10 hard (in that order).
    """
    parts = []
    parts.append('                <h3>Easy &#x1F7E2;</h3>')
    for i, (title, q, s, d) in enumerate(exercises[:10], 1):
        combined = f'<p>{q}</p>'
        parts.append(exercise_block(i, 'easy', title, combined, s))
    
    parts.append('                <h3>Medium &#x1F7E1;</h3>')
    for i, (title, q, s, d) in enumerate(exercises[10:20], 11):
        parts.append(exercise_block(i, 'medium', title, q, s))
    
    parts.append('                <h3>Hard &#x1F534;</h3>')
    for i, (title, q, s, d) in enumerate(exercises[20:30], 21):
        parts.append(exercise_block(i, 'hard', title, q, s))
    
    return '\n'.join(parts)
