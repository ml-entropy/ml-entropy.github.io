# CLAUDE.md

## Project Overview

This is **ml-entropy.github.io**, an educational static website hosted on GitHub Pages. It teaches Machine Learning, Linear Algebra, Calculus, and Physics from an information-theoretic perspective — everything is framed through entropy, compression, and efficient representation.

The site is built with **vanilla HTML/CSS/JS** (no framework, no bundler, no static site generator). GitHub Pages serves the `docs/` folder directly. A `.nojekyll` file disables Jekyll processing.

## Repository Structure

```
/
├── docs/                          # DEPLOYED SITE (GitHub Pages serves this)
│   ├── index.html                 # Homepage
│   ├── css/
│   │   ├── main.css               # Global styles, CSS variables, themes
│   │   ├── components.css         # Reusable component styles
│   │   ├── sidebar.css            # Tutorial 3-column layout
│   │   └── style.css              # Legacy styles
│   ├── js/
│   │   └── main.js                # Theme toggle, mobile nav, TOC, KaTeX
│   └── tutorials/
│       ├── ml/                    # 27 ML tutorials + index
│       ├── linear-algebra/        # 8 tutorials + index
│       ├── calculus/              # 4 tutorials + index
│       └── physics/               # 3 tutorials + index
│
├── tutorials/                     # SOURCE CONTENT (Jupyter notebooks, markdown)
├── linear_algebra_tutorials/      # Source content for linear algebra
├── calculus_tutorials/            # Source content for calculus
├── physics_tutorials/             # Source content for physics
│
├── update_sidebars.py             # Regenerates sidebar navigation across all tutorials
├── check_links.py                 # Validates all internal links in docs/
├── requirements.txt               # Python deps: numpy, matplotlib, scipy, torch, jupyter
├── index.html                     # Root landing page (redirects into docs/)
├── .nojekyll                      # Disables Jekyll on GitHub Pages
│
├── .github/workflows/             # Gemini AI integration workflows
└── .gemini/                       # Gemini AI settings
```

**Key distinction**: `tutorials/` etc. contain raw source (notebooks, markdown). `docs/tutorials/` contains the deployed HTML that users see.

## Technology Stack

| Layer | Technology |
|-------|-----------|
| Frontend | Vanilla HTML5, CSS3, JavaScript |
| Math | KaTeX v0.16.9 (CDN) |
| Fonts | Google Fonts: Inter, Playfair Display, JetBrains Mono |
| Hosting | GitHub Pages (from `docs/` folder) |
| Source content | Jupyter Notebooks, Markdown |
| Automation | Python 3 scripts |
| CI | GitHub Actions (Gemini AI review/triage) |

## Development Workflows

### Serving locally

```bash
cd docs/
python -m http.server 8000
# Visit http://localhost:8000
```

No build step is required — the HTML files in `docs/` are the final output.

### Validating links

```bash
python check_links.py
```

Scans all HTML in `docs/` and checks that internal `href` targets resolve to existing files.

### Updating sidebar navigation

```bash
python update_sidebars.py
```

Regenerates the `<aside class="tutorial-sidebar">` block in every tutorial's `index.html`. The navigation order and display titles are defined in the `NAV_LINKS` dictionary inside `update_sidebars.py`.

### Adding a new tutorial

1. Create source content in the appropriate source directory (e.g., `tutorials/XX_topic_name/`)
2. Create `docs/tutorials/<category>/XX-topic-name/index.html` following the existing template structure
3. Add the entry to `NAV_LINKS` in `update_sidebars.py`
4. Run `python update_sidebars.py` to update all sidebars
5. Run `python check_links.py` to verify no broken links
6. Commit and push to deploy

### Deployment

Push to `main` — GitHub Pages automatically deploys from the `docs/` folder.

## HTML Template Structure

Every tutorial page follows this three-column layout:

```html
<!DOCTYPE html>
<html lang="en" data-theme="light">
<head>
    <!-- KaTeX CDN, Google Fonts, CSS links -->
    <link rel="stylesheet" href="../../../css/main.css">
    <link rel="stylesheet" href="../../../css/components.css">
    <link rel="stylesheet" href="../../../css/sidebar.css">
</head>
<body>
    <nav class="navbar"><!-- Top navigation bar --></nav>
    <div class="tutorial-wrapper">
        <aside class="tutorial-sidebar"><!-- Left: tutorial nav --></aside>
        <main class="tutorial-main"><!-- Center: content --></main>
        <aside class="toc-container"><!-- Right: table of contents --></aside>
    </div>
</body>
</html>
```

- Left sidebar (280px): Links to other tutorials in the same category + cross-links
- Center: Main tutorial content
- Right sidebar (240px): Sticky table of contents
- Responsive: collapses to single column at 1024px

## CSS Architecture

- **main.css** — CSS variables (colors, spacing), base styles, navbar, cards, buttons. Defines both light and dark theme variables via `[data-theme]`.
- **components.css** — Tutorial components: exercise blocks, code blocks, tables, equations, info boxes.
- **sidebar.css** — Three-column grid layout, sticky sidebars, responsive breakpoints.
- **style.css** — Legacy fallback styles (minimal).

### Theme system

- Dark mode via `data-theme="light|dark"` attribute on `<html>`
- Stored in `localStorage` key `'theme'`
- Respects `prefers-color-scheme` system preference
- All colors use CSS custom properties that swap with theme

### Fonts

- Sans-serif: Inter
- Serif: Playfair Display
- Monospace: JetBrains Mono

## Naming Conventions

| Context | Pattern | Example |
|---------|---------|---------|
| Source folders | `NN_topic_with_underscores` | `01_entropy_fundamentals` |
| Deployed folders | `NN-topic-with-hyphens` | `01-entropy` |
| Tutorial categories | lowercase with hyphens | `linear-algebra`, `ml` |
| CSS classes | lowercase with hyphens | `tutorial-sidebar`, `sidebar-link` |

## Navigation System

The `NAV_LINKS` dictionary in `update_sidebars.py` is the single source of truth for tutorial ordering and display names. Categories: `ml`, `linear-algebra`, `calculus`, `physics`.

When adding or reordering tutorials, edit `NAV_LINKS` and run `python update_sidebars.py`. The script uses regex to find and replace the sidebar `<aside>` block in every tutorial HTML file.

## Math Rendering

KaTeX auto-renders with these delimiters:
- Display math: `$$...$$` and `\[...\]`
- Inline math: `$...$` and `\(...\)`

## Key Files Reference

| File | Purpose |
|------|---------|
| `docs/css/main.css` | Global styles and theme variables |
| `docs/css/components.css` | Tutorial component styles |
| `docs/css/sidebar.css` | Layout grid and responsive rules |
| `docs/js/main.js` | Theme toggle, mobile nav, TOC generation, scroll effects |
| `update_sidebars.py` | Sidebar generation script (contains `NAV_LINKS`) |
| `check_links.py` | Link validation script |
| `index.html` (root) | Landing page |
| `.nojekyll` | GitHub Pages Jekyll bypass |

## Important Notes for AI Assistants

- **Do not modify `docs/` HTML without understanding the template**. All tutorial pages share the same structure. Changes to one should typically be propagated to all via `update_sidebars.py` or a similar bulk operation.
- **Run `python check_links.py` after any structural changes** to catch broken links.
- **Run `python update_sidebars.py` after adding or renaming tutorials** to keep navigation consistent.
- **The `docs/` folder is the deployed site**. Any file change there goes live on the next push to `main`.
- **No build pipeline exists** — what you see in `docs/` is what gets served. There is no compilation, bundling, or transformation step.
- **CSS variables in `main.css` control theming**. Always use the existing CSS custom properties rather than hardcoded colors.
- **KaTeX is loaded from CDN**. Math expressions in tutorial HTML must use the supported delimiter syntax.
