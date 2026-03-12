# Opus Tutorials

**Opus Tutorials** is a comprehensive educational project designed to teach Machine Learning, Calculus, Linear Algebra, and Physics fundamentals through the lens of Information Theory (Entropy, Compression, Surprise).

The project is structured as a static website hosted in the `docs/` directory, supported by Python scripts for automation and maintenance.

## 🏗 Project Architecture

*   **Static Site (`docs/`)**: The public-facing website. It contains the HTML, CSS, and JavaScript files served to users.
    *   `docs/tutorials/`: Hierarchy of tutorials grouped by subject (`ml`, `calculus`, `linear-algebra`, `physics`).
    *   `docs/css/` & `docs/js/`: Styling and interactivity.
*   **Source Content (`tutorials/`, `*_tutorials/`)**: The raw educational content, primarily in Jupyter Notebooks (`.ipynb`) and Markdown (`.md`) files.
*   **Automation Scripts**: Python tools in the root directory used to maintain the site structure, update navigation sidebars, and generate exercise content.

## 🛠 Building and Maintenance

This project does not use a standard static site generator like Jekyll or Hugo. Instead, it relies on custom Python scripts to manipulate the HTML structure.

### Key Scripts

*   `update_sidebars.py`: Scans the `docs/tutorials/` directory and updates the sidebar navigation HTML in every `index.html` file. This ensures all tutorials link to each other correctly.
    *   **Usage:** `python update_sidebars.py`
*   `generate_detailed_exercises.py`: Generates HTML blocks for exercises (question + solution). It contains the content for exercises in Python strings and outputs formatted HTML.
*   `replace_detailed_exercises.py`: Automates the insertion/update of exercise sections within the HTML files.
*   `check_links.py`: Validates internal links to ensure site integrity.

### Dependencies

The project requires Python 3 and the following libraries (for notebooks and generation):

```text
numpy>=1.21.0
matplotlib>=3.5.0
scipy>=1.7.0
torch>=1.10.0
jupyter>=1.0.0
```

Install them via:
```bash
pip install -r requirements.txt
```

## 📂 Directory Structure

*   `tutorials/`: Main content source for ML tutorials (e.g., `01_entropy_fundamentals`).
*   `calculus_tutorials/`: Source content for Calculus.
*   `linear_algebra_tutorials/`: Source content for Linear Algebra.
*   `physics_tutorials/`: Source content for Physics.
*   `docs/`: The deployed build artifact. **Do not manually delete this** unless performing a full rebuild.
    *   `docs/tutorials/ml/`: Machine Learning HTML pages.
    *   `docs/tutorials/calculus/`: Calculus HTML pages.
    *   ... (and so on for other subjects)

## 📝 Development Conventions

*   **Content First:** Tutorials usually start as Jupyter Notebooks to ensure code correctness.
*   **HTML Structure:** The `index.html` files in `docs/` use specific classes (`tutorial-wrapper`, `tutorial-sidebar`, `tutorial-main`) which the Python scripts rely on for regex-based updates.
*   **Math Rendering:** Uses KaTeX for rendering mathematical notation (configured in the HTML headers).
*   **Difficulty Badges:** Exercises are categorized as Easy (🟢), Medium (🟡), or Hard (🔴).

## ✍️ Tutorial Writing Style

**This is critical.** When asked to write or expand a tutorial, you must produce **comprehensive, in-depth educational content** — not summaries or bullet-point outlines. Treat every tutorial as a university-level lecture that a student will read from start to finish.

### Length and Depth Requirements

- **Minimum 800–1200 lines of HTML** for a full tutorial page. Never produce a short, skeletal page.
- Each section must contain **multiple paragraphs** of explanation, not just one or two sentences.
- Explain the **"why"** behind every concept, not just the "what." Why does this formula look like this? What would go wrong with a different choice? What historical alternatives existed?
- Include **step-by-step mathematical derivations** with intermediate steps explained in words (use `<div class="math-derivation">` blocks).
- Build **intuition before formalism**: start each concept with a concrete analogy, real-world example, or thought experiment, then transition to the math.
- After a derivation, add a **"What does this mean?"** paragraph connecting the result back to intuition.

### Structure for Each Section

1. **Motivating question or scenario** — hook the reader with a concrete problem or paradox.
2. **Intuitive explanation** — explain the concept as you would to a curious non-expert, using analogies and examples.
3. **Formal definition / derivation** — present the math rigorously, with every step justified.
4. **Worked examples** — at least one concrete numerical example per major concept.
5. **Common misconceptions** — address what students typically get wrong.
6. **Connection to the bigger picture** — how does this relate to the information-theoretic thread of the project?

### What NOT to Do

- **Never** write one-sentence paragraphs for conceptual explanations. Each paragraph should be 3–6 sentences minimum.
- **Never** skip "obvious" steps in derivations — what's obvious to you is not obvious to the reader.
- **Never** list facts without explaining their significance.
- **Never** produce a skeleton/outline and call it done. The tutorial should be **publication-ready** as written.
- **Never** sacrifice depth for brevity. When in doubt, explain more, not less.

### Tone

- Conversational but rigorous — like a great professor who respects the reader's intelligence.
- Use rhetorical questions to guide thinking: "But wait — why should we use log and not some other function?"
- Refer to the reader directly: "Notice that...", "You might wonder...", "Let's verify this..."

## 🚀 Common Tasks

**Adding a New Tutorial:**
1.  Create the content (Notebook/Markdown).
2.  Create a new folder in `docs/tutorials/[category]/[topic-name]/`.
3.  Add an `index.html` (copying a template from an existing tutorial).
4.  Register the new tutorial in the `NAV_LINKS` dictionary within `update_sidebars.py`.
5.  Run `python update_sidebars.py` to propagate the link to all other pages.

**Updating Exercises:**
1.  Modify the exercise definitions in `generate_detailed_exercises.py`.
2.  Run the generation/replacement script (or manually update the HTML if doing ad-hoc changes).
