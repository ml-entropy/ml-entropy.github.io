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
