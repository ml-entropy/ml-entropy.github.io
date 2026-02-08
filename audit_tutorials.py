#!/usr/bin/env python3
"""
Tutorial Template Audit Script
Checks all tutorial HTML files against the reference template structure.
"""

import os
import re
from pathlib import Path
from typing import Dict, List, Tuple

# Reference template
REFERENCE_TEMPLATE = "docs/tutorials/ml/01-entropy/index.html"

# Required elements to check
REQUIRED_ELEMENTS = {
    'html_theme': r'<html[^>]*data-theme="light"',
    'meta_charset': r'<meta charset="UTF-8">',
    'meta_viewport': r'<meta name="viewport"',
    'meta_description': r'<meta name="description"',
    'fonts_google': r'fonts\.googleapis\.com',
    'katex_css': r'katex.*\.css',
    'katex_js': r'katex.*\.js',
    'css_main': r'css/main\.css',
    'css_components': r'css/components\.css',
    'css_sidebar': r'css/sidebar\.css',
    'navbar': r'<nav class="navbar"',
    'breadcrumb': r'<nav class="breadcrumb"',
    'tutorial_tabs': r'<div class="tutorial-tabs"',
    'tutorial_wrapper': r'<div class="tutorial-wrapper"',
    'tutorial_sidebar': r'<aside class="tutorial-sidebar"',
    'toc_container': r'<aside class="toc-container"',
    'theme_toggle': r'<button class="theme-toggle"',
}

def check_file(filepath: str) -> Dict[str, bool]:
    """Check a single file against the template requirements."""
    results = {}
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        for key, pattern in REQUIRED_ELEMENTS.items():
            results[key] = bool(re.search(pattern, content, re.IGNORECASE | re.DOTALL))
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return {}
    
    return results

def audit_tutorials(base_dir: str) -> Dict[str, Dict[str, bool]]:
    """Audit all tutorial files."""
    results = {}
    
    # Find all index.html files in docs/tutorials
    tutorials_dir = Path(base_dir) / "docs" / "tutorials"
    
    for html_file in tutorials_dir.rglob("index.html"):
        rel_path = str(html_file.relative_to(base_dir))
        results[rel_path] = check_file(str(html_file))
    
    return results

def generate_report(results: Dict[str, Dict[str, bool]]) -> str:
    """Generate a human-readable report."""
    report = []
    report.append("=" * 80)
    report.append("TUTORIAL TEMPLATE AUDIT REPORT")
    report.append("=" * 80)
    report.append("")
    
    # Calculate statistics
    total_files = len(results)
    
    # Track missing elements per file
    files_with_issues = []
    
    for filepath, checks in sorted(results.items()):
        missing = [key for key, present in checks.items() if not present]
        
        if missing:
            files_with_issues.append((filepath, missing))
    
    report.append(f"Total files checked: {total_files}")
    report.append(f"Files with issues: {len(files_with_issues)}")
    report.append(f"Files compliant: {total_files - len(files_with_issues)}")
    report.append("")
    
    if files_with_issues:
        report.append("ISSUES BY FILE:")
        report.append("-" * 80)
        
        for filepath, missing in files_with_issues:
            report.append(f"\n{filepath}")
            report.append(f"  Missing elements: {len(missing)}")
            for elem in missing:
                report.append(f"    - {elem}")
    else:
        report.append("✅ All files are compliant with the template!")
    
    report.append("")
    report.append("=" * 80)
    report.append("SUMMARY BY ELEMENT:")
    report.append("=" * 80)
    
    # Count how many files are missing each element
    element_counts = {}
    for elem in REQUIRED_ELEMENTS.keys():
        missing_count = sum(1 for checks in results.values() if not checks.get(elem, False))
        element_counts[elem] = missing_count
    
    for elem, count in sorted(element_counts.items(), key=lambda x: -x[1]):
        if count > 0:
            report.append(f"{elem:30s}: {count:2d} files missing")
    
    report.append("")
    
    return "\n".join(report)

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    results = audit_tutorials(base_dir)
    report = generate_report(results)
    print(report)
    
    # Save report to file
    report_file = os.path.join(base_dir, "tutorial_audit_report.txt")
    with open(report_file, 'w') as f:
        f.write(report)
    print(f"\nReport saved to: {report_file}")
