import os
import re
from pathlib import Path

ROOT_DIR = Path("docs")

def check_links():
    broken_links = []
    
    # Compile regex - finding href="..." or href='...'
    link_pattern = re.compile(r'href=[\'"](.*?)[\'"]')
    
    for html_file in ROOT_DIR.rglob("*.html"):
        try:
            with open(html_file, 'r', encoding='utf-8') as f:
                content = f.read()
        except Exception as e:
            print(f"Error reading {html_file}: {e}")
            continue

        links = link_pattern.findall(content)
        
        for link in links:
            if link.startswith(('http', 'https', '#', 'mailto:', 'javascript:', 'data:')):
                continue
            
            target_path = None
            if link.startswith('/'):
                # Handle absolute path relative to project root
                # Assuming / starts at docs/
                # Remove leading slash to join
                clean_link = link.lstrip('/')
                target_path = ROOT_DIR / clean_link
            else:
                # Relative link
                # Resolve relative to current file's parent directory
                target_path = (html_file.parent / link).resolve()
            
            # Remove anchor part
            path_str = str(target_path).split('#')[0]
            link_path = Path(path_str)
            
            # If directory, assume index.html
            if link_path.is_dir():
                link_path = link_path / "index.html"
                
            # Check if file exists
            if not link_path.exists():
                try:
                    # Try to get relative path for display
                    display_path = link_path.relative_to(Path.cwd())
                except:
                    display_path = link_path
                    
                broken_links.append({
                    'file': str(html_file),
                    'link': link,
                    'resolved': str(display_path)
                })

    return broken_links

if __name__ == "__main__":
    issues = check_links()
    if issues:
        print(f"Found {len(issues)} broken links:")
        for issue in issues:
            print(f"File: {issue['file']}")
            print(f"  Link: {issue['link']}")
            print(f"  Resolved: {issue['resolved']}")
            print("-" * 20)
    else:
        print("No broken links found.")