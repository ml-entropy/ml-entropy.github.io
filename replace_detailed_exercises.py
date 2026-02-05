
import os

target_file = 'docs/tutorials/ml/01-entropy/index.html'
exercises_file = 'new_exercises_detailed.html'

with open(target_file, 'r') as f:
    content = f.read()

with open(exercises_file, 'r') as f:
    new_exercises = f.read()

# We need to find the start and end of the CURRENT exercises section.
# Previous replacement used:
start_marker = '<!-- Exercises Section -->'
# The end marker should be the LAST </section> tag inside the wrapper or just careful matching.
# But wait, I added "30 Practice Exercises" heading last time.
# Let's verify the file content around the exercises section to be sure.
# I'll rely on the comment <!-- Exercises Section --> which I preserved.

start_idx = content.find(start_marker)
if start_idx == -1:
    print("Error: Start marker not found")
    exit(1)

# Finding the end is trickier because I might have nested sections or just </section>.
# The structure is:
# <section id="exercises" ...>
#    ...
# </section>
# <!-- Navigation Buttons -->

# I can search for "<!-- Navigation Buttons -->" and find the rfind('</section>') before it.
nav_marker = '<!-- Navigation Buttons -->'
nav_idx = content.find(nav_marker)

if nav_idx != -1:
    # Find the last </section> before the nav buttons
    end_idx = content.rfind('</section>', start_idx, nav_idx)
    if end_idx == -1:
        print("Error: End marker </section> not found before navigation")
        exit(1)
    # Include the length of </section>
    end_idx += len('</section>')
else:
    # Fallback: Find the first </section> after start_marker?
    # No, that's risky if nested. But here it's likely not nested.
    # The exercises section is top level in main.
    end_idx = content.find('</section>', start_idx) + len('</section>')

# Construct new content
new_content = content[:start_idx] + new_exercises + content[end_idx:]

with open(target_file, 'w') as f:
    f.write(new_content)

print("Successfully replaced exercises section with detailed solutions.")
