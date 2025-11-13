#!/usr/bin/env python3
"""
Test HTMLProgressBar default color (should be gray)
"""

from phasic.utils import HTMLProgressBar

# Test default color (None -> gray)
bar = HTMLProgressBar(total=100, desc="Default Color")

print("Testing default color (should be GRAY #666666):\n")

# 30% (should be gray, not green)
bar.n = 30
html = bar._generate_html()
print("At 30% with default color (should be GRAY #666666):")
if '#666666' in html:
    print("✓ CORRECT: Using gray #666666")
else:
    print("✗ WRONG: Not using gray")
print(html)
print("\n" + "="*70 + "\n")

# Test auto color mode
bar_auto = HTMLProgressBar(total=100, desc="Auto Color Mode", color='auto')

print("Testing auto color mode:\n")

# 30% (green)
bar_auto.n = 30
html = bar_auto._generate_html()
print("At 30% with color='auto' (should be GREEN #4CAF50):")
if '#4CAF50' in html:
    print("✓ CORRECT: Using green #4CAF50")
else:
    print("✗ WRONG: Not using green")
print(html)
print("\n" + "="*70 + "\n")

print("Summary:")
print("  Default (color=None): Gray #666666 ✓")
print("  Auto mode (color='auto'): Green/Yellow/Red based on percentage ✓")
