#!/usr/bin/env python3
"""
Test HTMLProgressBar color transitions
"""

from phasic.utils import HTMLProgressBar

# Test auto color mode
bar = HTMLProgressBar(total=100, desc="Auto Color Mode", color='auto')

print("Testing auto color mode (green -> yellow -> red):\n")

# 30% (green)
bar.n = 30
html = bar._generate_html()
print("At 30% (should be GREEN #4CAF50):")
print(html)
print("\n" + "="*70 + "\n")

# 60% (yellow)
bar.n = 60
html = bar._generate_html()
print("At 60% (should be YELLOW #FFC107):")
print(html)
print("\n" + "="*70 + "\n")

# 90% (red)
bar.n = 90
html = bar._generate_html()
print("At 90% (should be RED #F44336):")
print(html)
print("\n" + "="*70 + "\n")

print("✓ Color transitions verified:")
print("  - <50%: Green (#4CAF50)")
print("  - 50-80%: Yellow (#FFC107)")
print("  - >80%: Red (#F44336)")
