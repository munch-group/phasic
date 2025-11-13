#!/usr/bin/env python3
"""
Test HTMLProgressBar class HTML generation
"""

from phasic.utils import HTMLProgressBar

# Test HTML generation
bar = HTMLProgressBar(total=100, desc="Test Progress", color="#4CAF50")

print("Initial HTML (0%):")
print(bar._generate_html())
print("\n" + "="*70 + "\n")

# Simulate 50% progress
bar.n = 50
print("HTML at 50%:")
print(bar._generate_html())
print("\n" + "="*70 + "\n")

# Simulate 100% progress
bar.n = 100
print("HTML at 100%:")
print(bar._generate_html())
print("\n" + "="*70 + "\n")

print("✓ HTML generation test complete")
print("\nThe HTML above should render as thin, sleek progress bars in notebooks")
print("matching the style of cpu_monitor.py")
