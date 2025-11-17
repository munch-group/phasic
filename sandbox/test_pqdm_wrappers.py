#!/usr/bin/env python3
"""
Test pqdm and prange tqdm wrappers
"""

import time
from phasic import pqdm, prange

print("=" * 70)
print("Test 1: prange wrapper")
print("=" * 70)

for i in prange(10, desc="Processing items"):
    time.sleep(0.1)

print("\n✓ prange test complete\n")

print("=" * 70)
print("Test 2: pqdm wrapper")
print("=" * 70)

items = list(range(15))
for item in pqdm(items, desc="Processing list", total=len(items)):
    time.sleep(0.1)

print("\n✓ pqdm test complete\n")

print("=" * 70)
print("Test 3: pqdm with custom bar_format")
print("=" * 70)

for item in pqdm(items, desc="Custom format", bar_format="{desc}: |{bar}| {n}/{total}"):
    time.sleep(0.05)

print("\n✓ Custom format test complete\n")

print("=" * 70)
print("SUCCESS: All pqdm/prange wrapper tests passed!")
print("=" * 70)
