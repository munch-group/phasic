#!/usr/bin/env python
"""Test numerical stability improvements"""

from phasic import Graph
from phasic.logging_config import set_log_level
import numpy as np

# Enable DEBUG to see stability diagnostics
set_log_level('DEBUG')

print('Testing numerical stability improvements...\n')

# Create simple exponential distribution
g = Graph(1)
v0 = g.starting_vertex()
v1 = g.find_or_create_vertex([1])
v2 = g.find_or_create_vertex([2])
v0.add_edge(v1, 1.0)
v1.add_edge(v2, 2.0)

print(f'Graph: {g.vertices_length()} vertices\n')

# Test PDF with auto granularity (should see DEBUG output)
print('Computing PDF at t=1.0 (auto granularity)...')
pdf = g.pdf(1.0, granularity=0)
print(f'PDF: {pdf:.10e}')

# Expected: 2*exp(-2*1) = 0.2706705665
expected = 2.0 * np.exp(-2.0)
error = abs(pdf - expected) / expected
print(f'Expected: {expected:.10e}')
print(f'Relative error: {error:.2e}\n')

if error < 0.01:  # Less than 1% error
    print('✓ All tests passed!')
else:
    print(f'✗ Error too large: {error:.2%}')
