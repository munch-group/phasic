"""
Tests XXXX
"""


from phasic import Graph
from phasic.test_utils import (
    bin_coef,
    AssertResources,
    coalescent_manual_construction,
    coalescent_callback,
    coalescent_callback_with_ipv,
    coalescent_callback_with_abbr_ipv,
    coalescent_callback_parameterized,
)
from pytest import approx, raises
import numpy as np
import os




# class TestMaxResourcesUsed:
#     """Test how well resources are exploited."""

#     def test_resource_usage(self):

#         with AssertResources(max_peak_mem=3000, min_peak_cpu=110, min_peak_cpu_per_core=50):
#             for i in range(100):
#                 graph = Graph(coalescent_callback_parameterized, ipv=[4, 0, 0, 0])
#                 graph.expectation()
