"""
Tests XXXX
"""


from phasic import Graph
from phasic.test_utils import (
    bin_coef,
    ResourceMonitor,
    coalescent_manual_construction,
    coalescent_callback,
    coalescent_callback_with_ipv,
    coalescent_callback_with_abbr_ipv,
    coalescent_callback_parameterized,
)
import numpy as np

def callback(state):
    ...

class TestXXXX:
    """Test XXXX"""

    def test_XXXX(self):
        ...


class TestMaxResourcesUsed:
    """Test how well resources are exploited."""

    def test_max_cpu_mem_usage(self):

        graph = Graph(coalescent_callback_parameterized, ipv=[4, 0, 0, 0])

        # FOR TESTING:
        with ResourceMonitor(interval=0.05) as used:
            for i in range(10000000):
                pass
        # ACTUAL CODE:
        # with ResourceMonitor(interval=0.05) as used:
        #     svgd = graph.svgd()

        max_cpu_percent = used.max_cpu
        max_mem_gb = used.max_rss/(1024**3)
        print(f'Max CPU: {max_cpu_percent}%, Max memory: {max_mem_gb} GB')

        # FOR TESTING:
        assert max_cpu_percent > 99 # use more than one CPU
        assert max_mem_gb < 1 # max memory < 1GB
        # ACTUAL CODE:
        # assert max_cpu_percent > 110 # use more than one CPU
        # assert max_mem_gb < 1 # max memory < 1GB
