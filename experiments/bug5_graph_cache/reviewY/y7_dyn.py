"""reviewY y7: the hit-path expression _Graph(1).get_dyn_ordering() exists and follows PHASIC_DYN_ORDERING."""
import os, phasic
from phasic import _Graph
print("env set:", "PHASIC_DYN_ORDERING" in os.environ, "-> _Graph(1).get_dyn_ordering() =", _Graph(1).get_dyn_ordering())
