"""Where does the prototype wrapper get a wrong-length vector? (which call sites need the unwrapped transform)"""
import traceback, sys, os
sys.path.insert(0, os.path.dirname(__file__))
from r2_lib import *
try:
    fit(2.0, True)
except Exception:
    tb = traceback.format_exc()
    lines = [l for l in tb.splitlines() if 'phasic/' in l or 'review3' in l]
    print('\n'.join(lines[-14:])); print(tb.splitlines()[-1])
