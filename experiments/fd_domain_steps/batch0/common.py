"""Plan A Batch 0 shared helpers (no source change; master ff049679).

Import order matters: phasic must be imported BEFORE any jnp array is made,
or arrays are float32 and the FFI rejects them (the artefact that blanked
review 6's pmf_from_graph rows)."""
import os, re, logging, warnings
warnings.filterwarnings("ignore")
import jax
jax.config.update("jax_enable_x64", True)
import numpy as np
import phasic
from phasic import Graph, set_log_level
import jax.numpy as jnp
set_log_level("ERROR")
logging.getLogger("jax._src.callback").setLevel(logging.CRITICAL)
logging.getLogger("jax").setLevel(logging.CRITICAL)
logging.getLogger("phasic").setLevel(logging.CRITICAL)

HERE = os.path.dirname(os.path.abspath(__file__))

# Messages that identify a refusal of an invalid weight, normalised to a tag.
_TAGS = [
    ("NEGATIVE transition rate", "neg-rate(daisy)"),
    ("non-positive value", "nonpos-weight(add_edge)"),
    ("products to be positive", "log-nonpos"),
    ("negative weight", "neg-weight"),
    ("outgoing rate <= 1", "rowsum>1"),
    ("finite, non-negative", "nonfinite/neg"),
    ("must be non-negative", "neg"),
    ("negative", "neg(other)"),
    ("non-finite", "nonfinite"),
    ("nan", "nan(other)"),
]


def tag(exc):
    msg = str(exc).replace("\n", " ")
    for k, t in _TAGS:
        if k.lower() in msg.lower():
            return "RAISE:" + t
    m = re.sub(r"\s+", " ", msg)
    return "RAISE:" + type(exc).__name__ + ":" + m[-90:]


def outcome(f, *a):
    """Run f(*a); 'ok', 'ok(nan)' or 'RAISE:<tag>'. Forces the value."""
    try:
        v = f(*a)
        leaves = jax.tree_util.tree_leaves(v)
        arr = np.concatenate([np.ravel(np.asarray(x, dtype=np.float64)) for x in leaves])
        if np.all(np.isfinite(arr)):
            return "ok"
        return "ok(nan)" if np.any(np.isnan(arr)) else "ok(inf)"
    except Exception as e:  # noqa: BLE001  (recording every refusal is the point)
        return tag(e)


def rel(a, b):
    return abs(a - b) / max(abs(b), 1e-300)
