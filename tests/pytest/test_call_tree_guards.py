"""Tests for the parameter-path features of scripts/call_tree_analyzer.py.

Covers:
  A. the three-valued guard evaluator (evaluate_guard / evaluate_value),
  B. branch-guard tracking on synthetic source,
  C. inter-procedural propagation + top-level assignment folding,
  D. a golden path-resolution test on the real Graph.svgd,
  E. an honesty audit (attribute/runtime guards must stay UNKNOWN),
  F. the misattribution fix (self/class preference + ambiguity marking),
  G. backward-compatibility of the no-flag tree and parse_params.

The analyzer lives in scripts/ (not an importable package), so it is loaded by
path via importlib.
"""

import ast
import importlib.util
import os

import pytest

HERE = os.path.dirname(__file__)
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
SCRIPT = os.path.join(ROOT, "scripts", "call_tree_analyzer.py")

spec = importlib.util.spec_from_file_location("call_tree_analyzer", SCRIPT)
cta = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cta)

UNKNOWN = cta.UNKNOWN

_MSPEC = importlib.util.spec_from_file_location(
    "call_graph_to_mermaid", os.path.join(ROOT, "scripts", "call_graph_to_mermaid.py"))
mermaid = importlib.util.module_from_spec(_MSPEC)
_MSPEC.loader.exec_module(mermaid)


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def G(expr, env):
    """Three-valued truth of a guard expression string."""
    return cta.evaluate_guard(ast.parse(expr, mode="eval").body, env)


def V(expr, env):
    """Resolved value (or UNKNOWN) of an expression string."""
    return cta.evaluate_value(ast.parse(expr, mode="eval").body, env)


def make_pkg(tmp_path, source, name="mod.py"):
    (tmp_path / name).write_text(source)
    pa = cta.PackageAnalyzer(str(tmp_path))
    pa.analyze_package()
    return pa


def sites_of(pa, func_name, class_name=None):
    key = (pa.find_method(class_name, func_name) if class_name
           else pa.find_function(func_name))
    assert key is not None, f"{class_name}.{func_name} not found"
    return key, pa.call_graph.get(key, [])


def guard_text_by_callee(pa, func_name, class_name=None):
    """Map callee bare-name -> rendered guard chain for each call site."""
    _key, sites = sites_of(pa, func_name, class_name)
    out = {}
    for s in sites:
        callee = s.call_key.rsplit(":", 1)[-1]
        out.setdefault(callee, []).append(cta._format_guard_chain(s.guards))
    return out


def walk_states(node, acc=None):
    """Collect {label_name: path_state} over a built tree (first occurrence)."""
    if acc is None:
        acc = {}
    name = node.signature.name
    acc.setdefault(name, node.path_state)
    for c in node.children:
        walk_states(c, acc)
    return acc


def find_nodes(node, name):
    out = []
    if node.signature.name == name:
        out.append(node)
    for c in node.children:
        out.extend(find_nodes(c, name))
    return out


# --------------------------------------------------------------------------- #
# A. evaluator unit tests (pure, no filesystem)
# --------------------------------------------------------------------------- #
class TestEvaluator:
    def test_name_truthiness_three_valued(self):
        assert G("discrete", {"discrete": True}) is True
        assert G("discrete", {"discrete": None}) is False
        assert G("discrete", {"discrete": 0}) is False
        assert G("discrete", {}) is None  # unbound -> UNKNOWN, never guessed

    def test_is_not_none_idiom(self):
        assert G("epoch_starts is not None", {"epoch_starts": None}) is False
        assert G("epoch_starts is not None", {"epoch_starts": [0, 1]}) is True
        assert G("epoch_starts is not None", {}) is None

    def test_equality_string(self):
        assert G("weight_mode == 'formula'", {"weight_mode": "formula"}) is True
        assert G("weight_mode == 'formula'", {"weight_mode": "linear"}) is False
        assert G("weight_mode == 'formula'", {}) is None

    def test_kleene_and_or(self):
        # AND: a False operand dominates even when the other is unknown
        assert G("a and b", {"a": False}) is False
        assert G("a and b", {"a": True}) is None        # b unbound
        assert G("a and b", {"a": True, "b": True}) is True
        # OR: a True operand dominates
        assert G("a or b", {"a": True}) is True
        assert G("a or b", {"a": False}) is None         # b unbound
        assert G("a or b", {"a": False, "b": False}) is False

    def test_not(self):
        assert G("not joint_index", {"joint_index": True}) is False
        assert G("not joint_index", {"joint_index": False}) is True
        assert G("not joint_index", {}) is None

    def test_membership(self):
        assert G("x in (1, 2, 3)", {"x": 2}) is True
        assert G("x in (1, 2, 3)", {"x": 9}) is False
        assert G("x not in (1, 2)", {"x": 9}) is True

    @pytest.mark.parametrize("expr", [
        "self.x",                       # Attribute
        "f(x)",                         # Call
        "rewards.ndim == 2",            # Attribute inside compare
        "n * (n - 1) > 0",              # BinOp
        "theta[0] > 1",                 # Subscript
        "[v for v in xs]",              # comprehension
    ])
    def test_whitelist_rejections_are_unknown(self, expr):
        assert G(expr, {"n": 5, "rewards": object(), "theta": [3]}) is None

    def test_never_raises_on_bad_comparison(self):
        # Comparing incomparable types must yield UNKNOWN, not raise.
        assert G("1 < 'a'", {}) is None

    def test_evaluate_value_atoms(self):
        assert V("None", {}) is None
        assert V("3", {}) == 3
        assert V("'hi'", {}) == "hi"
        assert V("x", {"x": None}) is None             # bound to None, not UNKNOWN
        assert V("x", {}) is UNKNOWN                    # unbound
        assert V("(1, 2)", {}) == (1, 2)

    def test_evaluate_value_boolean_for_folding(self):
        # This is exactly the `_callback_overridden = callback is not None` case.
        assert V("callback is not None", {"callback": None}) is False
        assert V("callback is not None", {"callback": object()}) is True
        assert V("callback is not None", {}) is UNKNOWN

    def test_tri_helpers(self):
        assert cta.not3(None) is None
        assert cta.and3(True, None) is None
        assert cta.and3(False, None) is False
        assert cta.or3(None, True) is True


# --------------------------------------------------------------------------- #
# B. guard tracking on synthetic source
# --------------------------------------------------------------------------- #
GUARD_SRC = '''
def f(x, y=None):
    a()
    if x > 0:
        b()
        if y is None:
            c()
    else:
        d()
    for i in items:
        e()
    z = g() if x else h()
'''


class TestGuardTracking:
    def test_guard_chains(self, tmp_path):
        pa = make_pkg(tmp_path, GUARD_SRC)
        gt = guard_text_by_callee(pa, "f")
        assert gt["a"] == [""]                       # unconditional
        assert gt["b"] == ["if x > 0"]
        assert gt["c"] == ["if x > 0 and y is None"]
        assert gt["d"] == ["if not (x > 0)"]         # else branch
        assert gt["e"] == ["for i in items"]
        assert gt["g"] == ["if x"]                   # ternary, then-arm
        assert gt["h"] == ["if not (x)"]             # ternary, else-arm

    def test_line_numbers_recorded(self, tmp_path):
        pa = make_pkg(tmp_path, GUARD_SRC)
        _key, sites = sites_of(pa, "f")
        for s in sites:
            assert s.lineno > 0

    def test_try_body_is_unconditional(self, tmp_path):
        src = (
            "def f():\n"
            "    try:\n"
            "        a()\n"
            "    except ValueError:\n"
            "        b()\n"
        )
        pa = make_pkg(tmp_path, src)
        gt = guard_text_by_callee(pa, "f")
        assert gt["a"] == [""]                        # try body runs unconditionally
        assert gt["b"] == ["except ValueError"]       # handler is conditional

    def test_nested_function_resets_guards(self, tmp_path):
        # A call inside a nested def must not inherit the outer if-guard.
        src = (
            "def outer(x):\n"
            "    if x:\n"
            "        def inner():\n"
            "            a()\n"
        )
        pa = make_pkg(tmp_path, src)
        gt = guard_text_by_callee(pa, "inner")
        assert gt["a"] == [""]


# --------------------------------------------------------------------------- #
# C. propagation + folding
# --------------------------------------------------------------------------- #
PROP_SRC = '''
def caller(flag):
    callee(flag, 5)
    helper(callback=None)
    spread(**opts)

def callee(a, b):
    if a:
        taken_call()
    if b == 5:
        five_call()
    if b == 9:
        unreached_call()

def helper(callback=None):
    _ov = callback is not None
    if _ov:
        ov_call()
    else:
        no_ov_call()

def spread(callback=None):
    if callback is not None:
        spread_inner()

def taken_call(): pass
def five_call(): pass
def unreached_call(): pass
def ov_call(): pass
def no_ov_call(): pass
def spread_inner(): pass
'''


class TestPropagationAndFolding:
    def _states(self, tmp_path):
        pa = make_pkg(tmp_path, PROP_SRC)
        key = pa.find_function("caller")
        ctx = cta.make_root_context(pa.all_functions[key], {"flag": True}, prune=False)
        tree = pa.build_call_tree(key, eval_ctx=ctx, max_depth=6)
        return walk_states(tree)

    def test_positional_and_literal_binding(self, tmp_path):
        st = self._states(tmp_path)
        # callee(flag=True, b=5): both guards resolve to taken
        assert st["taken_call"] == "taken"      # a == True (known name forwarded)
        assert st["five_call"] == "taken"       # b == 5 (literal forwarded)
        assert st["unreached_call"] == "not-taken"  # b == 9 is provably false

    def test_assignment_folding(self, tmp_path):
        st = self._states(tmp_path)
        # helper(callback=None): _ov = (None is not None) = False is folded
        assert st["ov_call"] == "not-taken"
        assert st["no_ov_call"] == "taken"

    def test_kwargs_spread_is_unknown(self, tmp_path):
        st = self._states(tmp_path)
        # spread(**opts): callback unbound -> guard UNKNOWN -> conditional
        assert st["spread_inner"] == "conditional"

    def test_literal_default_seeding(self, tmp_path):
        # A param not supplied by the caller takes its literal default.
        src = (
            "def caller():\n"
            "    callee()\n"
            "def callee(discrete=None):\n"
            "    if discrete:\n"
            "        on()\n"
            "    else:\n"
            "        off()\n"
            "def on(): pass\n"
            "def off(): pass\n"
        )
        pa = make_pkg(tmp_path, src)
        key = pa.find_function("caller")
        ctx = cta.make_root_context(pa.all_functions[key], {}, prune=False)
        st = walk_states(pa.build_call_tree(key, eval_ctx=ctx, max_depth=5))
        assert st["on"] == "not-taken"   # discrete default None -> falsy
        assert st["off"] == "taken"

    def test_prune_drops_not_taken(self, tmp_path):
        pa = make_pkg(tmp_path, PROP_SRC)
        key = pa.find_function("caller")
        ctx = cta.make_root_context(pa.all_functions[key], {"flag": True}, prune=True)
        tree = pa.build_call_tree(key, eval_ctx=ctx, max_depth=6)
        names = walk_states(tree)
        assert "unreached_call" not in names      # pruned (provably not taken)
        assert "ov_call" not in names             # pruned
        assert names.get("taken_call") == "taken"  # kept


# --------------------------------------------------------------------------- #
# D + E. golden path resolution + honesty on the real Graph.svgd
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def phasic_analyzer():
    src = os.path.join(ROOT, "src", "phasic")
    if not os.path.isdir(src):
        pytest.skip("src/phasic not present")
    pa = cta.PackageAnalyzer(src)
    pa.analyze_package()
    return pa


def _svgd_state_for(pa, name, params):
    key = pa.find_method("Graph", "svgd")
    ctx = cta.make_root_context(pa.all_functions[key], params, prune=False)
    tree = pa.build_call_tree(key, eval_ctx=ctx, max_depth=2)
    nodes = find_nodes(tree, name)
    assert nodes, f"{name} not found in svgd tree"
    return {n.path_state for n in nodes}


class TestGraphSvgdGolden:
    def test_multivariate_not_taken_when_rewards_none(self, phasic_analyzer):
        # `elif rewards is not None: ... multivariate` cannot fire when rewards=None.
        states = _svgd_state_for(phasic_analyzer,
                                 "pmf_and_moments_from_graph_multivariate",
                                 {"rewards": None})
        assert states == {"not-taken"}

    def test_multivariate_conditional_when_rewards_given(self, phasic_analyzer):
        # With rewards present the inner `rewards.ndim == 2` is a runtime value,
        # so the multivariate builder becomes conditional (not provably anything).
        states = _svgd_state_for(phasic_analyzer,
                                 "pmf_and_moments_from_graph_multivariate",
                                 {"rewards": [[1.0, 2.0]]})
        assert "not-taken" not in states
        assert "conditional" in states

    def test_dispatch_visibly_flips_with_rewards(self, phasic_analyzer):
        # The headline property: changing a parameter changes the resolved subtree.
        none_states = _svgd_state_for(phasic_analyzer,
                                      "pmf_and_moments_from_graph_multivariate",
                                      {"rewards": None})
        given_states = _svgd_state_for(phasic_analyzer,
                                       "pmf_and_moments_from_graph_multivariate",
                                       {"rewards": [[1.0, 2.0]]})
        assert none_states != given_states

    def test_attribute_gated_path_stays_unknown(self, phasic_analyzer):
        # joint-index dispatch is gated by `self._joint_prob_base_graph_indexer`,
        # an attribute we cannot resolve statically -> must stay conditional, never
        # silently taken/not-taken.
        states = _svgd_state_for(phasic_analyzer,
                                 "pmf_from_graph_joint_index",
                                 {"rewards": None, "joint_index": True})
        assert states == {"conditional"}


class TestHonestyAudit:
    def test_attribute_guard_unknown(self):
        assert G("self._weight_mode == 'callback'", {}) is None

    def test_local_flag_unknown_without_folding(self):
        # Bare evaluation of a guard that tests a local flag is UNKNOWN until the
        # assignment is folded into the env (we must not guess).
        assert G("_callback_overridden", {}) is None


# --------------------------------------------------------------------------- #
# F. misattribution fix
# --------------------------------------------------------------------------- #
MISATTR_SRC = '''
class A:
    def m(self):
        self.compute()
    def compute(self):
        pass

class B:
    def compute(self):
        pass

def uses_instance(obj):
    obj.compute()
'''


class TestMisattribution:
    def test_self_call_resolves_to_own_class(self, tmp_path):
        pa = make_pkg(tmp_path, MISATTR_SRC)
        key = pa.find_method("A", "m")
        tree = pa.build_call_tree(key, show_conditions=True, max_depth=2)
        comp = find_nodes(tree, "compute")
        assert len(comp) == 1
        assert comp[0].signature.class_name == "A"   # not B
        assert comp[0].ambiguous is False

    def test_ambiguous_instance_call_is_marked(self, tmp_path):
        pa = make_pkg(tmp_path, MISATTR_SRC)
        key = pa.find_function("uses_instance")
        tree = pa.build_call_tree(key, show_conditions=True, max_depth=2)
        comp = find_nodes(tree, "compute")
        assert len(comp) == 1
        assert comp[0].ambiguous is True
        assert "A.compute" in (comp[0].reason or "")
        assert "B.compute" in (comp[0].reason or "")

    def test_legacy_default_does_not_resolve_self_calls(self, tmp_path):
        # Backward-compat: the bare (no-flag) resolver leaves self.method() calls
        # unresolved, exactly as before.
        pa = make_pkg(tmp_path, MISATTR_SRC)
        key = pa.find_method("A", "m")
        tree = pa.build_call_tree(key)            # no flags -> legacy resolver
        assert find_nodes(tree, "compute") == []


# --------------------------------------------------------------------------- #
# G. backward compatibility + param parsing
# --------------------------------------------------------------------------- #
class TestParseParams:
    def test_literals(self):
        d = cta.parse_params("a=1,b=2.5,c=True,d=None,e='x'")
        assert d == {"a": 1, "b": 2.5, "c": True, "d": None, "e": "x"}

    def test_brackets_and_quotes_not_split(self):
        d = cta.parse_params("xs=[1, 2, 3],s='a, b'")
        assert d == {"xs": [1, 2, 3], "s": "a, b"}

    def test_non_literal_falls_back_to_string(self):
        # e.g. a weight_formula expression
        d = cta.parse_params("weight_formula=exp(c0*t0)")
        assert d == {"weight_formula": "exp(c0*t0)"}

    def test_bad_entry_raises(self):
        with pytest.raises(ValueError):
            cta.parse_params("not_a_pair")


# --------------------------------------------------------------------------- #
# H. mermaid edge-condition labels (Stage 3, call_graph_to_mermaid.py)
# --------------------------------------------------------------------------- #
def _flow(children_meta):
    """Build a minimal call-tree data dict: root -> one child per (extra-keys)."""
    children = []
    for meta in children_meta:
        node = {"label": "b() -- f.py:2", "file": "f.py", "line": 2, "children": []}
        node.update(meta)
        children.append(node)
    return {"call_trees": [{"label": "a() -- f.py:1", "file": "f.py",
                            "line": 1, "children": children}]}


class TestMermaidConditions:
    def test_default_flow_has_no_edge_labels(self):
        # No condition/path_state keys -> byte-compatible plain edges, no labels.
        out = mermaid.generate_flowchart(_flow([{}]), max_depth=5, by_class=False)
        assert "a1 --> " in out.replace("n", "a")  # a plain --> edge exists
        assert '|"' not in out                      # no pipe-labels

    def test_condition_becomes_edge_label(self):
        out = mermaid.generate_flowchart(_flow([{"condition": "if x > 0"}]), 5, False)
        assert '-->|"if x > 0"|' in out

    def test_not_taken_edge_is_dotted(self):
        out = mermaid.generate_flowchart(
            _flow([{"condition": "if x", "path_state": "not-taken"}]), 5, False)
        assert '-.->|"✗ if x"|' in out

    def test_conditional_edge_marked(self):
        out = mermaid.generate_flowchart(
            _flow([{"condition": "if x", "path_state": "conditional"}]), 5, False)
        assert '-->|"? if x"|' in out

    def test_same_callee_two_conditions_two_edges(self):
        # The same callee reached under two conditions must yield two labeled edges.
        out = mermaid.generate_flowchart(
            _flow([{"condition": "if x"}, {"condition": "if y"}]), 5, False)
        assert '-->|"if x"|' in out
        assert '-->|"if y"|' in out

    def test_label_sanitized_and_truncated(self):
        long = "if " + "a" * 100
        out = mermaid.generate_flowchart(_flow([{"condition": long}]), 5, False)
        assert '…' in out          # truncated
        assert '"' not in long or out.count('"') % 2 == 0  # balanced quotes


class TestBackwardCompat:
    def test_default_tree_unchanged_vs_show_conditions(self, tmp_path):
        # Default mode collapses duplicate callees and omits self-calls; the
        # annotated mode keeps distinct guarded sites and resolves self-calls.
        src = (
            "class A:\n"
            "    def m(self, x):\n"
            "        helper()\n"
            "        if x:\n"
            "            helper()\n"      # same callee under a guard
            "        self.other()\n"
            "    def other(self):\n"
            "        pass\n"
            "def helper():\n"
            "    pass\n"
        )
        pa = make_pkg(tmp_path, src)
        key = pa.find_method("A", "m")

        default = pa.build_call_tree(key, max_depth=2)
        default_names = sorted(c.signature.name for c in default.children)
        # legacy: duplicate `helper` collapses to one, self.other() unresolved
        assert default_names == ["helper"]

        annotated = pa.build_call_tree(key, show_conditions=True, max_depth=2)
        annotated_names = sorted(c.signature.name for c in annotated.children)
        # ranked + per-guard: helper appears twice (guarded vs not) + other resolves
        assert annotated_names == ["helper", "helper", "other"]
