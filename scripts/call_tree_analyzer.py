#!/usr/bin/env python3
"""
Python Call Tree Analyzer
Builds a hierarchical tree of function/method calls starting from __init__.py exports.
Tracks only calls to source code functions, filtering out built-in, standard library,
and third-party library calls.
"""

import ast
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional, Any
from dataclasses import dataclass, field, asdict
from collections import defaultdict
import argparse


# Three-valued logic for static guard evaluation (used by --params, Stage 2):
#   True  = the guard provably holds for the supplied parameters
#   False = the guard provably fails
#   None  = UNKNOWN: cannot be decided statically (an attribute, a call, a
#           runtime value, an unbound name, ...). An UNKNOWN branch is always
#           KEPT and marked -- never dropped, never assumed. See evaluate_guard().
Tri = Optional[bool]


@dataclass
class ParamSpec:
    """One formal parameter of a function, in declaration order.

    Captures what FunctionSignature.args/kwargs deliberately discard (order,
    posonly-ness, default expressions) so a call site's actual arguments can be
    bound positionally/by-keyword to a callee's parameters (Stage 2 propagation).
    """
    name: str
    kind: str  # 'posonly' | 'pos_or_kw' | 'vararg' | 'kwonly' | 'kwarg'
    default: Optional[ast.AST] = None
    has_default: bool = False


@dataclass
class GuardPredicate:
    """A single branch condition under which a call executes.

    `text` is the raw rendered test source (negation/keyword are applied at
    display time from `negated`/`kind`, see _format_guard_chain).
    """
    test: Optional[ast.AST]  # the raw test expression (None for synthetic frames)
    negated: bool            # True for the else / elif-fallthrough / or-shortcircuit form
    kind: str                # 'if'|'elif'|'else'|'for'|'while'|'try'|'except'|'ifexp'|'and'|'or'
    text: str


@dataclass
class CallSite:
    """A single call expression in the source plus the guards under which it runs.

    Replaces the old flat ``Set[str]`` of call keys: it preserves source order,
    the per-call line number, the guard chain, and the actual argument AST nodes
    needed for Stage 2 parameter propagation.
    """
    call_key: str
    guards: List['GuardPredicate'] = field(default_factory=list)
    lineno: int = 0
    arg_nodes: List[ast.AST] = field(default_factory=list)
    kw_nodes: List[Tuple[Optional[str], ast.AST]] = field(default_factory=list)
    receiver: Optional[str] = None  # e.g. 'self' for self.method(...); used by misattribution fix


@dataclass
class FunctionSignature:
    """Represents a function/method signature"""
    name: str
    args: List[str]
    kwargs: List[str]
    is_method: bool
    class_name: Optional[str]
    file_path: str
    line_number: int
    # Ordered formal parameters for Stage 2 binding (additive; args/kwargs above
    # remain the source of truth for the displayed label).
    params: List[ParamSpec] = field(default_factory=list)
    # Top-level (unguarded) simple ``name = <expr>`` assignments in source order,
    # used by Stage 2 assignment folding to resolve guards that test locals.
    assigns: List[Tuple[str, ast.AST, int]] = field(default_factory=list)

    def get_function_label(self, show_params: bool = True) -> str:
        """Get just the function/method name with parameters (no file path)"""
        if show_params:
            args_str = ", ".join(self.args)
            kwargs_str = ", ".join(f"{k}={v}" for k, v in self.kwargs)
            params = ", ".join(filter(None, [args_str, kwargs_str]))
            params_part = f"({params})"
        else:
            params_part = "()"

        if self.class_name:
            return f"{self.class_name}.{self.name}{params_part}"
        else:
            return f"{self.name}{params_part}"

    def get_file_location(self) -> str:
        """Get file path and line number"""
        return f"{self.file_path}:{self.line_number}"


@dataclass
class CallNode:
    """Represents a node in the call tree"""
    signature: FunctionSignature
    children: List['CallNode'] = field(default_factory=list)
    visited: bool = False
    # Stage 1/2 annotations (all None/default unless --show-conditions/--params).
    condition: Optional[str] = None      # guard chain for the parent->this edge
    path_state: str = "static"           # 'static'|'taken'|'conditional'|'not-taken'
    reason: Optional[str] = None         # why taken/not-taken/unknown/ambiguous
    ambiguous: bool = False              # callee identity was ambiguous (misattribution fix)

    def get_max_line_width(self, prefix: str = "", is_last: bool = True, current_depth: int = 0, max_depth: int = 100,
                           show_conditions: bool = False, show_state: bool = False) -> int:
        """Calculate maximum line width in the tree INCLUDING tree structure prefixes"""
        if current_depth >= max_depth:
            connector = "└── " if is_last else "├── "
            return len(prefix) + len(connector) + len("... [max depth reached]")

        # Function label + any state glyph / condition / reason annotations.
        # Shared with print_ascii_tree so width and print can never diverge.
        func_part = format_node_label(self, show_conditions=show_conditions, show_state=show_state)

        # Calculate this line's width with tree structure
        connector = "└── " if is_last else "├── "
        my_width = len(prefix) + len(connector) + len(func_part)

        # Check circular reference case - no children for circular refs
        if self.signature.file_path == "CIRCULAR":
            return my_width

        # Get max width from children
        max_child_width = my_width
        if self.children and current_depth + 1 < max_depth:
            extension = "    " if is_last else "│   "
            for i, child in enumerate(self.children):
                is_last_child = i == len(self.children) - 1
                child_width = child.get_max_line_width(
                    prefix + extension,
                    is_last_child,
                    current_depth + 1,
                    max_depth,
                    show_conditions,
                    show_state,
                )
                max_child_width = max(max_child_width, child_width)

        return max_child_width


def _safe_unparse(node) -> str:
    """ast.unparse that never raises (returns a placeholder on failure)."""
    if node is None:
        return ""
    try:
        return ast.unparse(node)
    except Exception:
        return "<expr>"


def _guard_token(g: GuardPredicate) -> str:
    """Render one guard frame as a display token (negation/keyword applied)."""
    if g.kind == "for":
        return f"for {g.text}"
    if g.kind == "while":
        return f"while {g.text}"
    if g.kind == "try":
        return "try"
    if g.kind == "except":
        return f"except {g.text}" if g.text else "except"
    return f"not ({g.text})" if g.negated else g.text


def _format_guard_chain(guards: List[GuardPredicate]) -> str:
    """Render a guard chain as a short human-readable condition string.

    e.g. [if joint_index] -> "if joint_index"; an else-branch -> "if not (discrete)";
    a loop-guarded call -> "for x in items" (no global 'if' prefix when a loop/try
    frame is present).
    """
    if not guards:
        return ""
    if_family = {"if", "elif", "else", "and", "or", "ifexp"}
    only_if_family = all(g.kind in if_family for g in guards)
    tokens = [t for t in (_guard_token(g) for g in guards) if t]
    chain = " and ".join(tokens)
    if only_if_family and chain:
        chain = f"if {chain}"
    return chain


# State glyphs prefixed to a node's label in --params mode.
_STATE_GLYPH = {"taken": "● ", "not-taken": "✗ ", "conditional": "? "}


def format_node_label(node: 'CallNode', show_conditions: bool = False, show_state: bool = False) -> str:
    """Build the text between the tree connector and the aligned file column.

    Single source of truth used by BOTH CallNode.get_max_line_width (to size the
    alignment column) and PackageAnalyzer.print_ascii_tree (to print), so the two
    can never drift. With no flags it returns exactly the historical label.
    """
    label = node.signature.get_function_label()
    annotating = show_conditions or show_state
    prefix = ""
    if annotating and node.ambiguous:   # ranked resolver runs in either mode
        prefix += "⚠ "
    if show_state:
        prefix += _STATE_GLYPH.get(node.path_state, "")
    suffix = ""
    if show_conditions and node.condition:
        suffix += f"  [{node.condition}]"
    if annotating and node.reason:
        suffix += f"  ({node.reason})"
    return f"{prefix}{label}{suffix}"


# ---------------------------------------------------------------------------
# Stage 2: static parameter-path resolution
#
# A guard is evaluated to three-valued logic: True / False / None(=UNKNOWN).
# UNKNOWN is returned for ANYTHING not provably decidable from the supplied
# parameters (attributes, calls, subscripts, arithmetic, unbound names, ...).
# An UNKNOWN branch is always KEPT and marked -- never dropped, never guessed.
# ---------------------------------------------------------------------------

# Backstop on total nodes built per root, guaranteeing termination even on
# pathological (densely mutually-recursive) graphs. Far above any real tree.
BUILD_NODE_BUDGET = 200000

UNKNOWN = object()  # sentinel: a value that cannot be determined statically


def _truth(value) -> Tri:
    """Truthiness of a resolved value, or None if UNKNOWN / not boolable."""
    if value is UNKNOWN:
        return None
    try:
        return bool(value)
    except Exception:
        return None


def not3(t: Tri) -> Tri:
    return None if t is None else (not t)


def and3(a: Tri, b: Tri) -> Tri:
    if a is False or b is False:
        return False
    if a is True and b is True:
        return True
    return None


def or3(a: Tri, b: Tri) -> Tri:
    if a is True or b is True:
        return True
    if a is False and b is False:
        return False
    return None


_CMP_OPS = {
    ast.Eq: lambda a, b: a == b,
    ast.NotEq: lambda a, b: a != b,
    ast.Lt: lambda a, b: a < b,
    ast.LtE: lambda a, b: a <= b,
    ast.Gt: lambda a, b: a > b,
    ast.GtE: lambda a, b: a >= b,
    ast.Is: lambda a, b: a is b,
    ast.IsNot: lambda a, b: a is not b,
    ast.In: lambda a, b: a in b,
    ast.NotIn: lambda a, b: a not in b,
}


def evaluate_value(node, env: Dict[str, Any]):
    """Resolve an expression AST to a concrete Python value, or UNKNOWN.

    Strictly whitelisted (Constant, bound Name, unary +/- on numbers, literal
    containers, and boolean-valued Compare/BoolOp/not). Everything else --
    Attribute, Call, Subscript, BinOp, comprehensions, f-strings, unbound names
    -- is UNKNOWN. Never raises.
    """
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Name):
        return env[node.id] if node.id in env else UNKNOWN
    if isinstance(node, ast.UnaryOp):
        if isinstance(node.op, ast.Not):
            t = evaluate_guard(node, env)
            return UNKNOWN if t is None else t
        v = evaluate_value(node.operand, env)
        if v is UNKNOWN:
            return UNKNOWN
        try:
            if isinstance(node.op, ast.USub):
                return -v
            if isinstance(node.op, ast.UAdd):
                return +v
        except Exception:
            return UNKNOWN
        return UNKNOWN
    if isinstance(node, (ast.Tuple, ast.List, ast.Set)):
        vals = [evaluate_value(e, env) for e in node.elts]
        if any(v is UNKNOWN for v in vals):
            return UNKNOWN
        if isinstance(node, ast.Tuple):
            return tuple(vals)
        if isinstance(node, ast.Set):
            try:
                return set(vals)
            except Exception:
                return UNKNOWN
        return list(vals)
    if isinstance(node, (ast.Compare, ast.BoolOp)):
        t = evaluate_guard(node, env)
        return UNKNOWN if t is None else t
    return UNKNOWN


def evaluate_guard(node, env: Dict[str, Any]) -> Tri:
    """Three-valued truth of a guard expression (True / False / None=UNKNOWN).

    Never raises and never guesses: any unsupported construct or unbound name
    yields None so the branch is kept and marked.
    """
    if isinstance(node, ast.BoolOp):
        results = [evaluate_guard(v, env) for v in node.values]
        if isinstance(node.op, ast.And):
            acc: Tri = True
            for r in results:
                acc = and3(acc, r)
            return acc
        acc = False
        for r in results:
            acc = or3(acc, r)
        return acc
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
        return not3(evaluate_guard(node.operand, env))
    if isinstance(node, ast.Compare):
        left = evaluate_value(node.left, env)
        if left is UNKNOWN:
            return None
        cur = left
        for op, comp in zip(node.ops, node.comparators):
            fn = _CMP_OPS.get(type(op))
            if fn is None:
                return None
            right = evaluate_value(comp, env)
            if right is UNKNOWN:
                return None
            try:
                ok = fn(cur, right)
            except Exception:
                return None  # e.g. incomparable types -> undecidable, not a guess
            if not ok:
                return False
            cur = right
        return True
    # Fallback: truthiness of a resolvable atom (Constant / bound Name / container).
    return _truth(evaluate_value(node, env))


# Loop/except/try frames are inherently maybe-taken: a provably-false test can
# still prune (loop never entered), but a true test never promotes to "taken".
_NEVER_TAKEN_KINDS = ("for", "while", "try", "except")


def classify_call_site(site: CallSite, env: Dict[str, Any]) -> Tuple[str, Optional[str]]:
    """Classify an edge from its guard chain under env.

    Returns (state, reason) where state is 'taken' | 'not-taken' | 'conditional'.
    The first provably-false guard prunes; the first undecidable guard is the
    reason a surviving branch is 'conditional'.
    """
    unknown_reason = None
    state: Tri = True
    for g in site.guards:
        if g.test is None:
            r = None  # synthetic frame (bool-op short circuit / bare try)
        else:
            r = evaluate_guard(g.test, env)
            if g.negated:
                r = not3(r)
        if g.kind in _NEVER_TAKEN_KINDS:
            r = False if r is False else None
        if r is False:
            return "not-taken", f"{_guard_token(g)} → false"
        if r is None and unknown_reason is None:
            unknown_reason = f"unknown: {_guard_token(g)}"
        state = and3(state, r)
    if state is True:
        return "taken", None
    return "conditional", unknown_reason


@dataclass
class EvalContext:
    """Stage 2 evaluation state threaded through build_call_tree."""
    env: Dict[str, Any]
    prune: bool = False

    def derive(self, callee_sig: 'FunctionSignature', site: CallSite,
               caller_env: Dict[str, Any]) -> 'EvalContext':
        """Bind the call site's actuals to the callee's parameters (best-effort).

        Only literal/known-name actuals and literal defaults are bound; everything
        else is left unbound so its guards read UNKNOWN. caller_env is the caller's
        folded local env (params + resolvable top-level assigns).
        """
        params = list(callee_sig.params)
        if params and params[0].name in ('self', 'cls'):
            params = params[1:]
        pos_params = [p for p in params if p.kind in ('posonly', 'pos_or_kw')]
        by_name = {p.name: p for p in params}
        child_env: Dict[str, Any] = {}

        # A *args / **kwargs spread at the call site can supply parameters we
        # cannot see, so we must NOT assume an unsupplied parameter took its
        # default in that case (it might be filled by the spread). Explicit
        # actuals are still authoritative (Python forbids duplicate keys).
        has_star = any(isinstance(a, ast.Starred) for a in site.arg_nodes)
        has_kw_spread = any(name is None for name, _ in site.kw_nodes)

        for i, actual in enumerate(site.arg_nodes):
            if isinstance(actual, ast.Starred):
                break  # *args splat — cannot align positions past here
            if i >= len(pos_params):
                break
            v = evaluate_value(actual, caller_env)
            if v is not UNKNOWN:
                child_env[pos_params[i].name] = v

        for kwname, vnode in site.kw_nodes:
            if kwname is None or kwname not in by_name:
                continue  # **spread or unknown keyword
            v = evaluate_value(vnode, caller_env)
            if v is not UNKNOWN:
                child_env[kwname] = v

        if not (has_star or has_kw_spread):
            for p in params:
                if p.name in child_env:
                    continue
                if p.has_default and p.default is not None:
                    v = evaluate_value(p.default, {})  # literals only
                    if v is not UNKNOWN:
                        child_env[p.name] = v

        return EvalContext(env=child_env, prune=self.prune)


def make_root_context(sig: 'FunctionSignature', params_dict: Dict[str, Any], prune: bool) -> EvalContext:
    """Bind user --params to the root callable's parameters (warn on unknowns)."""
    valid = {p.name for p in sig.params}
    env: Dict[str, Any] = {}
    for k, v in params_dict.items():
        if k in valid:
            env[k] = v
        else:
            print(f"Warning: --params key '{k}' is not a parameter of "
                  f"{sig.class_name + '.' if sig.class_name else ''}{sig.name}", file=sys.stderr)
    return EvalContext(env=env, prune=prune)


def _split_top_level(s: str) -> List[str]:
    """Split on top-level commas, ignoring those inside quotes or brackets."""
    parts, cur = [], []
    depth = 0
    quote = None
    for ch in s:
        if quote:
            cur.append(ch)
            if ch == quote:
                quote = None
            continue
        if ch in ("'", '"'):
            quote = ch
        elif ch in '([{':
            depth += 1
        elif ch in ')]}':
            depth = max(0, depth - 1)
        elif ch == ',' and depth == 0:
            parts.append(''.join(cur))
            cur = []
            continue
        cur.append(ch)
    if cur:
        parts.append(''.join(cur))
    return parts


def parse_params(s: str) -> Dict[str, Any]:
    """Parse "k=v,k2=v2,..." into {name: value}.

    Values are coerced with ast.literal_eval (ints, floats, bools, None, strings,
    lists, tuples, dicts); a non-literal token is kept as a raw string (e.g. a
    weight_formula expression). Comma-splitting is quote/bracket aware.
    """
    result: Dict[str, Any] = {}
    if not s:
        return result
    for piece in _split_top_level(s):
        piece = piece.strip()
        if not piece:
            continue
        if '=' not in piece:
            raise ValueError(f"--params entry '{piece}' is not of the form key=value")
        k, v = piece.split('=', 1)
        k, v = k.strip(), v.strip()
        try:
            result[k] = ast.literal_eval(v)
        except Exception:
            result[k] = v  # raw string fallback
    return result


class ImportResolver:
    """Resolves import statements to actual module paths"""
    
    def __init__(self, package_root: Path):
        self.package_root = package_root
        self.import_map: Dict[str, Dict[str, str]] = {}
        
    def add_imports(self, file_path: str, tree: ast.AST):
        """Extract and store imports from a file"""
        self.import_map[file_path] = {}
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    name = alias.asname if alias.asname else alias.name
                    self.import_map[file_path][name] = alias.name
                    
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    for alias in node.names:
                        name = alias.asname if alias.asname else alias.name
                        full_name = f"{node.module}.{alias.name}"
                        self.import_map[file_path][name] = full_name
    
    def resolve_call(self, file_path: str, call_name: str) -> Optional[str]:
        """Resolve a function call to its full module path"""
        if file_path in self.import_map:
            return self.import_map[file_path].get(call_name, call_name)
        return call_name


class CallGraphAnalyzer(ast.NodeVisitor):
    """AST visitor that builds a call graph"""

    # Built-in and common standard library functions to exclude
    BUILTIN_NAMES = set(dir(__builtins__))
    COMMON_STDLIB = {
        'print', 'len', 'range', 'enumerate', 'zip', 'map', 'filter',
        'sum', 'min', 'max', 'abs', 'all', 'any', 'sorted', 'reversed',
        'open', 'input', 'format', 'isinstance', 'issubclass', 'super',
        'getattr', 'setattr', 'hasattr', 'delattr', 'type', 'int', 'str',
        'float', 'bool', 'list', 'dict', 'set', 'tuple', 'frozenset',
        'append', 'extend', 'insert', 'remove', 'pop', 'clear', 'get',
        'items', 'keys', 'values', 'update', 'add', 'join', 'split',
        'strip', 'replace', 'startswith', 'endswith', 'lower', 'upper'
    }

    def __init__(self, file_path: str, package_root: Path, import_resolver: ImportResolver):
        self.file_path = file_path
        self.relative_path = os.path.relpath(file_path, package_root)
        self.package_root = package_root
        self.import_resolver = import_resolver

        # Current context
        self.current_class: Optional[str] = None
        self.current_function: Optional[str] = None

        # Storage
        self.functions: Dict[str, FunctionSignature] = {}
        # Per-function ordered list of call sites (was a flat Set[str]); carries
        # guard chain, line, args and receiver for condition/path analysis.
        self.calls: Dict[str, List[CallSite]] = defaultdict(list)
        self.exports: Set[str] = set()

        # Stack of branch guards under which the currently-visited call runs.
        self.guard_stack: List[GuardPredicate] = []

    def get_function_key(self, name: str, class_name: Optional[str] = None) -> str:
        """Generate unique key for a function"""
        if class_name:
            return f"{self.relative_path}::{class_name}:{name}"
        return f"{self.relative_path}::{name}"
    
    def extract_arguments(self, node: ast.FunctionDef) -> Tuple[List[str], List[str]]:
        """Extract function arguments and keyword arguments"""
        args = []
        kwargs = []
        
        # Regular arguments
        for arg in node.args.args:
            # Skip 'self' and 'cls' for methods
            if self.current_class and arg.arg in ('self', 'cls'):
                continue
            args.append(arg.arg)
        
        # Keyword-only arguments
        for arg in node.args.kwonlyargs:
            default_idx = len(node.args.kwonlyargs) - len(node.args.kw_defaults)
            arg_idx = node.args.kwonlyargs.index(arg)
            if arg_idx >= default_idx:
                default = node.args.kw_defaults[arg_idx - default_idx]
                if default:
                    kwargs.append((arg.arg, ast.unparse(default)))
                else:
                    kwargs.append((arg.arg, 'None'))
            else:
                args.append(arg.arg)
        
        # *args
        if node.args.vararg:
            args.append(f"*{node.args.vararg.arg}")
        
        # **kwargs
        if node.args.kwarg:
            args.append(f"**{node.args.kwarg.arg}")

        return args, kwargs

    def extract_param_specs(self, node: ast.FunctionDef) -> List[ParamSpec]:
        """Ordered ParamSpec list (incl. self/cls, posonly, defaults) for binding.

        Unlike extract_arguments (which shapes the human-readable label and drops
        self/cls + posonly + defaults), this keeps the true parameter order and
        default expressions so a call's actuals can be bound to a callee. Defaults
        in Python apply to the trailing ``len(defaults)`` of posonlyargs+args.
        """
        a = node.args
        specs: List[ParamSpec] = []
        posonly = list(getattr(a, 'posonlyargs', []))
        positional = posonly + list(a.args)
        n_required = len(positional) - len(a.defaults)
        for i, arg in enumerate(positional):
            kind = 'posonly' if i < len(posonly) else 'pos_or_kw'
            if i >= n_required:
                specs.append(ParamSpec(arg.arg, kind, a.defaults[i - n_required], True))
            else:
                specs.append(ParamSpec(arg.arg, kind, None, False))
        if a.vararg:
            specs.append(ParamSpec(a.vararg.arg, 'vararg', None, False))
        for j, arg in enumerate(a.kwonlyargs):
            default = a.kw_defaults[j]  # None ⇒ required keyword-only
            specs.append(ParamSpec(arg.arg, 'kwonly', default, default is not None))
        if a.kwarg:
            specs.append(ParamSpec(a.kwarg.arg, 'kwarg', None, False))
        return specs

    @staticmethod
    def _extract_top_level_assigns(node: ast.FunctionDef) -> List[Tuple[str, ast.AST, int]]:
        """Top-level (unguarded) single-target ``name = <expr>`` assignments, in order.

        Only direct children of the function body are collected (NOT assignments
        nested inside if/for/try) so Stage 2 folding stays conservative: a value
        assigned on only one branch is never treated as known.
        """
        out: List[Tuple[str, ast.AST, int]] = []
        for stmt in node.body:
            if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1 \
                    and isinstance(stmt.targets[0], ast.Name):
                out.append((stmt.targets[0].id, stmt.value, stmt.lineno))
            elif isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name) \
                    and stmt.value is not None:
                out.append((stmt.target.id, stmt.value, stmt.lineno))
        return out

    def visit_ClassDef(self, node: ast.ClassDef):
        """Visit class definition"""
        old_class = self.current_class
        self.current_class = node.name
        self.generic_visit(node)
        self.current_class = old_class
    
    def visit_FunctionDef(self, node: ast.FunctionDef):
        """Visit function definition"""
        args, kwargs = self.extract_arguments(node)

        signature = FunctionSignature(
            name=node.name,
            args=args,
            kwargs=kwargs,
            is_method=self.current_class is not None,
            class_name=self.current_class,
            file_path=self.relative_path,
            line_number=node.lineno,
            params=self.extract_param_specs(node),
            assigns=self._extract_top_level_assigns(node),
        )

        key = self.get_function_key(node.name, self.current_class)
        self.functions[key] = signature

        # Track function context. The guard stack is reset for each function body
        # (outer guards do not apply to a nested def's calls) and restored after.
        old_function = self.current_function
        old_guard_stack = self.guard_stack
        self.current_function = key
        self.guard_stack = []
        self.generic_visit(node)
        self.current_function = old_function
        self.guard_stack = old_guard_stack

    visit_AsyncFunctionDef = visit_FunctionDef

    # --- Branch-guard tracking -------------------------------------------------
    # Each visitor pushes GuardPredicate frames onto self.guard_stack around the
    # statements they guard, then pops them. visit_Call snapshots the live stack.
    # Child statement lists are visited explicitly (not generic_visit) so the
    # frames are in scope; the *test* expression of each construct is visited
    # WITHOUT the frame (it always evaluates before the branch is chosen).

    def visit_If(self, node: ast.If):
        self.visit(node.test)  # the test itself always runs
        test_text = _safe_unparse(node.test)
        self.guard_stack.append(GuardPredicate(node.test, False, 'if', test_text))
        for stmt in node.body:
            self.visit(stmt)
        self.guard_stack.pop()
        if node.orelse:
            is_elif = len(node.orelse) == 1 and isinstance(node.orelse[0], ast.If)
            self.guard_stack.append(
                GuardPredicate(node.test, True, 'elif' if is_elif else 'else', test_text))
            for stmt in node.orelse:
                self.visit(stmt)
            self.guard_stack.pop()

    def visit_For(self, node):
        self.visit(node.iter)
        text = f"{_safe_unparse(node.target)} in {_safe_unparse(node.iter)}"
        self.guard_stack.append(GuardPredicate(node.iter, False, 'for', text))
        for stmt in node.body:
            self.visit(stmt)
        self.guard_stack.pop()
        for stmt in node.orelse:  # for...else runs after normal completion
            self.visit(stmt)

    visit_AsyncFor = visit_For

    def visit_While(self, node: ast.While):
        text = _safe_unparse(node.test)
        self.guard_stack.append(GuardPredicate(node.test, False, 'while', text))
        self.visit(node.test)
        for stmt in node.body:
            self.visit(stmt)
        self.guard_stack.pop()
        for stmt in node.orelse:
            self.visit(stmt)

    def visit_Try(self, node):
        # The try BODY runs unconditionally (until an exception is raised), so it
        # gets NO guard frame. Only the except handlers are conditional (they run
        # only on error). orelse/finalbody also run unconditionally.
        for stmt in node.body:
            self.visit(stmt)
        for handler in node.handlers:
            htext = _safe_unparse(handler.type) if handler.type else ''
            self.guard_stack.append(GuardPredicate(handler.type, False, 'except', htext))
            for stmt in handler.body:
                self.visit(stmt)
            self.guard_stack.pop()
        for stmt in node.orelse:      # runs only if no exception
            self.visit(stmt)
        for stmt in node.finalbody:   # always runs
            self.visit(stmt)

    visit_TryStar = visit_Try

    def visit_IfExp(self, node: ast.IfExp):
        self.visit(node.test)
        text = _safe_unparse(node.test)
        self.guard_stack.append(GuardPredicate(node.test, False, 'ifexp', text))
        self.visit(node.body)
        self.guard_stack.pop()
        self.guard_stack.append(GuardPredicate(node.test, True, 'ifexp', text))
        self.visit(node.orelse)
        self.guard_stack.pop()

    def visit_BoolOp(self, node: ast.BoolOp):
        if not node.values:
            return
        self.visit(node.values[0])  # first operand always evaluated
        is_and = isinstance(node.op, ast.And)
        kind = 'and' if is_and else 'or'
        for i in range(1, len(node.values)):
            prior = " and ".join(_safe_unparse(p) for p in node.values[:i])
            # 'and': operand runs only if prior is truthy; 'or': only if prior is falsy.
            self.guard_stack.append(GuardPredicate(None, not is_and, kind, prior))
            self.visit(node.values[i])
            self.guard_stack.pop()

    # Note: is_source_code_call method removed - filtering now done in build_call_tree

    def visit_Call(self, node: ast.Call):
        """Visit function call"""
        if not self.current_function:
            self.generic_visit(node)
            return

        call_name = None
        class_name = None

        # Direct function call
        if isinstance(node.func, ast.Name):
            call_name = node.func.id

        # Method call
        elif isinstance(node.func, ast.Attribute):
            if isinstance(node.func.value, ast.Name):
                # Could be module.function or instance.method
                call_name = node.func.attr
                class_name = node.func.value.id
            elif isinstance(node.func.value, ast.Attribute):
                # Nested attribute (e.g., self.obj.method)
                call_name = node.func.attr

        if call_name:
            # Try to resolve the full path
            if class_name:
                resolved = self.import_resolver.resolve_call(self.file_path, class_name)
                if resolved:
                    call_key = f"{resolved}:{call_name}"
                else:
                    call_key = f"{class_name}:{call_name}"
            else:
                resolved = self.import_resolver.resolve_call(self.file_path, call_name)
                call_key = resolved if resolved else call_name

            # Record the call site with the guards under which it runs, plus the
            # actuals/receiver needed for Stage 2 path analysis. `class_name` here
            # is the receiver Name (e.g. 'self') for method calls.
            self.calls[self.current_function].append(CallSite(
                call_key=call_key,
                guards=list(self.guard_stack),
                lineno=getattr(node, 'lineno', 0),
                arg_nodes=list(node.args),
                kw_nodes=[(kw.arg, kw.value) for kw in node.keywords],
                receiver=class_name,
            ))

        self.generic_visit(node)


class PackageAnalyzer:
    """Main analyzer for Python packages"""

    def __init__(self, package_path: str):
        self.package_path = Path(package_path).resolve()
        self.package_name = self.package_path.name
        self.import_resolver = ImportResolver(self.package_path)

        # Results
        self.all_functions: Dict[str, FunctionSignature] = {}
        self.call_graph: Dict[str, List[CallSite]] = defaultdict(list)
        self.exports: Set[str] = set()
        
    def analyze_file(self, file_path: Path) -> CallGraphAnalyzer:
        """Analyze a single Python file"""
        with open(file_path, 'r', encoding='utf-8') as f:
            try:
                tree = ast.parse(f.read(), filename=str(file_path))
                
                # First pass: collect imports
                self.import_resolver.add_imports(str(file_path), tree)
                
                # Second pass: analyze calls
                analyzer = CallGraphAnalyzer(
                    str(file_path), 
                    self.package_path,
                    self.import_resolver
                )
                analyzer.visit(tree)
                
                return analyzer
                
            except SyntaxError as e:
                print(f"Syntax error in {file_path}: {e}", file=sys.stderr)
                return None
    
    def analyze_package(self):
        """Analyze entire package"""
        # Find all Python files
        python_files = list(self.package_path.rglob("*.py"))
        
        # First, analyze __init__.py to find exports
        init_file = self.package_path / "__init__.py"
        if init_file.exists():
            analyzer = self.analyze_file(init_file)
            if analyzer:
                # Track exports from __init__.py
                for key in analyzer.functions:
                    self.exports.add(key)
                
                # Also track __all__ exports if present
                with open(init_file, 'r') as f:
                    tree = ast.parse(f.read())
                    for node in ast.walk(tree):
                        if isinstance(node, ast.Assign):
                            for target in node.targets:
                                if isinstance(target, ast.Name) and target.id == '__all__':
                                    if isinstance(node.value, ast.List):
                                        for elt in node.value.elts:
                                            if isinstance(elt, ast.Constant):
                                                self.exports.add(elt.value)
        
        # Analyze all files
        analyzers = []
        for file_path in python_files:
            analyzer = self.analyze_file(file_path)
            if analyzer:
                analyzers.append(analyzer)
                
                # Merge results (preserve source order; do NOT collapse here —
                # the same callee under different guards is distinct).
                self.all_functions.update(analyzer.functions)
                for func, sites in analyzer.calls.items():
                    self.call_graph[func].extend(sites)
    
    def build_call_tree(self, root_key: str, visited: Optional[Set[str]] = None,
                        show_conditions: bool = False, eval_ctx: Optional['EvalContext'] = None,
                        max_depth: Optional[int] = None, _depth: int = 0,
                        _budget: Optional[List[int]] = None) -> Optional[CallNode]:
        """Build a tree starting from a root function (only package code).

        show_conditions annotates each edge with its guard chain; eval_ctx (Stage 2)
        carries the parameter environment used to classify each branch as
        taken/not-taken/conditional and is propagated into callees. When either is
        active, call sites that differ only by guard are kept distinct; otherwise
        duplicate callees collapse to preserve the historical tree shape.

        max_depth bounds the BUILD to the displayed depth (print truncates anyway);
        this is essential with the ranked resolver, whose denser self-call graph
        combined with per-path visited.copy() would otherwise blow up. _budget is a
        global node backstop guaranteeing termination on pathological graphs.
        """
        if visited is None:
            visited = set()
        if _budget is None:
            _budget = [BUILD_NODE_BUDGET]

        if root_key in visited:
            # Circular dependency - extract function name from key
            if root_key in self.all_functions:
                sig = self.all_functions[root_key]
                return CallNode(
                    signature=FunctionSignature(
                        name=sig.name,
                        args=[],
                        kwargs=[],
                        is_method=sig.is_method,
                        class_name=sig.class_name,
                        file_path="CIRCULAR",
                        line_number=0
                    )
                )
            else:
                # Fallback for unresolved circular references
                return CallNode(
                    signature=FunctionSignature(
                        name="CIRCULAR",
                        args=[],
                        kwargs=[],
                        is_method=False,
                        class_name=None,
                        file_path="CIRCULAR",
                        line_number=0
                    )
                )

        visited.add(root_key)

        # Get function signature - only include if it's in our package
        if root_key not in self.all_functions:
            # Skip - not from our package
            return None

        node = CallNode(signature=self.all_functions[root_key])

        # Stop building beyond the displayed depth or once the global node budget
        # is spent (return this node as a leaf; print shows the truncation marker).
        _budget[0] -= 1
        if (max_depth is not None and _depth >= max_depth) or _budget[0] <= 0:
            return node

        annotate = show_conditions or eval_ctx is not None
        caller_sig = self.all_functions[root_key]
        sites = self._select_sites(self.call_graph.get(root_key, []), annotate)

        # Stage 2: fold this function's top-level assigns into the eval env so
        # guards that test locals (e.g. `_callback_overridden = callback is not None`)
        # can be resolved. Conservative: only unguarded, evaluable assigns.
        local_env = None
        if eval_ctx is not None:
            local_env = self._fold_assigns(caller_sig, eval_ctx)

        for site in sites:
            # Default (no-flag) keeps the legacy first-match for output stability;
            # the new --show-conditions/--params views use the ranked resolver,
            # which also resolves self.method()/constructors and marks ambiguity.
            if annotate:
                func_key, ambiguous, candidates = self._resolve_callee_ranked(site, caller_sig)
            else:
                func_key, ambiguous, candidates = self._resolve_callee_legacy(site)
            if func_key is None:
                continue

            # Stage 2: classify the edge from the guard chain + current env.
            state, reason = ("static", None)
            if eval_ctx is not None:
                state, reason = classify_call_site(site, local_env)
                if state == "not-taken" and eval_ctx.prune:
                    continue  # drop only definitively-false branches under --prune

            # Stage 2: propagate actuals into the callee's parameter env.
            child_ctx = eval_ctx
            if eval_ctx is not None and func_key in self.all_functions:
                child_ctx = eval_ctx.derive(self.all_functions[func_key], site, local_env)

            child = self.build_call_tree(func_key, visited.copy(), show_conditions, child_ctx,
                                         max_depth, _depth + 1, _budget)
            if child is None:
                continue
            if annotate:
                child.condition = _format_guard_chain(site.guards)
            reason_parts = []
            if eval_ctx is not None:
                child.path_state = state
                if reason:
                    reason_parts.append(reason)
            if ambiguous:
                child.ambiguous = True
                if candidates:
                    reason_parts.append("ambiguous: " + ", ".join(candidates))
            if reason_parts:
                child.reason = "; ".join(reason_parts)
            node.children.append(child)

        return node

    @staticmethod
    def _select_sites(sites: List[CallSite], annotate: bool) -> List[CallSite]:
        """De-duplicate call sites.

        Without annotation, collapse by callee key (the historical Set-like shape).
        With annotation, keep distinct (callee, guard-chain) pairs so the same
        callee under different branches is shown separately. Source order preserved.
        """
        seen = set()
        out = []
        for s in sites:
            key = (s.call_key, _format_guard_chain(s.guards)) if annotate else s.call_key
            if key in seen:
                continue
            seen.add(key)
            out.append(s)
        return out

    def _resolve_callee_legacy(self, site: CallSite) -> Tuple[Optional[str], bool, List[str]]:
        """Historical resolver: first function key matching by substring.

        Preserves the exact bare (no-flag) tree shape. self.method() calls do NOT
        match here (call_key 'self:method' is not a substring of 'path::Class:method'),
        matching prior behaviour.
        """
        for func_key in self.all_functions:
            if func_key.endswith(site.call_key) or site.call_key in func_key:
                return func_key, False, []
        return None, False, []

    @staticmethod
    def _parse_func_key(fk: str) -> Tuple[str, Optional[str], str]:
        """'path::Class:name' -> (path, 'Class', 'name'); 'path::name' -> (path, None, 'name')."""
        path, _, tail = fk.partition('::')
        if ':' in tail:
            fclass, fname = tail.split(':', 1)
            return path, fclass, fname
        return path, None, tail

    @staticmethod
    def _parse_call_key(ck: str) -> Tuple[Optional[str], str]:
        """'Recv:name' -> ('Recv', 'name'); 'name' -> (None, 'name')."""
        if ':' in ck:
            recv, name = ck.rsplit(':', 1)
            return recv, name
        return None, ck

    def _resolve_callee_ranked(self, site: CallSite, caller_sig: FunctionSignature) -> Tuple[Optional[str], bool, List[str]]:
        """Resolve a callee by exact name + class-context ranking.

        Prefers, in order: a self.method() call -> a method of the caller's own
        class; a Recv.method() call -> a method of class Recv; a bare name() ->
        a free function (or a class constructor name() -> name.__init__). When the
        top score is shared by several definitions the node is marked ambiguous and
        the tied candidates are listed (deterministic pick: sorted-first). Falls
        back to the legacy substring match only when nothing matches by name.

        Returns (func_key | None, ambiguous, candidate_labels).
        """
        recv, name = self._parse_call_key(site.call_key)
        receiver = site.receiver or recv
        caller_class = caller_sig.class_name
        caller_file = caller_sig.file_path

        scored: List[Tuple[int, str]] = []
        for fk, fsig in self.all_functions.items():
            if fsig.name != name:
                continue
            fclass = fsig.class_name
            if receiver in ('self', 'cls'):
                score = 100 if fclass == caller_class else (10 if fclass is not None else 1)
            elif receiver is not None:
                score = 100 if fclass == receiver else (5 if fclass is None else 10)
            else:  # bare name() — free function preferred
                score = 100 if fclass is None else 8
            if fsig.file_path == caller_file:
                score += 3
            scored.append((score, fk))

        # Bare name() may be a constructor: name == ClassName -> ClassName.__init__
        if receiver is None:
            for fk, fsig in self.all_functions.items():
                if fsig.class_name == name and fsig.name == '__init__':
                    score = 90 + (3 if fsig.file_path == caller_file else 0)
                    scored.append((score, fk))

        if not scored:
            return self._resolve_callee_legacy(site)

        best = max(s for s, _ in scored)
        winners = sorted(fk for s, fk in scored if s == best)
        if len(winners) == 1:
            return winners[0], False, []
        labels = [self._short_key(fk) for fk in winners[:4]]
        if len(winners) > 4:
            labels.append(f"+{len(winners) - 4} more")
        return winners[0], True, labels

    @staticmethod
    def _short_key(fk: str) -> str:
        """Compact 'file:Class.name' label for an ambiguous candidate list."""
        path, fclass, fname = PackageAnalyzer._parse_func_key(fk)
        base = os.path.basename(path)
        return f"{base}:{fclass + '.' if fclass else ''}{fname}"

    def _fold_assigns(self, sig: FunctionSignature, eval_ctx: 'EvalContext') -> Dict[str, Any]:
        """Return a copy of the eval env extended with this function's resolvable
        top-level assignments (Stage 2 assignment folding)."""
        env = dict(eval_ctx.env)
        for name, value_node, _lineno in sig.assigns:
            res = evaluate_value(value_node, env)
            if res is not UNKNOWN:
                env[name] = res
            else:
                env.pop(name, None)  # shadow any stale binding; local is unknown
        return env


    def find_class_methods(self, class_name: str) -> List[str]:
        """Find all methods for a given class"""
        methods = []
        for key, sig in self.all_functions.items():
            if sig.class_name == class_name:
                methods.append(key)
        return methods

    def find_function(self, function_name: str) -> Optional[str]:
        """Find a function by name (returns first match)"""
        for key, sig in self.all_functions.items():
            if sig.name == function_name and sig.class_name is None:
                return key
        return None

    def find_method(self, class_name: str, method_name: str) -> Optional[str]:
        """Find a specific method of a class"""
        for key, sig in self.all_functions.items():
            if sig.class_name == class_name and sig.name == method_name:
                return key
        return None

    def build_full_tree(self, show_conditions: bool = False, max_depth: Optional[int] = None) -> List[CallNode]:
        """Build complete call tree starting from exports"""
        roots = []

        # Start from exports
        if self.exports:
            for export in self.exports:
                tree = self.build_call_tree(export, show_conditions=show_conditions, max_depth=max_depth)
                if tree:
                    roots.append(tree)
        else:
            # If no exports, use all top-level functions that nothing else calls.
            called_keys = {s.call_key for sites in self.call_graph.values() for s in sites}
            for key in self.all_functions:
                if '::' not in key:
                    continue
                name = key.split('::')[1]
                if not any(name.endswith(ck) or ck in name for ck in called_keys):
                    tree = self.build_call_tree(key, show_conditions=show_conditions, max_depth=max_depth)
                    if tree:
                        roots.append(tree)

        return roots
    
    def print_ascii_tree(self, node: CallNode, prefix: str = "", is_last: bool = True, max_depth: int = 10, current_depth: int = 0, align_width: Optional[int] = None,
                         show_conditions: bool = False, show_state: bool = False):
        """Print tree in ASCII format like Linux tree command with aligned file paths"""
        # Calculate alignment width on first call (longest line + 5 spaces)
        if align_width is None:
            align_width = node.get_max_line_width(prefix="", is_last=True, max_depth=max_depth,
                                                  show_conditions=show_conditions, show_state=show_state) + 5

        if current_depth >= max_depth:
            print(f"{prefix}{'└── ' if is_last else '├── '}... [max depth reached]")
            return

        # Current node (label may carry a state glyph / condition / reason).
        connector = "└── " if is_last else "├── "
        func_label = format_node_label(node, show_conditions=show_conditions, show_state=show_state)
        file_location = node.signature.get_file_location()

        # Calculate current line width (prefix + connector + function label)
        current_line_width = len(prefix) + len(connector) + len(func_label)

        # Pad to alignment width
        padding = max(1, align_width - current_line_width)

        # Print with aligned file path
        print(f"{prefix}{connector}{func_label}{' ' * padding}{file_location}")

        # Children
        if node.children:
            extension = "    " if is_last else "│   "
            for i, child in enumerate(node.children):
                is_last_child = i == len(node.children) - 1
                self.print_ascii_tree(child, prefix + extension, is_last_child, max_depth, current_depth + 1,
                                      align_width, show_conditions, show_state)
    
    @staticmethod
    def tree_to_dict(node: CallNode) -> Dict:
        """Convert a CallNode tree to a plain dict.

        condition/path_state/reason/ambiguous are included ONLY when set, so the
        default (no-flag) JSON is unchanged and the mermaid companion (which reads
        only label/file/children) is unaffected.
        """
        func_label = node.signature.get_function_label()
        file_location = node.signature.get_file_location()
        d = {
            'label': f"{func_label} -- {file_location}",
            'file': node.signature.file_path,
            'line': node.signature.line_number,
            'children': [PackageAnalyzer.tree_to_dict(child) for child in node.children]
        }
        if node.condition:
            d['condition'] = node.condition
        if node.path_state != 'static':
            d['path_state'] = node.path_state
        if node.reason:
            d['reason'] = node.reason
        if node.ambiguous:
            d['ambiguous'] = True
        return d

    def call_trees_to_data(self, roots: List[CallNode]) -> Dict:
        """Convert a list of CallNode roots to the JSON-compatible data dict."""
        return {
            'package': str(self.package_path),
            'exports': list(self.exports),
            'call_trees': [self.tree_to_dict(root) for root in roots]
        }

    def save_to_json(self, output_path: str, show_conditions: bool = False, max_depth: Optional[int] = None):
        """Save call tree to JSON file"""
        roots = self.build_full_tree(show_conditions=show_conditions, max_depth=max_depth)
        data = self.call_trees_to_data(roots)

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)

        print(f"Call tree saved to {output_path}")
    
    def print_summary(self, show_conditions: bool = False, max_depth: Optional[int] = None):
        """Print analysis summary"""
        print(f"\n{'='*60}")
        print(f"Package Analysis Summary: {self.package_name}")
        print(f"{'='*60}")
        print(f"Total functions/methods found: {len(self.all_functions)}")
        print(f"Functions with calls: {len(self.call_graph)}")
        print(f"Exported functions: {len(self.exports)}")
        
        if self.exports:
            print(f"\nExported functions:")
            for export in sorted(self.exports):
                print(f"  - {export}")
        
        print(f"\n{'='*60}")
        print("Call Tree:")
        print(f"{'='*60}\n")
        
        roots = self.build_full_tree(show_conditions=show_conditions, max_depth=max_depth)
        for root in roots:
            self.print_ascii_tree(root, max_depth=(max_depth if max_depth is not None else 10),
                                  show_conditions=show_conditions)
            print()  # Empty line between trees


def _resolve_callable_keys(analyzer, callable_name):
    """Resolve --callable to a list of function keys (a method, every method of a
    class, or a free function), exiting with an error if nothing matches."""
    if '.' in callable_name:
        parts = callable_name.split('.', 1)
        key = analyzer.find_method(parts[0], parts[1])
        if not key:
            print(f"Error: Method '{callable_name}' not found", file=sys.stderr)
            sys.exit(1)
        return [key]
    methods = analyzer.find_class_methods(callable_name)
    if methods:
        return sorted(methods)
    key = analyzer.find_function(callable_name)
    if not key:
        print(f"Error: Callable '{callable_name}' not found (tried as function, class, and method)", file=sys.stderr)
        sys.exit(1)
    return [key]


def _root_eval_ctx(analyzer, key, params_dict, prune):
    """Build the root EvalContext for a callable key, or None when no --params."""
    if params_dict is None:
        return None
    return make_root_context(analyzer.all_functions[key], params_dict, prune)


def _build_callable_trees(analyzer, callable_name, show_conditions=False, params_dict=None, prune=False, max_depth=None):
    """Resolve --callable to a list of CallNode roots, applying path evaluation."""
    roots = []
    for key in _resolve_callable_keys(analyzer, callable_name):
        eval_ctx = _root_eval_ctx(analyzer, key, params_dict, prune)
        tree = analyzer.build_call_tree(key, show_conditions=show_conditions, eval_ctx=eval_ctx, max_depth=max_depth)
        if tree:
            roots.append(tree)
    return roots


def main():
    parser = argparse.ArgumentParser(
        description='Analyze Python package call tree (only package code, no external libraries)',
        epilog='''
Examples:
  # Analyze entire package
  %(prog)s src/phasic

  # Show call tree for a specific function
  %(prog)s src/phasic --callable record_elimination_trace

  # Show call tree for all methods of a class
  %(prog)s src/phasic --callable Graph

  # Show call tree for a specific method
  %(prog)s src/phasic --callable Graph.serialize

  # Output as mermaid sequence diagram
  %(prog)s src/phasic --callable Graph.serialize --diagram sequence

  # Output as mermaid flowchart
  %(prog)s src/phasic --callable Graph.serialize --diagram flow

  # Mermaid flowchart grouped by class, depth-limited
  %(prog)s src/phasic --callable Graph.serialize --diagram flow --by-class -d 3

  # Annotate every branch with the condition that selects it (always correct)
  %(prog)s src/phasic --callable Graph.svgd --show-conditions -d 4

  # Resolve which path concrete parameters select (● taken, ✗ not-taken, ? unknown)
  %(prog)s src/phasic --callable Graph.svgd --params "rewards=None,callback=None" -d 4

  # Same, but drop the provably not-taken branches
  %(prog)s src/phasic --callable Graph.svgd --params "rewards=None" --prune -d 4

Note: Only shows calls to functions/methods implemented in the specified package.
External library calls (numpy, jax, etc.) are excluded. --params resolves only
branches decidable from the given values; runtime-/attribute-/**kwargs-dependent
branches are kept and marked '?' (never guessed).
        ''',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('package_path', help='Path to Python package')
    parser.add_argument('-o', '--output', default='call_tree.json',
                       help='Output JSON file (default: call_tree.json)')
    parser.add_argument('-d', '--max-depth', type=int, default=10,
                       help='Maximum tree depth to display (default: 10)')
    parser.add_argument('--callable', dest='callable_name',
                       help='Show call tree for function, class, or Class.method')
    parser.add_argument('--diagram', choices=['sequence', 'flow'],
                       help="Output a mermaid diagram: 'sequence' or 'flow'")
    parser.add_argument('--by-class', action='store_true',
                       help='Group diagram participants/nodes by class instead of module')
    parser.add_argument('--quiet', action='store_true',
                       help='Suppress summary output, only show diagram or JSON')
    parser.add_argument('--show-conditions', action='store_true',
                       help='Annotate every branch with its guard condition '
                            '(static, no evaluation -- always correct)')
    parser.add_argument('--params',
                       help='Resolve the executed path for concrete parameter '
                            'values, e.g. --params "rewards=None,discrete=True,callback=None". '
                            'Decidable branches are marked taken (●) / not-taken (✗); '
                            'branches that depend on runtime values, attributes or **kwargs '
                            'stay conditional (?) -- kept and marked, never guessed.')
    parser.add_argument('--prune', action='store_true',
                       help='With --params, drop branches that are provably not '
                            'taken (default: keep and mark them with ✗).')

    args = parser.parse_args()

    try:
        params_dict = parse_params(args.params) if args.params else None
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    # --params shows the resolved state (glyphs + decisive-guard reason). The full
    # guard chain [condition] is only added when --show-conditions is also given
    # (it is verbose; the reason already names the deciding guard).
    show_state = params_dict is not None
    show_conditions = args.show_conditions

    if not os.path.exists(args.package_path):
        print(f"Error: Package path '{args.package_path}' does not exist", file=sys.stderr)
        sys.exit(1)

    # Run analysis
    analyzer = PackageAnalyzer(args.package_path)
    if not args.quiet:
        print(f"Analyzing package: {args.package_path}", file=sys.stderr)
    analyzer.analyze_package()

    # --diagram mode: generate mermaid and exit
    if args.diagram:
        from call_graph_to_mermaid import generate_diagram

        if args.callable_name:
            roots = _build_callable_trees(analyzer, args.callable_name,
                                          show_conditions=show_conditions,
                                          params_dict=params_dict, prune=args.prune,
                                          max_depth=args.max_depth)
        else:
            roots = analyzer.build_full_tree(show_conditions=show_conditions,
                                             max_depth=(args.max_depth if show_conditions else None))

        if not roots:
            print("No call tree found", file=sys.stderr)
            sys.exit(1)

        data = analyzer.call_trees_to_data(roots)
        diagram = generate_diagram(data, args.diagram, args.max_depth, args.by_class)

        if args.output and args.output != 'call_tree.json':
            with open(args.output, 'w', encoding='utf-8') as f:
                if args.output.endswith('.md'):
                    f.write('```mermaid\n')
                    f.write(diagram)
                    f.write('```\n')
                else:
                    f.write(diagram)
            print(f"Diagram written to {args.output}", file=sys.stderr)
        else:
            print(diagram)
        return

    # Handle specific callable request (ASCII tree)
    if args.callable_name:
        def _root_tree(key):
            return analyzer.build_call_tree(
                key, show_conditions=show_conditions,
                eval_ctx=_root_eval_ctx(analyzer, key, params_dict, args.prune),
                max_depth=args.max_depth)

        def _print(tree):
            analyzer.print_ascii_tree(tree, max_depth=args.max_depth,
                                      show_conditions=show_conditions, show_state=show_state)

        # Check if it's a Class.method format
        if '.' in args.callable_name:
            parts = args.callable_name.split('.', 1)
            class_name = parts[0]
            method_name = parts[1]

            # Try to find the specific method
            key = analyzer.find_method(class_name, method_name)
            if not key:
                print(f"Error: Method '{class_name}.{method_name}' not found", file=sys.stderr)
                sys.exit(1)
            tree = _root_tree(key)
            if tree:
                _print(tree)
            else:
                print("No call tree found")

        else:
            # Try as a class first (show all methods)
            methods = analyzer.find_class_methods(args.callable_name)
            if methods:
                for method_key in sorted(methods):
                    sig = analyzer.all_functions[method_key]
                    print(f"\n{sig.class_name}.{sig.name}():")
                    print("-" * 60)
                    tree = _root_tree(method_key)
                    if tree:
                        _print(tree)
                    print()
            else:
                # Try as a function
                key = analyzer.find_function(args.callable_name)
                if not key:
                    print(f"Error: Callable '{args.callable_name}' not found (tried as function, class, and method)", file=sys.stderr)
                    sys.exit(1)
                print(f"\n{'='*60}")
                print(f"Call Tree for {args.callable_name}()")
                print(f"{'='*60}\n")
                tree = _root_tree(key)
                if tree:
                    _print(tree)
                else:
                    print("No call tree found")

    else:
        if not args.quiet:
            # Print full summary
            analyzer.print_summary(show_conditions=show_conditions,
                                   max_depth=(args.max_depth if show_conditions else None))

        # Save to JSON (keep the legacy unbounded depth in the default dump; bound
        # only when the ranked resolver is engaged via --show-conditions).
        analyzer.save_to_json(args.output, show_conditions=show_conditions,
                              max_depth=(args.max_depth if show_conditions else None))


if __name__ == "__main__":
    main()

# # Basic usage
# python call_tree_analyzer.py /path/to/package

# # With custom output file
# python call_tree_analyzer.py /path/to/package -o my_tree.json

# # Limit tree depth
# python call_tree_analyzer.py /path/to/package -d 5

# # Example with a package
# python call_tree_analyzer.py ./src/phasic
# ```

# ## Example Output
# ```
# Package Analysis Summary: phasic
# ============================================================
# Total functions/methods found: 42
# Functions with calls: 18
# Exported functions: 3

# Call Tree:
# ============================================================

# └── src/phasic/__init__.py::Graph.__init__(nodes, edges)
#     ├── src/cpp/phasic_pybind.py::Graph:__init__()
#     ├── src/phasic/graph.py::validate_edges(edges)
#     │   └── src/phasic/utils.py::check_type(obj, expected_type)
#     └── src/phasic/graph.py::build_adjacency_list(nodes, edges)
#         ├── src/phasic/utils.py::create_dict()
#         └── src/phasic/graph.py::Edge:validate()