"""Compile a per-edge weight *formula string* into a flat bytecode tape.

This is the Python front-end for the ``weight_mode='formula'`` feature: the
user writes an expression in ``t0,t1,…`` (the parameter vector ``theta``) and
``c0,c1,…`` (the *coefficients of the edge being evaluated*); we parse it once
into a small stack-machine "tape" of integer opcodes + a float constant pool.
The same tape is shipped (as plain JSON data) to the C/C++ side and executed
**per edge in C** with no Python in the hot loop — see ``weight-formula-plan.md``.

The tape is theta-independent structure; it is evaluated for every theta and
every edge, so it is compiled exactly once (at ``graph.weight_formula = …``).

Grammar
-------
    expr        := comparison
    comparison  := addsub ( ( '==' | '!=' | '<' | '>' | '<=' | '>=' ) addsub )*
    addsub      := muldiv ( ( '+' | '-' ) muldiv )*
    muldiv      := unary  ( ( '*' | '/' ) unary  )*
    unary       := '-' unary | power
    power       := atom ( '**' unary )?            # right-assoc; exponent may be unary
    atom        := NUMBER
                 | 't' DIGITS                       # theta[i]
                 | 'c' DIGITS                       # coeff[j]
                 | NAME '(' arglist ')'             # function call
                 | '(' expr ')'

Functions: ``exp log sqrt logistic`` (1 arg), ``pow`` (2), ``delta`` (2, ==),
``and or`` (2), ``not`` (1), ``select`` (3). ``and/or/not`` treat nonzero as
true and yield 1.0/0.0; comparisons yield 1.0/0.0; ``select(cond,a,b)`` is a
branchless ternary ``(cond!=0)?a:b``.

theta-independence guard (SVGD soundness)
-----------------------------------------
SVGD gradients are finite differences, so a comparison/indicator that depends
on ``theta`` would be a non-differentiable step on the gradient path. We
therefore *require* that the operands of every comparison, of ``delta``, of
``and/or/not``, and the **condition** of ``select`` are theta-independent
(they may reference ``c<j>`` and constants but not ``t<i>``). ``select``'s two
*value* branches may use theta. Violations raise ``WeightFormulaError`` at
compile time (i.e. at assignment), pointing to ``logistic(...)`` for smooth
theta-gating.
"""
from __future__ import annotations

import re
from typing import Dict, List, Tuple

__all__ = ["WeightFormulaError", "compile_formula", "eval_tape", "OPCODES"]


class WeightFormulaError(ValueError):
    """Raised when a weight formula is syntactically or semantically invalid.

    Subclasses ``ValueError`` so callers/tests catching ``ValueError`` also
    catch it.
    """


# --- opcodes (must stay in lock-step with the C executor) ------------------
OPCODES: Dict[str, int] = {
    "PUSH_THETA": 0,   # operand: theta index   -> push theta[i]
    "PUSH_COEFF": 1,   # operand: coeff index   -> push coeff[j]
    "PUSH_CONST": 2,   # operand: const-pool ix -> push consts[k]
    "ADD": 3, "SUB": 4, "MUL": 5, "DIV": 6, "POW": 7,
    "NEG": 8,          # unary
    "EXP": 9, "LOG": 10, "SQRT": 11, "LOGISTIC": 12,
    "EQ": 13, "NE": 14, "LT": 15, "GT": 16, "LE": 17, "GE": 18,
    "AND": 19, "OR": 20, "NOT": 21,   # NOT is unary
    "SELECT": 22,      # ternary: cond, a, b -> (cond!=0)?a:b
}
_PUSH_OPS = {OPCODES["PUSH_THETA"], OPCODES["PUSH_COEFF"], OPCODES["PUSH_CONST"]}

# Function name -> (arity, AST builder)
_COMPARISONS = {"eq", "ne", "lt", "gt", "le", "ge"}
_FUNC_ARITY = {
    "exp": 1, "log": 1, "sqrt": 1, "logistic": 1, "not": 1,
    "pow": 2, "delta": 2, "and": 2, "or": 2,
    "select": 3,
}
# binary infix operator token -> AST op name
_CMP_OPS = {"==": "eq", "!=": "ne", "<": "lt", ">": "gt", "<=": "le", ">=": "ge"}

# AST node -> opcode name for the simple unary/binary cases
_UNOP_OPCODE = {"neg": "NEG", "exp": "EXP", "log": "LOG", "sqrt": "SQRT",
                "logistic": "LOGISTIC", "not": "NOT"}
_BINOP_OPCODE = {"add": "ADD", "sub": "SUB", "mul": "MUL", "div": "DIV",
                 "pow": "POW", "eq": "EQ", "ne": "NE", "lt": "LT", "gt": "GT",
                 "le": "LE", "ge": "GE", "and": "AND", "or": "OR"}
_BINOP_DISPLAY = {"add": "+", "sub": "-", "mul": "*", "div": "/", "pow": "**",
                  "eq": "==", "ne": "!=", "lt": "<", "gt": ">", "le": "<=",
                  "ge": ">=", "and": "and", "or": "or"}


# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------
_TOKEN_RE = re.compile(r"""
      (?P<NUMBER>(?:\d+\.\d*|\.\d+|\d+)(?:[eE][+-]?\d+)?)
    | (?P<NAME>[A-Za-z_][A-Za-z0-9_]*)
    | (?P<OP>\*\*|==|!=|<=|>=|[+\-*/<>(),])
    | (?P<WS>\s+)
""", re.VERBOSE)


def _tokenize(src: str) -> List[Tuple[str, str]]:
    toks: List[Tuple[str, str]] = []
    pos = 0
    n = len(src)
    while pos < n:
        m = _TOKEN_RE.match(src, pos)
        if not m or m.end() == pos:
            raise WeightFormulaError(
                f"weight_formula: invalid character {src[pos]!r} at position "
                f"{pos} in {src!r}")
        pos = m.end()
        kind = m.lastgroup
        if kind == "WS":
            continue
        toks.append((kind, m.group()))
    toks.append(("EOF", ""))
    return toks


# ---------------------------------------------------------------------------
# Parser  (recursive descent / precedence climbing) -> AST of tuples
#   ('const', float) | ('theta', int) | ('coeff', int)
#   ('unop', name, child) | ('binop', name, left, right) | ('select', c, a, b)
# ---------------------------------------------------------------------------
class _Parser:
    def __init__(self, src: str):
        self.src = src
        self.toks = _tokenize(src)
        self.i = 0

    def _peek(self) -> Tuple[str, str]:
        return self.toks[self.i]

    def _next(self) -> Tuple[str, str]:
        t = self.toks[self.i]
        self.i += 1
        return t

    def _expect_op(self, op: str) -> None:
        kind, val = self._next()
        if not (kind == "OP" and val == op):
            found = val if val else "end of input"
            raise WeightFormulaError(
                f"weight_formula: expected {op!r} but found {found!r} in "
                f"{self.src!r}")

    def parse(self):
        node = self._comparison()
        kind, val = self._peek()
        if kind != "EOF":
            raise WeightFormulaError(
                f"weight_formula: unexpected trailing {val!r} in {self.src!r}")
        return node

    def _comparison(self):
        left = self._addsub()
        while True:
            kind, val = self._peek()
            if kind == "OP" and val in _CMP_OPS:
                self._next()
                right = self._addsub()
                left = ("binop", _CMP_OPS[val], left, right)
            else:
                return left

    def _addsub(self):
        left = self._muldiv()
        while True:
            kind, val = self._peek()
            if kind == "OP" and val in ("+", "-"):
                self._next()
                right = self._muldiv()
                left = ("binop", "add" if val == "+" else "sub", left, right)
            else:
                return left

    def _muldiv(self):
        left = self._unary()
        while True:
            kind, val = self._peek()
            if kind == "OP" and val in ("*", "/"):
                self._next()
                right = self._unary()
                left = ("binop", "mul" if val == "*" else "div", left, right)
            else:
                return left

    def _unary(self):
        kind, val = self._peek()
        if kind == "OP" and val == "-":
            self._next()
            return ("unop", "neg", self._unary())
        return self._power()

    def _power(self):
        base = self._atom()
        kind, val = self._peek()
        if kind == "OP" and val == "**":
            self._next()
            exponent = self._unary()      # right-assoc; exponent may be unary
            return ("binop", "pow", base, exponent)
        return base

    def _atom(self):
        kind, val = self._next()
        if kind == "NUMBER":
            return ("const", float(val))
        if kind == "OP" and val == "(":
            node = self._comparison()
            self._expect_op(")")
            return node
        if kind == "NAME":
            nk, nv = self._peek()
            if nk == "OP" and nv == "(":         # function call
                return self._call(val)
            # variable reference
            m = re.fullmatch(r"t(\d+)", val)
            if m:
                return ("theta", int(m.group(1)))
            m = re.fullmatch(r"c(\d+)", val)
            if m:
                return ("coeff", int(m.group(1)))
            if val in _FUNC_ARITY:
                raise WeightFormulaError(
                    f"weight_formula: function {val!r} used without arguments "
                    f"in {self.src!r}")
            raise WeightFormulaError(
                f"weight_formula: unknown identifier {val!r} in {self.src!r}. "
                f"Use t<i> for theta, c<j> for coefficients, or a known "
                f"function (exp, log, sqrt, logistic, pow, delta, select, "
                f"and, or, not).")
        raise WeightFormulaError(
            f"weight_formula: unexpected {val or 'end of input'!r} in "
            f"{self.src!r}")

    def _call(self, name: str):
        if name not in _FUNC_ARITY:
            raise WeightFormulaError(
                f"weight_formula: unknown function {name!r} in {self.src!r}")
        self._expect_op("(")
        args = []
        if not (self._peek() == ("OP", ")")):
            args.append(self._comparison())
            while self._peek() == ("OP", ","):
                self._next()
                args.append(self._comparison())
        self._expect_op(")")
        arity = _FUNC_ARITY[name]
        if len(args) != arity:
            raise WeightFormulaError(
                f"weight_formula: {name}() takes {arity} argument(s) but got "
                f"{len(args)} in {self.src!r}")
        if name in ("exp", "log", "sqrt", "logistic", "not"):
            return ("unop", name, args[0])
        if name == "pow":
            return ("binop", "pow", args[0], args[1])
        if name == "delta":
            return ("binop", "eq", args[0], args[1])   # delta is sugar for ==
        if name in ("and", "or"):
            return ("binop", name, args[0], args[1])
        if name == "select":
            return ("select", args[0], args[1], args[2])
        raise AssertionError(name)  # unreachable


# ---------------------------------------------------------------------------
# Static analysis: theta-dependence + the soundness guard + unparse
# ---------------------------------------------------------------------------
def _uses_theta(node) -> bool:
    tag = node[0]
    if tag == "theta":
        return True
    if tag in ("const", "coeff"):
        return False
    if tag == "unop":
        return _uses_theta(node[2])
    if tag == "binop":
        return _uses_theta(node[2]) or _uses_theta(node[3])
    if tag == "select":
        return _uses_theta(node[1]) or _uses_theta(node[2]) or _uses_theta(node[3])
    raise AssertionError(node)


def _unparse(node) -> str:
    tag = node[0]
    if tag == "const":
        return repr(node[1])
    if tag == "theta":
        return f"t{node[1]}"
    if tag == "coeff":
        return f"c{node[1]}"
    if tag == "unop":
        op = node[1]
        if op == "neg":
            return f"-({_unparse(node[2])})"
        return f"{op}({_unparse(node[2])})"
    if tag == "binop":
        op = node[1]
        if op in ("and", "or"):
            return f"{op}({_unparse(node[2])}, {_unparse(node[3])})"
        return f"({_unparse(node[2])} {_BINOP_DISPLAY[op]} {_unparse(node[3])})"
    if tag == "select":
        return (f"select({_unparse(node[1])}, {_unparse(node[2])}, "
                f"{_unparse(node[3])})")
    raise AssertionError(node)


def _check_theta_independent(node, where: str, src: str) -> None:
    """Raise if a condition subexpression references theta."""
    if _uses_theta(node):
        raise WeightFormulaError(
            f"weight_formula: {where} must be theta-independent (it may use "
            f"c<j> and constants but not t<i>), got {_unparse(node)!r} in "
            f"{src!r}.\nComparisons/indicators are non-differentiable, and "
            f"SVGD gradients are finite differences, so a theta-dependent "
            f"condition would sit on the gradient path. For smooth "
            f"theta-gating use logistic(...) instead.")


def _enforce_guard(node, src: str) -> None:
    """Walk the AST enforcing the theta-independence of all conditions."""
    tag = node[0]
    if tag in ("const", "theta", "coeff"):
        return
    if tag == "unop":
        if node[1] == "not":
            _check_theta_independent(node[2], "the operand of not()", src)
        _enforce_guard(node[2], src)
        return
    if tag == "binop":
        op = node[1]
        if op in _COMPARISONS:
            disp = _BINOP_DISPLAY[op]
            _check_theta_independent(node[2], f"the left operand of {disp!r}", src)
            _check_theta_independent(node[3], f"the right operand of {disp!r}", src)
        elif op in ("and", "or"):
            _check_theta_independent(node[2], f"an operand of {op}()", src)
            _check_theta_independent(node[3], f"an operand of {op}()", src)
        _enforce_guard(node[2], src)
        _enforce_guard(node[3], src)
        return
    if tag == "select":
        _check_theta_independent(node[1], "the condition of select()", src)
        _enforce_guard(node[1], src)
        _enforce_guard(node[2], src)
        _enforce_guard(node[3], src)
        return
    raise AssertionError(node)


# ---------------------------------------------------------------------------
# Compile AST -> flat tape
# ---------------------------------------------------------------------------
class _Compiler:
    def __init__(self):
        self.ops: List[int] = []
        self.consts: List[float] = []
        self._const_ix: Dict[float, int] = {}
        self.n_theta = 0
        self.n_coeff = 0

    def _const_index(self, v: float) -> int:
        # dedup identical constants (keeps the pool small; order-stable)
        if v not in self._const_ix:
            self._const_ix[v] = len(self.consts)
            self.consts.append(v)
        return self._const_ix[v]

    def emit(self, node) -> None:
        tag = node[0]
        if tag == "const":
            self.ops += [OPCODES["PUSH_CONST"], self._const_index(node[1])]
        elif tag == "theta":
            self.n_theta = max(self.n_theta, node[1] + 1)
            self.ops += [OPCODES["PUSH_THETA"], node[1]]
        elif tag == "coeff":
            self.n_coeff = max(self.n_coeff, node[1] + 1)
            self.ops += [OPCODES["PUSH_COEFF"], node[1]]
        elif tag == "unop":
            self.emit(node[2])
            self.ops.append(OPCODES[_UNOP_OPCODE[node[1]]])
        elif tag == "binop":
            self.emit(node[2])
            self.emit(node[3])
            self.ops.append(OPCODES[_BINOP_OPCODE[node[1]]])
        elif tag == "select":
            self.emit(node[1])
            self.emit(node[2])
            self.emit(node[3])
            self.ops.append(OPCODES["SELECT"])
        else:
            raise AssertionError(node)


def _stack_depth(node) -> int:
    """Max stack occupancy while evaluating ``node`` (tight bound)."""
    tag = node[0]
    if tag in ("const", "theta", "coeff"):
        return 1
    if tag == "unop":
        return _stack_depth(node[2])               # pop1/push1: unchanged peak
    if tag == "binop":
        return max(_stack_depth(node[2]), 1 + _stack_depth(node[3]))
    if tag == "select":
        return max(_stack_depth(node[1]),
                   1 + _stack_depth(node[2]),
                   2 + _stack_depth(node[3]))
    raise AssertionError(node)


def compile_formula(src: str) -> Dict[str, object]:
    """Parse + validate ``src`` and return its tape dict.

    Returns ``{'ops', 'consts', 'n_theta', 'n_coeff', 'stack_depth', 'src'}``.
    Raises ``WeightFormulaError`` on any syntax/semantic error, including a
    theta-dependent comparison/indicator condition.
    """
    if not isinstance(src, str):
        raise WeightFormulaError(
            f"weight_formula must be a string, got {type(src).__name__}")
    if not src.strip():
        raise WeightFormulaError("weight_formula must be a non-empty string")
    ast = _Parser(src).parse()
    _enforce_guard(ast, src)
    comp = _Compiler()
    comp.emit(ast)
    return {
        "ops": comp.ops,
        "consts": comp.consts,
        "n_theta": comp.n_theta,
        "n_coeff": comp.n_coeff,
        "stack_depth": _stack_depth(ast),
        "src": src,
    }


# ---------------------------------------------------------------------------
# Reference VM (Python) — the C executor must match this bit-for-bit.
# ---------------------------------------------------------------------------
def _logistic(a: float) -> float:
    # numerically stable 1/(1+exp(-a)); mirrors the C executor
    import math
    if a >= 0.0:
        z = math.exp(-a)
        return 1.0 / (1.0 + z)
    z = math.exp(a)
    return z / (1.0 + z)


def eval_tape(tape: Dict[str, object], theta, coeff) -> float:
    """Evaluate a tape for one edge. ``theta``/``coeff`` are index-able floats.

    This is the authoritative reference semantics; the C executor mirrors it.
    """
    import math
    ops = tape["ops"]
    consts = tape["consts"]
    st: List[float] = []
    i = 0
    n = len(ops)
    O = OPCODES
    while i < n:
        op = ops[i]; i += 1
        if op == O["PUSH_THETA"]:
            st.append(float(theta[ops[i]])); i += 1
        elif op == O["PUSH_COEFF"]:
            st.append(float(coeff[ops[i]])); i += 1
        elif op == O["PUSH_CONST"]:
            st.append(consts[ops[i]]); i += 1
        elif op == O["ADD"]:
            b = st.pop(); a = st.pop(); st.append(a + b)
        elif op == O["SUB"]:
            b = st.pop(); a = st.pop(); st.append(a - b)
        elif op == O["MUL"]:
            b = st.pop(); a = st.pop(); st.append(a * b)
        elif op == O["DIV"]:
            b = st.pop(); a = st.pop(); st.append(a / b)
        elif op == O["POW"]:
            b = st.pop(); a = st.pop(); st.append(a ** b)
        elif op == O["NEG"]:
            a = st.pop(); st.append(-a)
        elif op == O["EXP"]:
            a = st.pop(); st.append(math.exp(a))
        elif op == O["LOG"]:
            a = st.pop(); st.append(math.log(a))
        elif op == O["SQRT"]:
            a = st.pop(); st.append(math.sqrt(a))
        elif op == O["LOGISTIC"]:
            a = st.pop(); st.append(_logistic(a))
        elif op == O["EQ"]:
            b = st.pop(); a = st.pop(); st.append(1.0 if a == b else 0.0)
        elif op == O["NE"]:
            b = st.pop(); a = st.pop(); st.append(1.0 if a != b else 0.0)
        elif op == O["LT"]:
            b = st.pop(); a = st.pop(); st.append(1.0 if a < b else 0.0)
        elif op == O["GT"]:
            b = st.pop(); a = st.pop(); st.append(1.0 if a > b else 0.0)
        elif op == O["LE"]:
            b = st.pop(); a = st.pop(); st.append(1.0 if a <= b else 0.0)
        elif op == O["GE"]:
            b = st.pop(); a = st.pop(); st.append(1.0 if a >= b else 0.0)
        elif op == O["AND"]:
            b = st.pop(); a = st.pop()
            st.append(1.0 if (a != 0.0 and b != 0.0) else 0.0)
        elif op == O["OR"]:
            b = st.pop(); a = st.pop()
            st.append(1.0 if (a != 0.0 or b != 0.0) else 0.0)
        elif op == O["NOT"]:
            a = st.pop(); st.append(1.0 if a == 0.0 else 0.0)
        elif op == O["SELECT"]:
            b = st.pop(); a = st.pop(); c = st.pop()
            st.append(a if c != 0.0 else b)
        else:
            raise AssertionError(f"bad opcode {op}")
    if len(st) != 1:
        raise AssertionError(f"tape left {len(st)} values on the stack")
    return st[0]
