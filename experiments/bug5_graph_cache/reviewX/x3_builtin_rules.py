"""reviewX x3: apply the plan's §2.2.1 builtin / dispatcher / immutable-type
admission rules (as written) to callables commonly captured in callback
closures, to measure false-refusal exposure. Pure stdlib + numpy; no phasic."""
import sys, types, math, itertools, operator, collections, functools
import numpy as np

DISP = type(np.sum)

def resolve(mod, qual):
    m = sys.modules.get(mod)
    if m is None:
        return None
    obj = m
    for part in qual.split('.'):
        obj = getattr(obj, part, None)
        if obj is None:
            return None
    return obj

def builtin_rule(x):
    t = type(x)
    if not (t is types.BuiltinFunctionType or t is np.ufunc or t is DISP):
        return None
    self_ = getattr(x, '__self__', None)
    if not (self_ is None or type(self_) is types.ModuleType):
        return 'REFUSED(__self__)'
    mod = getattr(x, '__module__', None)
    qual = getattr(x, '__qualname__', None) if t is not np.ufunc else getattr(x, '__name__', None)
    if mod is None or qual is None:
        return f'REFUSED(missing attr mod={mod!r} qual={qual!r})'
    return 'ADMIT' if resolve(mod, qual) is x else f'REFUSED(resolve {mod}.{qual})'

def type_rule(x):
    if not issubclass(type(x), type):
        return None
    flags = getattr(x, '__flags__', 0)
    if not flags & (1 << 8):
        return 'REFUSED(mutable type)'
    mod = getattr(x, '__module__', None); qual = getattr(x, '__qualname__', None)
    return 'ADMIT' if resolve(mod, qual) is x else f'REFUSED(resolve {mod}.{qual})'

cands = {
    'np.array': np.array, 'np.zeros': np.zeros, 'np.arange': np.arange,
    'np.where': np.where, 'np.concatenate': np.concatenate, 'np.dot': np.dot,
    'np.exp': np.exp, 'np.sum': np.sum, 'np.linalg.solve': np.linalg.solve,
    'np.add.reduce': np.add.reduce, 'np.random.default_rng': np.random.default_rng,
    'np.empty': np.empty, 'np.frombuffer': np.frombuffer, 'np.asarray': np.asarray,
    'math.comb': math.comb, 'math.factorial': math.factorial, 'len': len, 'max': max,
    'sorted': sorted, 'operator.add': operator.add, 'operator.mul': operator.mul,
    'str.join': str.join, 'functools.reduce': functools.reduce,
    'itertools.product': itertools.product, 'itertools.combinations': itertools.combinations,
    'collections.defaultdict': collections.defaultdict, 'np.int64': np.int64,
    'np.float64': np.float64, 'np.int32': np.int32, 'np.ndarray': np.ndarray,
    'np.dtype': np.dtype, 'int': int, 'float': float, 'tuple': tuple, 'range': range,
    'frozenset': frozenset, 'dict': dict, 'np.dtype(int64) instance': np.dtype('int64'),
    'np.bincount': np.bincount, 'np.isin': np.isin, 'np.searchsorted': np.searchsorted,
    'np.unravel_index': np.unravel_index, 'np.ravel_multi_index': np.ravel_multi_index,
}
try:
    import scipy.special as sp, scipy.stats as st
    cands['scipy.special.comb (py)'] = sp.comb
    cands['scipy.special.gammaln'] = sp.gammaln
except Exception as e:
    print('scipy import failed', e)

for name, x in cands.items():
    r = builtin_rule(x)
    if r is None:
        r = type_rule(x)
    if r is None:
        r = ('python function (code-keyed)' if type(x) is types.FunctionType
             else f'NOT FINGERPRINTABLE (type {type(x).__module__}.{type(x).__qualname__})')
    print(f'{name:32s} {r}')
