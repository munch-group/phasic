# reviewAK p3: plan §2.2.1 builtin / immutable-type admission, implemented as written, on common callables
import sys, types, math, itertools, operator, functools, numpy as np
def resolve(mod, qn):
    if type(mod) is not str or type(qn) is not str: return None
    m = sys.modules.get(mod)
    obj = m
    for part in qn.split("."):
        if obj is None: return None
        obj = getattr(obj, part, None)
    return obj
def builtin_rule(x):
    if not (type(x) is types.BuiltinFunctionType or type(x) is np.ufunc or type(x) is type(np.sum)):
        return "not-exact-type"
    s = getattr(x, "__self__", None)
    if not (s is None or type(s) is types.ModuleType): return "refused:__self__"
    mod = getattr(x, "__module__", None)
    qn = getattr(x, "__name__", None) if type(x) is np.ufunc else getattr(x, "__qualname__", None)
    if mod is None or qn is None: return f"refused:missing mod={mod!r} qn={qn!r}"
    return "ADMIT" if resolve(mod, qn) is x else f"refused:resolve {mod}.{qn}"
def type_rule(x):
    if type(x) is not type: return "not-type"
    if not (x.__flags__ & (1 << 8)): return "refused:mutable"
    return "ADMIT" if resolve(x.__module__, x.__qualname__) is x else f"refused:resolve {x.__module__}.{x.__qualname__}"
import scipy.special as sp
cases = dict(len=len, math_sqrt=math.sqrt, math_comb=math.comb, np_exp=np.exp, np_sum=np.sum, np_zeros=np.zeros,
    np_array=np.array, np_where=np.where, np_linalg_solve=np.linalg.solve, np_isclose=np.isclose,
    operator_add=operator.add, functools_reduce=functools.reduce, np_random_default_rng=np.random.default_rng,
    np_random_rand=np.random.rand, np_frompyfunc=np.frompyfunc(lambda a: a, 1, 1), sp_gamma=sp.gamma,
    dict_get={}.get, np_concatenate=np.concatenate, np_dot=np.dot, np_maximum=np.maximum,
    np_add_reduce=np.add.reduce, math_lgamma=math.lgamma, np_log=np.log)
for k, v in cases.items():
    print(f"builtin {k:24s} type={type(v).__name__:28s} -> {builtin_rule(v)}")
for k, v in dict(range=range, int=int, comb_wr=itertools.combinations_with_replacement, product=itertools.product,
                 np_float64=np.float64, np_ndarray=np.ndarray, Counter=__import__('collections').Counter,
                 partial=functools.partial, defaultdict=__import__('collections').defaultdict).items():
    print(f"type    {k:24s} -> {type_rule(v)}")
