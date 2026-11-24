"""Review-2 prototype of bug 3's `_resolve_param_transform`.

No source file is edited: an import hook patches the SOURCE TEXT of
phasic.svgd and phasic.mcmc in memory before they are imported, replacing the
transform-wiring blocks of SVGD.__init__ / MCMC.__init__ with a call to the
function below (injected into phasic.svgd).  Import this module BEFORE phasic.
Set PROTO_OFF=1 to load unpatched code (control).
"""
import sys, os, importlib.abc, importlib.machinery, importlib.util

FUNC = r'''

def _resolve_param_transform(prior, prior_list, positive_params, param_transform, theta_dim, where):
    """Prototype.  Returns the model-facing transform (or None); composite
    transforms carry `_per_dim_fns`.  Sets every prior's `_transform`."""
    if positive_params and param_transform is not None:
        raise ValueError("Cannot specify both positive_params=True and param_transform.")
    if param_transform is not None and not callable(param_transform):
        raise ValueError("param_transform must be a callable function")
    items = list(prior_list) if prior_list is not None else [prior]
    def nat(p):
        return None if p is None else getattr(p, '_natural_transform', None)
    if not positive_params:
        for i, p in enumerate(items):
            if nat(p) is not None:
                raise ValueError(
                    f"{where}: {type(p).__name__} needs its own transform and "
                    f"requires positive_params=True")
        tr = param_transform
        for p in items:
            if p is not None and hasattr(p, '_transform'):
                p._transform = tr      # NOTE: also RESETS to None (master never resets)
        return tr
    softplus = lambda phi: jnp.maximum(jax.nn.softplus(phi), 1e-9)
    if prior_list is not None:
        if theta_dim is not None and len(prior_list) != theta_dim:
            raise ValueError(f"prior list length ({len(prior_list)}) must match theta_dim ({theta_dim})")
        if any(nat(p) is not None for p in prior_list):
            fns = tuple(nat(p) if nat(p) is not None else softplus for p in prior_list)
            def composite(phi, _fns=fns):
                return jnp.concatenate([_fns[i](phi[..., i:i+1]) for i in range(len(_fns))], axis=-1)
            composite._per_dim_fns = fns
            for p, f in zip(prior_list, fns):
                if p is not None and hasattr(p, '_transform'):
                    p._transform = f
            return composite
        for p in prior_list:
            if p is not None and hasattr(p, '_transform'):
                p._transform = softplus
        return softplus
    tr = nat(prior) if nat(prior) is not None else softplus
    if prior is not None and hasattr(prior, '_transform'):
        prior._transform = tr
    return tr
'''

def patch_svgd(src):
    a = src.index("        # Handle parameter transformation\n        if positive_params and param_transform is not None:")
    b = src.index("        _check_gauss_prior_lower(self.prior, self.prior_list,\n                                 positive_params, \"SVGD\")")
    call = ("        self.param_transform = _resolve_param_transform(\n"
            "            self.prior, self.prior_list, positive_params, param_transform,\n"
            "            theta_dim, 'SVGD')\n\n")
    src = src[:a] + call + src[b:]
    return src + FUNC

def patch_mcmc(src):
    a = src.index("        # Parameter transformation (same as SVGD)\n")
    b = src.index("        from .svgd import _check_gauss_prior_lower\n")
    call = ("        self.positive_params = positive_params\n"
            "        from .svgd import _resolve_param_transform\n"
            "        self.param_transform = _resolve_param_transform(\n"
            "            self.prior, self.prior_list, positive_params, param_transform,\n"
            "            theta_dim, 'MCMC')\n\n")
    return src[:a] + call + src[b:]

PATCH = {'phasic.svgd': patch_svgd, 'phasic.mcmc': patch_mcmc}

class _Loader(importlib.machinery.SourceFileLoader):
    def get_data(self, path):
        data = super().get_data(path)
        if path.endswith('.py') and self.name in PATCH:
            return PATCH[self.name](data.decode()).encode()
        return data
    def get_code(self, fullname):          # bypass .pyc
        return self.source_to_code(self.get_data(self.path), self.path)

class _Finder(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path, target=None):
        if name not in PATCH:
            return None
        spec = importlib.machinery.PathFinder.find_spec(name, path)
        if spec is None:
            return None
        spec.loader = _Loader(name, spec.origin)
        return spec

if not os.environ.get('PROTO_OFF'):
    sys.meta_path.insert(0, _Finder())
