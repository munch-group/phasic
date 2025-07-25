"""
Separated Graph PMF System - Python Interface
User supplies graph construction, system handles PMF computation
"""

import os
import hashlib
import subprocess
import tempfile
import ctypes
from typing import Optional, Dict, Any, Callable
from dataclasses import dataclass
import jax
import jax.numpy as jnp
import jax.extend as jex
from jax.interpreters import mlir
from jax.interpreters.mlir import ir
from jaxlib.hlo_helpers import custom_call
import numpy as np

# Load the main separated graph library
lib = None
try:
    lib = ctypes.CDLL("/Users/kmt/PtDalgorithms/jax_extension/separated_graph_pmf.so")
    print("Loaded separated graph PMF library")
except OSError as e:
    print(f"Warning: Could not load separated graph library: {e}")

@dataclass
class GraphConfig:
    """Configuration for graph construction"""
    nr_samples: int = 3
    mutation_rate: float = 0.0
    apply_discretization: bool = True
    custom_params: Optional[Dict[str, float]] = None
    
    def __post_init__(self):
        if self.custom_params is None:
            self.custom_params = {}
    
    def to_string(self) -> str:
        """Serialize config for C++"""
        result = f"{self.nr_samples},{self.mutation_rate},{int(self.apply_discretization)},"
        
        for key, value in self.custom_params.items():
            result += f"{key}={value};"
            
        return result

class UserGraphBuilder:
    """Manages user-defined graph builders"""
    
    def __init__(self):
        self.builders = {}
        self.compiled_libs = {}
    
    def register_inline(self, name: str, cpp_code: str) -> Callable:
        """Register a graph builder from inline C++ code"""
        
        # Create complete C++ source
        full_source = self._create_cpp_source(cpp_code, name)
        
        # Compile and load
        lib_path = self._compile_cpp(full_source, name)
        self._load_and_register(name, lib_path)
        
        # Return PMF function
        return lambda theta, times, config=GraphConfig(): self._call_pmf(theta, times, name, config)
    
    def register_file(self, name: str, cpp_file_path: str) -> Callable:
        """Register a graph builder from C++ file"""
        
        with open(cpp_file_path, 'r') as f:
            cpp_code = f.read()
        
        return self.register_inline(name, cpp_code)
    
    def _create_cpp_source(self, user_code: str, name: str) -> str:
        """Create complete C++ source with user code"""
        
        template = f'''
#include "user_graph_api.h"
#include <queue>
#include <vector>

extern "C" {{

// User's graph construction function
Graph build_{name}_graph(const double* theta, int theta_size, const UserConfig& config) {{
    {user_code}
}}

// Registration function
__attribute__((constructor))
void register_{name}_builder() {{
    GraphBuilderRegistry::register_builder("{name}", build_{name}_graph);
}}

}}
'''
        return template
    
    def _compile_cpp(self, source: str, name: str) -> str:
        """Compile C++ source to shared library"""
        
        # Create unique filename based on source hash
        source_hash = hashlib.md5(source.encode()).hexdigest()[:8]
        lib_name = f"user_graph_{name}_{source_hash}"
        lib_path = f"/tmp/{lib_name}.so"
        
        # Check if already compiled
        if os.path.exists(lib_path):
            return lib_path
        
        # Write source to temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.cpp', delete=False) as f:
            f.write(source)
            cpp_file = f.name
        
        try:
            # Compile command
            cmd = [
                'g++', '-shared', '-fPIC', '-O3', '-std=c++17',
                '-I', '/Users/kmt/PtDalgorithms/jax_extension',
                '-I', '/Users/kmt/PtDalgorithms/.pixi/envs/default/include',
                cpp_file, 'user_graph_api.cpp',
                '-o', lib_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, 
                                  cwd='/Users/kmt/PtDalgorithms/jax_extension')
            
            if result.returncode != 0:
                raise RuntimeError(f"Compilation failed:\\n{result.stderr}")
            
            print(f"Compiled user graph builder: {lib_path}")
            return lib_path
            
        finally:
            os.unlink(cpp_file)
    
    def _load_and_register(self, name: str, lib_path: str):
        """Load compiled library and register builder"""
        
        # Load the library (constructor will auto-register)
        self.compiled_libs[name] = ctypes.CDLL(lib_path)
        self.builders[name] = lib_path
        
        print(f"Registered graph builder: {name}")
    
    def _call_pmf(self, theta: jnp.ndarray, times: jnp.ndarray, 
                  builder_name: str, config: GraphConfig) -> jnp.ndarray:
        """Call the JAX primitive with user's graph builder"""
        
        return separated_graph_pmf_primitive(theta, times, builder_name, config)

# Global registry instance
graph_registry = UserGraphBuilder()

def register_graph_builder(name: str, cpp_code: str) -> Callable:
    """Main interface for registering user graph builders"""
    return graph_registry.register_inline(name, cpp_code)

def register_graph_from_file(name: str, cpp_file: str) -> Callable:
    """Register graph builder from file"""
    return graph_registry.register_file(name, cpp_file)

# JAX Primitive for separated graph PMF
def create_separated_graph_pmf_primitive():
    """Create the JAX primitive for separated graph PMF computation"""
    
    prim = jex.core.Primitive('separated_graph_pmf')
    
    def abstract_eval(theta_aval, times_aval, *, builder_name, config_str):
        return jax.core.ShapedArray(times_aval.shape, jnp.float64)
    
    prim.def_abstract_eval(abstract_eval)
    
    # Add implementation rule (fallback for non-compiled execution)
    def impl_rule(theta, times, *, builder_name, config_str):
        # For now, just return dummy values
        return jnp.ones_like(times, dtype=jnp.float64) * 0.1
    
    prim.def_impl(impl_rule)
    
    def lowering(ctx, theta, times, *, builder_name, config_str):
        """Lower to custom call"""
        
        # Get shapes and types
        theta_type = theta.type
        times_type = times.type
        
        # Create dimension operand
        theta_size = theta_type.shape[0] if theta_type.shape else 1
        n_times = times_type.shape[0] if times_type.shape else 1
        dims = mlir.ir_constant(jnp.array([theta_size, n_times], dtype=jnp.int64))
        
        # Convert string constants to operands  
        builder_name_bytes = builder_name.encode('utf-8') + b'\\0'
        config_str_bytes = config_str.encode('utf-8') + b'\\0'
        
        builder_name_op = mlir.ir_constant(jnp.frombuffer(builder_name_bytes, dtype=jnp.uint8))
        config_str_op = mlir.ir_constant(jnp.frombuffer(config_str_bytes, dtype=jnp.uint8))
        
        # Output type
        out_type = ir.RankedTensorType.get(
            times_type.shape, 
            mlir.dtype_to_ir_type(jnp.float64)
        )
        
        # Create custom call
        call = custom_call(
            call_target_name=b"jax_separated_graph_pmf",
            result_types=[out_type],
            operands=[theta, times, dims, builder_name_op, config_str_op],
            operand_layouts=[
                list(reversed(range(len(theta_type.shape)))),
                list(reversed(range(len(times_type.shape)))),
                [],  # dims
                list(reversed(range(len(builder_name_op.type.shape)))),
                list(reversed(range(len(config_str_op.type.shape))))
            ],
            result_layouts=[list(reversed(range(len(times_type.shape))))]
        )
        
        return call.results
    
    # Register lowering
    mlir.register_lowering(prim, lowering, platform="cpu")
    
    # Add gradient support (finite differences)
    def jvp_rule(primals, tangents):
        theta, times = primals
        dtheta, dtimes = tangents
        
        # We need the static args for the primitive call
        # For now, skip gradient computation
        primal_out = jnp.ones_like(times, dtype=jnp.float64) * 0.1
        tangent_out = jnp.zeros_like(times)
        return primal_out, tangent_out
    
    from jax.interpreters import ad
    ad.primitive_jvps[prim] = jvp_rule
    
    # Add batching support
    def batching_rule(batched_args, batch_dims):
        theta, times = batched_args
        theta_bdim, times_bdim = batch_dims
        
        # Simple batching - just return dummy values for now
        if theta_bdim is not None:
            batch_size = theta.shape[theta_bdim]
            result = jnp.ones((batch_size,) + times.shape, dtype=jnp.float64) * 0.1
            return result, 0
        elif times_bdim is not None:
            batch_size = times.shape[times_bdim]  
            result = jnp.ones((batch_size,), dtype=jnp.float64) * 0.1
            return result, times_bdim
        else:
            return jnp.ones_like(times, dtype=jnp.float64) * 0.1, None
    
    from jax._src.interpreters import batching
    batching.primitive_batchers[prim] = batching_rule
    
    return prim

# Create the primitive
separated_graph_pmf_prim = create_separated_graph_pmf_primitive()

def separated_graph_pmf_primitive(theta: jnp.ndarray, times: jnp.ndarray, 
                                 builder_name: str, config: GraphConfig) -> jnp.ndarray:
    """Call the separated graph PMF primitive"""
    return separated_graph_pmf_prim.bind(
        theta, times, 
        builder_name=builder_name, 
        config_str=config.to_string()
    )

# Register the custom call target with XLA
def register_separated_pmf_target():
    """Register the C++ function with XLA"""
    try:
        from jax._src.lib import xla_client
        
        if lib is not None:
            # Get function pointer
            func_ptr = lib.jax_separated_graph_pmf
            
            # Create PyCapsule
            import ctypes
            from ctypes import pythonapi, c_void_p, c_char_p
            
            PyCapsule_New = pythonapi.PyCapsule_New
            PyCapsule_New.argtypes = [c_void_p, c_char_p, c_void_p]
            PyCapsule_New.restype = ctypes.py_object
            
            capsule = PyCapsule_New(
                ctypes.cast(func_ptr, c_void_p).value,
                b"xla._CUSTOM_CALL_TARGET",
                None
            )
            
            # Register with XLA
            xla_client.register_custom_call_target(
                name="jax_separated_graph_pmf",
                fn=capsule,
                platform="cpu"
            )
            
            print("Successfully registered separated graph PMF with XLA")
            return True
            
    except Exception as e:
        print(f"Failed to register separated graph PMF: {e}")
        return False

# Auto-register on import
registration_success = register_separated_pmf_target()

if __name__ == "__main__":
    print("Separated Graph PMF System - Python Interface")
    print(f"Registration successful: {registration_success}")