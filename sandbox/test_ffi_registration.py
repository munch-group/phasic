"""Test if FFI registration works"""
import phasic
import sys

print("Testing FFI registration...")

config = phasic.get_config()
print(f"config.ffi = {config.ffi}")

try:
    from phasic.ffi_wrappers import _register_ffi_targets
    print("Attempting to register FFI targets...")
    sys.stdout.flush()
    _register_ffi_targets()
    print("✓ FFI registration succeeded!")
except Exception as e:
    print(f"✗ FFI registration FAILED: {e}")
    import traceback
    traceback.print_exc()
