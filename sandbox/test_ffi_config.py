#!/usr/bin/env python
"""Check FFI configuration."""

from phasic import Graph, get_config

config = get_config()
print(f"FFI enabled: {config.ffi}")
print(f"Config: {config}")
