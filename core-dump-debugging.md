  1. Enable core dumps (macOS)

  By default macOS disables core files. To enable them for the kernel's process:

  # In the shell that will launch jupyter / pixi
  ulimit -c unlimited

  # macOS writes cores to /cores/core.<pid> — make sure it's writable
  ls -ld /cores
  sudo chmod 1777 /cores   # one-time, if needed

  Then launch Jupyter from that same shell:

  ulimit -c unlimited
  pixi run jupyter lab

  When the kernel segfaults you get /cores/core.<pid>. Inspect with lldb:

  lldb -c /cores/core.<pid> -- $(pixi run which python)
  (lldb) bt all

  Caveats on recent macOS:
  - SIP-protected / hardened binaries don't dump cores. The Python in a pixi env is normally fine, but
  if you get nothing, check codesign -d --entitlements - $(which python) for
  com.apple.security.get-task-allow.
  - Cores can be huge (the entire RSS — easily several GB for a phasic SVGD run).

  2. Get a Python + C traceback without a core file

  Often more useful than a raw core. Add this to the first cell of the notebook (or to a kernel startup
  file):

  import faulthandler, sys
  faulthandler.enable(file=sys.stderr, all_threads=True)

  On segfault you'll get a C-level stack trace (with phasic's .so frames) printed to the kernel's stderr
   — visible in the terminal that launched jupyter, or in the Jupyter log. This is usually enough to
  localize which phasic call crashed.

  To also dump on hangs:

  faulthandler.dump_traceback_later(60, repeat=True)

  3. Run the failing cell under a debugger

  When you have a reproducer, skip the kernel entirely:

  ulimit -c unlimited
  lldb -- $(pixi run which python) repro.py
  (lldb) run
  # ... crashes ...
  (lldb) bt all
  (lldb) frame variable

  This gives you a live debugger session at the crash site — strictly better than a post-mortem core for
   diagnosing phasic FFI / C-extension bugs.

  ---
  Recommendation for your case: start with faulthandler.enable() in the notebook — it's zero-friction
  and almost always pinpoints the offending phasic call. Fall back to lldb -- python repro.py once you
  have a minimal reproducer. Only bother with /cores dumps if the crash is non-deterministic and you
  can't reproduce on demand.
