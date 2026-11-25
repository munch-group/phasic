"""reviewS s9: does a write through os.fdopen(fd, 'w') call os.write? (test-26 short-write feasibility)
Writes one small temp file under the scratchpad and deletes it."""
import os, sys
p = sys.argv[1]
calls = {"n": 0}
real = os.write
def spy(fd, data):
    calls["n"] += 1
    return real(fd, data[: max(1, len(data) // 2)])   # short write
os.write = spy
fd = os.open(p, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o666)
with os.fdopen(fd, "w", encoding="utf-8") as f:
    f.write("x" * 100000)
    f.flush(); os.fsync(f.fileno())
os.write = real
print("os.write calls during fdopen text write:", calls["n"], " file size:", os.path.getsize(p))
os.remove(p)
