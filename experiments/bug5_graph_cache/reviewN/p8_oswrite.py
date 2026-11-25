"""Does a single os.write() of more than 2 GiB write everything? (plan sec 2.6
says: os.open(...), write, os.fsync, close, os.replace). Writes to /dev/null
only; no disk space used. ~2.1 GB of RAM."""
import os, sys
n = 2**31 + 16
buf = b"x" * n
fd = os.open("/dev/null", os.O_WRONLY)
try:
    w = os.write(fd, buf)
finally:
    os.close(fd)
print(sys.platform, "requested", n, "os.write returned", w, "PARTIAL" if w < n else "complete")
del buf
