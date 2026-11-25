"""reviewY y8: does writing one large ASCII str (a json.dumps field) through
os.fdopen(fd, "w", encoding="utf-8") create a full bytes copy (peak = str +
bytes), contrary to plan §2.6 'streaming avoids holding the string and its
bytes at once'? tracemalloc counts Python-heap allocations."""
import os, sys, tracemalloc, tempfile
d = sys.argv[1]
s = "1" * (100 * 1024 * 1024)          # 100 MiB ASCII str
for mode in ("text", "binary-encode", "binary-chunked"):
    p = os.path.join(d, f"y8_{mode}.tmp")
    fd = os.open(p, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o666)
    tracemalloc.start(); tracemalloc.reset_peak()
    if mode == "text":
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(s)
    elif mode == "binary-encode":
        with os.fdopen(fd, "wb") as f:
            f.write(s.encode("ascii"))
    else:
        with os.fdopen(fd, "wb") as f:
            mv = 8 * 1024 * 1024
            for i in range(0, len(s), mv):
                f.write(s[i:i + mv].encode("ascii"))
    cur, peak = tracemalloc.get_traced_memory(); tracemalloc.stop()
    os.remove(p)
    print(f"{mode:15s} extra peak during write: {peak/2**20:7.1f} MiB (str itself 100 MiB, pre-existing)")
