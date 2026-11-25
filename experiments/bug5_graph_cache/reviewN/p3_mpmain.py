"""Under multiprocessing 'spawn' (the macOS default; 'forkserver' is the Linux
default from 3.14), a callback defined in the main script has
__module__ == '__mp_main__' in workers but '__main__' in the parent. The plan
keys f.__module__ (sec 2.2.1), so parent and workers compute different keys
for the same callback."""
import multiprocessing as mp


def cb(state, rate=1.0):
    return []


def make(rate):
    def inner(state):
        return [rate]
    return inner


def child(_):
    return cb.__module__, make(1.0).__module__


if __name__ == "__main__":
    print("parent:", cb.__module__, make(1.0).__module__)
    for method in ("spawn", "forkserver", "fork"):
        try:
            ctx = mp.get_context(method)
            with ctx.Pool(1) as p:
                print(method, "worker:", p.map(child, [0])[0])
        except Exception as e:
            print(method, "failed", type(e).__name__, e)
