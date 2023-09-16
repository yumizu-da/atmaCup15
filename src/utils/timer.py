import time
from contextlib import contextmanager


@contextmanager
def timer(name: str):
    """時間計測用のcontextmanager"""
    t0 = time.time()
    yield
    print(f"[{name}] done in {time.time() - t0:.0f} s")
