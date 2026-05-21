import numpy as np
from numpy.random import Generator


def brownian_motion(times: np.ndarray, rng: Generator | None =None) -> np.ndarray:
    times = np.asarray(times)
    dt = np.diff(times)

    if rng is None:
        rng = np.random.default_rng()

    Z = rng.normal(0.0, 1.0, size=len(dt))
    dW = np.sqrt(dt) * Z

    W = np.empty(len(times))
    W[0] = 0.0
    W[1:] = np.cumsum(dW)

    return W


def brownian_motion_grid(t0: float =0.0, t1: float =1.0, n: int =10) -> tuple[np.ndarray, np.ndarray]:
    times = np.linspace(t0, t1, n)
    return times, brownian_motion(times)
