import matplotlib.pyplot as plt
import numpy as np

from sde.brownian_motion import brownian_motion
from sde.numerical_schemes import euler_maruyama_1d


def b(x, t):
    return np.sign(x)


def sigma(x, t):
    eps = 100
    return np.sqrt(eps)


if __name__ == "__main__":
    t0 = 0
    t1 = 50
    n = 1000
    times = np.linspace(t0, t1, n)
    x0 = 0
    M = 10

    rng = np.random.default_rng(42)
    X = np.zeros((M, len(times)))

    fig, ax = plt.subplots(figsize=(12, 8))

    for i in range(M):
        W = brownian_motion(times=times, rng=rng)
        X[i, :] = euler_maruyama_1d(b=b, sigma=sigma, times=times, x0=x0, W=W, rng=None)
        ax.plot(times, X[i, :])

    plt.tight_layout()
    plt.show()
