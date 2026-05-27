from typing import Callable

import numpy as np
from numpy.random import Generator

from sde.brownian_motion import brownian_motion


def euler_maruyama(
    b: Callable,
    sigma: Callable,
    times: np.ndarray,
    x0: np.ndarray,
    W: np.ndarray | None = None,
    rng: Generator | None = None,
    n: int = 1,
    k: int = 1,
) -> np.ndarray:

    T = len(times)

    if W is None:
        W = np.array([brownian_motion(times, rng) for _ in range(k)])
        if k == 1:
            W = W.flatten()
    else:
        W = np.asarray(W)
        if W.ndim == 1 and k == 1:
            _ = ""
        elif W.ndim == 1 and k != 1:
            raise ValueError(f"W must have shape ({k}, {T}), got {W.shape}")
        elif W.ndim != 2 or W.shape != (k, T):
            raise ValueError(f"W must have shape ({k}, {T}), got {W.shape}")

    X = np.zeros((n, T))
    X[:, 0] = x0

    if k == 1:
        dW = np.diff(W)
    else:
        dW = np.diff(W, axis=1)
    dt = np.diff(times)

    for i in range(T - 1):
        drift = np.asarray(b(X[:, i], times[i]))
        diffusion = np.asarray(sigma(X[:, i], times[i]))

        if n == 1 and drift.shape == ():
            drift = drift.reshape(1)
        elif drift.shape != (n,):
            raise ValueError(f"b must return shape ({n},), got {drift.shape}")

        if n == 1 and k == 1 and diffusion.shape == ():
            diffusion = diffusion.reshape(1)
        elif n != 1 and k == 1:
            diffusion = diffusion.reshape(n)
        elif n == 1 and k != 1:
            diffusion = diffusion.reshape(1, k)
        elif diffusion.shape != (n, k):
            raise ValueError(
                f"sigma must return shape ({n}, {k}), got {diffusion.shape}"
            )

        if k != 1:
            X[:, i + 1] = X[:, i] + drift * dt[i] + diffusion @ dW[:, i]
        elif k == 1:
            X[:, i + 1] = X[:, i] + drift * dt[i] + diffusion * dW[i]

    return X


def euler_maruyama_1d(
    b: Callable,
    sigma: Callable,
    times: np.ndarray,
    x0: float,
    W: np.ndarray | None = None,
    rng: Generator | None = None,
) -> np.ndarray:

    T = len(times)

    if W is None:
        W = brownian_motion(times, rng)
    else:
        W = np.asarray(W)
        if W.ndim != 1 or W.shape != (T,):
            raise ValueError(f"W must have shape ({T},), got {W.shape}")

    X = np.zeros(T)
    X[0] = x0

    dW = np.diff(W)
    dt = np.diff(times)

    for i in range(T - 1):
        drift = b(X[i], times[i])
        diffusion = sigma(X[i], times[i])

        if np.shape(drift) != ():
            raise ValueError(f"b must return a scalar, got shape {np.shape(drift)}")
        if np.shape(diffusion) != ():
            raise ValueError(
                f"sigma must return a scalar, got shape {np.shape(diffusion)}"
            )

        X[i + 1] = X[i] + drift * dt[i] + diffusion * dW[i]

    return X

def explicit_euler_1d(b: Callable,
    times: np.ndarray,
    x0: float
) -> np.ndarray:
    
    T = len(times)

    X = np.zeros(len(times))
    X[0] = x0

    dt = np.diff(times)

    for i in range(len(times)-1):
        drift = b(X[i], times[i])

        if np.shape(drift) != ():
            raise ValueError(f"b must return a scalar, got shape {np.shape(drift)}")
        
        X[i+1] = X[i] + drift * dt[i]
    return X