import numpy as np

from sde.brownian_motion import brownian_motion
from sde.numerical_schemes import euler_maruyama


def test_euler_maruyama_matches_manual_recursion():
    times = np.linspace(0.0, 1.0, 101)
    m = 1.0
    spring = 0.5
    damping = 0.2
    sigma = np.sqrt(100.0)

    a_matrix = np.array([[0.0, 1.0], [-spring / m, -damping / m]])
    x0 = np.array([0.0, 0.0])

    rng = np.random.default_rng(1234)
    brownian_path = brownian_motion(times, rng)

    drift = lambda x, t: a_matrix @ x
    diffusion = lambda x, t: np.array([0.0, sigma])

    result = euler_maruyama(
        b=drift,
        sigma=diffusion,
        times=times,
        x0=x0,
        W=brownian_path,
        n=2,
        k=1,
    )

    expected = np.zeros_like(result)
    expected[:, 0] = x0

    dt = np.diff(times)
    dW = np.diff(brownian_path)

    for index in range(len(dt)):
        expected[:, index + 1] = (
            expected[:, index]
            + a_matrix @ expected[:, index] * dt[index]
            + diffusion(expected[:, index], times[index]) * dW[index]
        )

    assert np.allclose(result, expected)
    assert result.shape == (2, len(times))
    assert np.isfinite(result).all()
