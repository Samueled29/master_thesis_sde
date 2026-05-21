import numpy as np
from scipy import stats
from sde.brownian_motion import brownian_motion


def generate_paths() -> tuple[np.ndarray, np.ndarray]:
    times = np.linspace(0, 1, 1000)
    rng = np.random.default_rng(123)
    paths = np.array([brownian_motion(times, rng) for _ in range(500)])
    return times, paths


# Mean of increments is 0
def test_mean_increments():
    """Test if the mean of brownian motion's final RV is 0."""
    _, paths = generate_paths()
    final = paths[:, -1]
    mean = np.mean(final)

    assert abs(mean) < 0.05


# Variance of increments is t
def test_var_increments():
    """Test if the variance of brownian motion's final RV at
    time 1 is 1 (grows as t)."""
    _, paths = generate_paths()
    final = paths[:, -1]
    var = np.var(final)

    assert abs(var - 1) < 0.1


def test__msd_linearity():
    """Check that mean square displacement grows
    linearly with time"""
    times, paths = generate_paths()
    msd = np.mean(paths**2, axis=0)
    slope, intercept, r, *_ = stats.linregress(times, msd)

    assert r**2 > 0.99  # Linearity
    assert abs(slope - 1) < 0.05


def test_normality():
    """Test the normal dsitribution of increments"""
    _, paths = generate_paths()
    dW = np.diff(paths, axis=1).flatten()
    _, p = stats.normaltest(dW)

    assert p > 0.05


def test_increments_uncorrelation():
    times = np.linspace(0, 1, 1000)
    path = brownian_motion(times)
    dW = np.diff(path)

    corr = np.corrcoef(dW[:-1], dW[1:])[0, 1]

    assert abs(corr) < 0.05
