import numpy as np

from mfi.model import mci


def test_mci_recovers_known_parameters():
    rng = np.random.default_rng(0)
    org, sto = rng.uniform(0, 5000, (300, 2)), rng.uniform(0, 5000, (25, 2))
    dist = np.hypot(*(org[:, None, :] - sto[None, :, :]).transpose(2, 0, 1))
    D = rng.uniform(100, 1000, 300)
    X = np.c_[rng.uniform(40, 300, 25), rng.uniform(3.5, 4.8, 25)]       # size m², rating
    true = np.r_[1.8, 0.6, 2.0]
    visits = mci.predict(true, D, dist, X)
    r = mci.fit(D, dist, X, visits, l2=0.0)
    assert abs(r["lambda"] - 1.8) < 0.2 and abs(r["beta"][0] - 0.6) < 0.15 and r["r2_log"] > 0.95
