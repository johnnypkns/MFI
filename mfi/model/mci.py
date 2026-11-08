"""Multiplicative Competitive Interaction (MCI) model — Huff with several store attributes (Nakanishi & Cooper).

    P_ij = A_j · d_ij^-λ / Σ_k A_k · d_ik^-λ,      A_j = Π_m x_jm^β_m
    visits_j = Σ_i D_i · P_ij

STATUS: code ready, NOT used by the live model yet — it needs an observed demand signal per store.
The plan is Google Maps review counts (Apify compass/crawler-google-places, not yet run citywide) as the
relative-visits target, and Google attributes (rating, price level, photo count) plus unit size as x_jm.

Fit: β and λ minimise the squared error between log predicted and log observed visits, each normalised to
its mean (review counts measure relative popularity, not absolute customers).
"""
import numpy as np
from scipy.optimize import minimize


def predict(params, D, dist, X):
    """D: demand per origin (I), dist: metres (I x J), X: attributes (J x M, positive). params = [λ, β_1..β_M]."""
    lam, beta = params[0], params[1:]
    A = np.exp(np.log(X) @ beta)                                  # J
    w = A[None, :] * np.maximum(dist, 50.0) ** (-lam)             # I x J
    P = w / w.sum(1, keepdims=True)
    return (D[:, None] * P).sum(0)


def fit(D, dist, X, visits, l2=0.01):
    """Returns dict(lambda, beta, r2_log, n). visits: observed relative visits per store (e.g. review counts)."""
    X = np.asarray(X, float)
    obs = np.log(np.asarray(visits, float) / np.mean(visits))

    def loss(p):
        if p[0] <= 0:
            return 1e9
        pr = predict(p, D, dist, X)
        e = np.log(pr / pr.mean()) - obs
        return float((e ** 2).mean() + l2 * (p[1:] ** 2).sum())

    p0 = np.r_[1.5, np.zeros(X.shape[1])]
    r = minimize(loss, p0, method="Nelder-Mead", options={"maxiter": 4000, "xatol": 1e-4, "fatol": 1e-8})
    pr = predict(r.x, D, dist, X)
    e = np.log(pr / pr.mean()) - obs
    return {"lambda": float(r.x[0]), "beta": r.x[1:].tolist(), "r2_log": float(1 - e.var() / obs.var()),
            "n": len(visits), "converged": bool(r.success)}
