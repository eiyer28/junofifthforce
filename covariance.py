"""Parse the Durante et al. (2020) gravity-field covariance matrix.

The file `context/durante_supporting/covariancematrix.txt` (Durante et al. 2020,
supporting Dataset S2) holds a 43x43 covariance matrix. Each line starts with a
parameter label followed by 43 covariance entries. The parameter order is:

    GM, J2, C[2][1], S[2][1], C[2][2], S[2][2], J3, J4, ..., J30,
    k[2][2], k[3][1], k[3][3], k[4][2], k[4][4], RA, Dec, dRA/dt, dDec/dt

GM is in km^3/s^2; the gravity coefficients are FULLY NORMALISED.

For the fifth-force analysis we need the block for [GM, J2, J3, ..., J10], i.e.
the zonal harmonics plus GM (the GR term enters through GM).
"""

import os

import numpy as np

DEFAULT_PATH = os.path.join(
    os.path.dirname(__file__),
    "context", "durante_supporting", "covariancematrix.txt",
)


def load_full_covariance(path=DEFAULT_PATH):
    """Return (labels, C) with C the 43x43 covariance matrix and labels its rows."""
    labels = []
    rows = []
    with open(path, "r") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            labels.append(parts[0])
            rows.append([float(x) for x in parts[1:]])
    C = np.array(rows)
    if C.shape[0] != C.shape[1]:
        raise ValueError(f"covariance not square: {C.shape}")
    return labels, C


def zonal_indices(labels, n_max=10):
    """Indices (into the full matrix) for [GM, J2, J3, ..., J_{n_max}]."""
    wanted = ["GM"] + [f"J[{l}]" for l in range(2, n_max + 1)]
    idx = []
    for name in wanted:
        if name not in labels:
            raise KeyError(f"parameter {name!r} not found in covariance labels")
        idx.append(labels.index(name))
    return idx


def covariance_slice(n_max=10, path=DEFAULT_PATH):
    """Return the (n_max) x (n_max) covariance block for [GM, J2..J_{n_max}],
    in the fully-normalised basis exactly as stored in the file.
    """
    labels, C = load_full_covariance(path)
    idx = zonal_indices(labels, n_max)
    return C[np.ix_(idx, idx)]


if __name__ == "__main__":
    labels, C = load_full_covariance()
    print(f"matrix shape: {C.shape}")
    print(f"labels: {labels}")
    print(f"symmetric? max|C - C^T| = {np.max(np.abs(C - C.T)):.3e}")
    sub = covariance_slice(10)
    print(f"\n[GM, J2..J10] block shape: {sub.shape}")
    print(f"sqrt(Var[J2_normalised]) = {np.sqrt(sub[1, 1]):.3e}")
    print(f"  -> unnormalised sigma_J2 = {np.sqrt(sub[1, 1]) * np.sqrt(5):.3e}")
