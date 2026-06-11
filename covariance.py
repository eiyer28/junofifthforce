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

# Updated solution (PJ37 normal-modes); ships names + a label-free covariance matrix
# in the SAME fully-normalised basis as Durante.
NEW_SOLUTION_DIR = os.path.join(
    os.path.dirname(__file__),
    "solution_ref_GRAVtoPJ37+PJ01_normal-modes_jnCnstr-0.1mGal",
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


def _map_new_label(raw):
    """Map a new-solution parameter name to the short label scheme used here.

    'Gm/Jupiter'            -> 'GM'
    'Gravity/Jupiter/J[3]'  -> 'J[3]'
    anything else           -> the raw (stripped) string.
    """
    raw = raw.strip()
    if raw == "Gm/Jupiter":
        return "GM"
    prefix = "Gravity/Jupiter/"
    if raw.startswith(prefix):
        return raw[len(prefix):]
    return raw


def load_new_solution(directory=NEW_SOLUTION_DIR):
    """Load the updated solution's covariance.

    The folder stores parameter names (names.txt) and a label-free NxN numeric
    matrix (covariance.txt) separately. Returns (labels, C) with labels mapped to
    the same short scheme as the Durante loader ('GM', 'J[2]', ...).
    """
    names_path = os.path.join(directory, "names.txt")
    cov_path = os.path.join(directory, "covariance.txt")

    labels = []
    with open(names_path, "r") as fh:
        for line in fh:
            if line.strip():
                labels.append(_map_new_label(line))

    rows = []
    with open(cov_path, "r") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rows.append([float(x) for x in line.split()])
    C = np.array(rows)

    if C.shape[0] != C.shape[1]:
        raise ValueError(f"new covariance not square: {C.shape}")
    if C.shape[0] != len(labels):
        raise ValueError(
            f"new covariance size {C.shape[0]} != number of labels {len(labels)}"
        )
    return labels, C


def covariance_slice_new(n_max=10, directory=NEW_SOLUTION_DIR):
    """Return the (n_max) x (n_max) [GM, J2..J_{n_max}] block of the updated
    solution, in the same fully-normalised basis as the Durante slice.

    The full matrix carries degenerate/unconstrained entries (inertia,
    Lense-Thirring ~1e36) that lie OUTSIDE this block, so we slice first and never
    operate on the full 48x48.
    """
    labels, C = load_new_solution(directory)
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

    new_labels, new_C = load_new_solution()
    print(f"\nnew solution matrix shape: {new_C.shape}")
    new_sub = covariance_slice_new(10)
    print(f"new [GM, J2..J10] block shape: {new_sub.shape}")
    print(f"new sqrt(Var[J2_normalised]) = {np.sqrt(new_sub[1, 1]):.3e}"
          f"  (Durante {np.sqrt(sub[1, 1]):.3e})")
    print("diagonal sigma ratio (new/old) per [GM, J2..J10]:")
    ratio = np.sqrt(np.diag(new_sub) / np.diag(sub))
    print("  " + ", ".join(f"{r:.2f}" for r in ratio))
