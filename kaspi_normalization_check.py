"""Normalization / sigma-level check for the Kaspi et al. (2023) covariance matrix.

Reproduces, for the Weizmann (Kaspi 2023) PJ37 normal-modes covariance, the
exercise the user did on the Durante et al. (2020) supplement:

    Durante finding (control):
        (3*sigma_table)^2 / C_ll = (2l+1)      i.e.   sqrt(C_ll) = 3*sigma_table / sqrt(2l+1)
    -> the Durante covariance is at 3-sigma, stored for the fully-normalised
       coefficient  Jbar_l = J_l / sqrt(2l+1).

Collaborator's claim to test for Kaspi:
        the covariance is at 1-sigma, stored for Jbar_l = J_l / sqrt(2l+1),
        i.e.   sqrt(C_ll) =?= 1*sigma_table / sqrt(2l+1)
We test BOTH the 1-sigma and 3-sigma hypotheses and print the diagnostic ratios.

Uncertainties:
  * Durante Table 2 (3-sigma) low-degree J's - only J2, J3 quoted by the user,
    used purely as a control that the pipeline reproduces the clean (2l+1) factor.
  * Kaspi 2023 Extended Data Table 1 (units 1e-6), read from the published table.
"""

import numpy as np

import covariance as cov

# ---------------------------------------------------------------------------
# Published uncertainties
# ---------------------------------------------------------------------------
# Durante et al. 2020, Table 2 ("Uncertainty (x1e-6)", quoted as 3*sigma), J2..J10:
DURANTE_3SIGMA = {l: u * 1e-6 for l, u in {
    2: 0.0017, 3: 0.0033, 4: 0.0024, 5: 0.0042, 6: 0.0067,
    7: 0.012, 8: 0.021, 9: 0.036, 10: 0.065}.items()}

# Kaspi et al. 2023, Extended Data Table 1 -- "Uncertainty (x1e-6)" column.
# (value x1e-6, uncertainty x1e-6); we only need the uncertainty here.
KASPI_UNC_1E6 = {
    2: 0.00163, 3: 0.00263, 4: 0.00213, 5: 0.00253, 6: 0.00203,
    7: 0.00249, 8: 0.00209, 9: 0.00257, 10: 0.00295,
}
KASPI_SIGMA = {l: u * 1e-6 for l, u in KASPI_UNC_1E6.items()}  # -> dimensionless


def diag_by_l(cov_block, n_max=10):
    """Return {l: C_ll} for l = 2..n_max from a [GM, J2..Jn] covariance block."""
    # index 0 is GM; J_l sits at position (l-1)
    return {l: cov_block[l - 1, l - 1] for l in range(2, n_max + 1)}


def report(name, diag, sigma_table):
    """Four hypotheses = {basis} x {sigma-level}; a hypothesis holds iff its column
    is constant (value ~1 -> 1 sigma in that basis, ~3 -> 3 sigma)."""
    print(f"\n{'='*92}\n{name}\n{'='*92}")
    print(f"{'l':>3} {'2l+1':>5} {'s_cov':>11} {'s_raw':>11} {'sigma_tab':>11}"
          f" | {'norm/1σ':>8} {'norm/3σ':>8} | {'other/1σ':>9} {'other/3σ':>9}")
    cols = {"norm/1σ": [], "norm/3σ": [], "other/1σ": [], "other/3σ": []}
    for l in sorted(diag):
        if l not in sigma_table:
            continue
        C = diag[l]
        s = sigma_table[l]
        s_cov = np.sqrt((2 * l + 1) * C)   # fully-normalised (Durante) basis
        s_raw = np.sqrt(C)                 # 'other' (already un-normalised) basis
        v = {"norm/1σ": s_cov / s, "norm/3σ": s_cov / (3 * s),
             "other/1σ": s_raw / s, "other/3σ": s_raw / (3 * s)}
        for k in cols:
            cols[k].append(v[k])
        print(f"{l:>3} {2*l+1:>5} {s_cov:>11.3e} {s_raw:>11.3e} {s:>11.3e}"
              f" | {v['norm/1σ']:>8.3f} {v['norm/3σ']:>8.3f} |"
              f" {v['other/1σ']:>9.3f} {v['other/3σ']:>9.3f}")
    print("  hypothesis holds if its column is constant (≈1 ⇒ that basis & σ-level):")
    for k, val in cols.items():
        val = np.array(val)
        flag = "  <-- FLAT" if val.max() / val.min() < 1.10 else ""
        print(f"    {k:>9}: mean={val.mean():.3f}, "
              f"range=[{val.min():.3f}, {val.max():.3f}], "
              f"max/min={val.max()/val.min():.2f}{flag}")


# ---------------------------------------------------------------------------
# Durante control
# ---------------------------------------------------------------------------
dur = cov.covariance_slice(10)
report("DURANTE 2020  (control: user says table is 3-sigma)",
       diag_by_l(dur), DURANTE_3SIGMA)

# ---------------------------------------------------------------------------
# Kaspi 2023
# ---------------------------------------------------------------------------
kas = cov.covariance_slice_new(10)
report("KASPI 2023  (claim: 1-sigma, Jbar = J/sqrt(2l+1))",
       diag_by_l(kas), KASPI_SIGMA)

# ---------------------------------------------------------------------------
# Off-diagonal correlation structure of the Kaspi J-block (drives sigma_omega(N))
# ---------------------------------------------------------------------------
print(f"\n{'='*78}\nKASPI 2023  correlation matrix of [GM, J2..J10]\n{'='*78}")
d = np.sqrt(np.diag(kas))
corr = kas / np.outer(d, d)
labels = ["GM"] + [f"J{l}" for l in range(2, 11)]
print("      " + " ".join(f"{lab:>6}" for lab in labels))
for i, lab in enumerate(labels):
    print(f"{lab:>5} " + " ".join(f"{corr[i, j]:>6.2f}" for j in range(len(labels))))

# nearest-neighbour J-J correlations
print("\nadjacent-J correlations (Kaspi):")
for l in range(2, 10):
    print(f"  corr(J{l}, J{l+1}) = {corr[l-1, l]:>6.2f}")

# Durante correlation for comparison
print(f"\n{'='*78}\nDURANTE 2020  correlation matrix of [GM, J2..J10]\n{'='*78}")
dd = np.sqrt(np.diag(dur))
corr_d = dur / np.outer(dd, dd)
print("      " + " ".join(f"{lab:>6}" for lab in labels))
for i, lab in enumerate(labels):
    print(f"{lab:>5} " + " ".join(f"{corr_d[i, j]:>6.2f}" for j in range(len(labels))))


# ---------------------------------------------------------------------------
# Why sigma_omega(N) is flat: diagonal vs off-diagonal (cross-term) budget
# ---------------------------------------------------------------------------
from constants import A_KM, ECC, INC, OMEGA, R_J_KM
from precession_gravity import gradient_vector


def sigma_budget(name, C):
    """sigma_omega^2 broken into diagonal and off-diagonal contributions vs N.

    Uses the fully-normalised gradient (apply_normalization=True) so it contracts
    consistently with the fully-normalised covariance.
    """
    print(f"\n{'='*78}\n{name}: sigma_omega(N) diagonal vs cross-term budget\n{'='*78}")
    print(f"{'N':>3} {'sigma_om':>11} {'sqrt(diag-only)':>16} {'cross/diag':>11}")
    for n_max in range(2, 11):
        g = gradient_vector(A_KM, ECC, INC, OMEGA, R_J_KM, n_max,
                            apply_normalization=True)
        Cb = C[:n_max, :n_max]
        var = float(g @ Cb @ g)
        diag_only = float(np.sum((g ** 2) * np.diag(Cb)))
        cross = var - diag_only
        print(f"{n_max:>3} {np.sqrt(max(var,0)):>11.4e} "
              f"{np.sqrt(max(diag_only,0)):>16.4e} {cross/diag_only:>11.3f}")


sigma_budget("KASPI 2023", kas)
sigma_budget("DURANTE 2020", dur)
