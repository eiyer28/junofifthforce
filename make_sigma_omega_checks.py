"""Two sigma_omega checks over Juno's realised orbit geometry, comparing the
Durante (2020) and re-derived cov_220824 gravity covariances on a
*basis-consistent* footing (same convention as the plateau plot).

Background
----------
Singh et al. JHEP 01 (2025) 098, Figure 3:
  * LEFT  - a filled contour of the precession-angle-drift uncertainty sigma_omega
            (their eq. 3.5) in the (omega, i) plane, at fixed truncation N = 10,
            using the Durante covariance. Red dots mark Juno orbits PJ01 + PJ03-PJ17
            (PJ02 was lost), rightmost = PJ01. sigma_omega spans 7.1e-10 to 9.0e-10,
            maximal at PJ01 (i = 1.57, omega = 3.08).
  * RIGHT - sigma_omega versus N at the fixed PJ01 geometry.

Basis convention (matches the notebook plateau)
-----------------------------------------------
  * Durante  : fully-normalised COV, contracted with the *normalised* gradient
               (sqrt(2l+1) ON); GM dropped.
  * cov_220824: un-normalised J2..JN block, contracted with the *raw* gradient
               (sqrt(2l+1) OFF).
This yields the defensible ~1.3x improvement at PJ01 (not the inflated ~2.65x from
contracting both matrices with the raw gradient).

  (a) PRIMARY: ratio map sigma_Durante / sigma_cov220824 across (omega, i) at N=10.
      SUPPORTING: side-by-side Durante | cov_220824 heatmaps, plus per-perijove
      evaluations.
  (b) sigma_omega vs N (2..30) for a handful of real perijoves.

Run:    python make_sigma_omega_checks.py
Output: sigma_omega_iw_ratio.png, sigma_omega_iw_compare.png,
        sigma_omega_per_perijove.png, sigma_omega_vs_N_multi.png (+ printed tables)
"""

import matplotlib
matplotlib.use("Agg")

import os
import re

import numpy as np
import matplotlib.pyplot as plt

import constants as C
from covariance import covariance_slice
from precession_gravity import domega_dmu, domega_dJl, norm_factor
from precession_zonal_gauss import domega_dJl_gauss

COV_220824_PATH = os.path.join(os.path.dirname(__file__), "cov_220824_base_a4_n40.txt")


def load_cov_220824(path=COV_220824_PATH):
    """Return (labels, C) for the re-derived J2..J40 covariance (raw / 1-sigma)."""
    text = open(path, "r", encoding="utf-8").read()
    labels = re.findall(r"^J\d+$", text, flags=re.M)
    nums = [float(x) for x in re.findall(r"[-+]?\d+\.\d+e[+-]?\d+", text)]
    n = len(labels)
    expect = n * (n + 1) // 2
    if len(nums) != expect:
        raise ValueError(f"expected {expect} entries for {n} params, got {len(nums)}")
    Cmat = np.zeros((n, n))
    k = 0
    for i in range(n):
        for j in range(i + 1):
            Cmat[i, j] = nums[k]
            Cmat[j, i] = nums[k]
            k += 1
    return labels, Cmat

# ---------------------------------------------------------------------------
# Collaborator table: per-perijove orbital geometry (radians).
# columns: perijove number, omega_0, i
# ---------------------------------------------------------------------------
PERIJOVES = np.array([
    [1,  3.0759882091491426, 1.5683158629479454],
    [3,  3.0428139538933077, 1.5747733460232516],
    [4,  3.026319456277397,  1.5808821363407362],
    [5,  3.0097431356254103, 1.5890860057811793],
    [6,  2.9937246785613585, 1.599130267488382],
    [7,  2.976734316723912,  1.6106175623541377],
    [8,  2.960191199161792,  1.6223652888081281],
    [9,  2.9441223605426443, 1.6347711071028432],
    [10, 2.928559650327988,  1.6487698351773719],
    [11, 2.9120580844728035, 1.6644717823520983],
    [12, 2.896316074076392,  1.6795673857505526],
    [13, 2.880500820054267,  1.6945365996354282],
    [14, 2.8646552174133966, 1.707285105237907],
    [15, 2.849210724977876,  1.7193807691952456],
    [16, 2.8336097362104513, 1.7307233866541525],
    [17, 2.8202153433239188, 1.7409278555769871],
    [18, 2.8049879472668846, 1.7497065276274435],
    [19, 2.792068437552366,  1.757480277785906],
    [20, 2.779960627628028,  1.7629230070242463],
    [21, 2.7678168892346915, 1.7662371552985183],
    [22, 2.75622919665361,   1.7670952061346519],
    [23, 2.733205782686796,  1.8480285506559697],
    [24, 2.7251664782505656, 1.8463399229128181],
    [25, 2.7153586551677082, 1.8440079201258188],
    [26, 2.704897227902968,  1.840120080206654],
    [27, 2.6946123036681344, 1.8329926495528204],
    [28, 2.6851034481144036, 1.8246607622877988],
    [29, 2.674778902196235,  1.8156189285271667],
    [30, 2.6645645901461426, 1.8043319817711307],
    [31, 2.6536547494326173, 1.7856462412542422],
    [32, 2.642978411572513,  1.769710114835928],
    [33, 2.6316498283675083, 1.7481635606789745],
    [34, 2.634538544899344,  1.791068764588133],
    [35, 2.622482940540002,  1.7711165467393906],
    [36, 2.6096433256737814, 1.758977487499696],
    [37, 2.5975630697716445, 1.7469056418764448],
    [38, 2.5839796217797364, 1.7354303111105112],
    [39, 2.5702717389309155, 1.724789910104389],
    [40, 2.5566159668234913, 1.7157258233700765],
])


def _pj(max_pj):
    """(omega, i) rows of PERIJOVES with perijove number <= max_pj."""
    m = PERIJOVES[:, 0] <= max_pj
    return PERIJOVES[m, 1], PERIJOVES[m, 2], PERIJOVES[m, 0]


# ---------------------------------------------------------------------------
# Gradient d<Delta omega_g>/d[mu, J2..J_N] and sigma_omega
# ---------------------------------------------------------------------------
def gradient(inc, omega, n_max, apply_normalization=False, n_f=20000):
    """[d<dw>/dmu, d<dw>/dJ2, ..., d<dw>/dJ_{n_max}].

    Analytic Appendix-B derivatives for l = 2..10 (fast, exact); the Gauss-equation
    model for l >= 11 (valid to l = 30). apply_normalization multiplies each J_l
    derivative by sqrt(2l+1) for the fully-normalised (Durante) basis.
    """
    grad = [domega_dmu(C.A_KM, C.ECC)]
    for l in range(2, n_max + 1):
        if l <= 10:
            d = domega_dJl(l, C.A_KM, C.ECC, inc, omega, C.R_J_KM)
        else:
            d = domega_dJl_gauss(l, C.A_KM, C.ECC, inc, omega, C.R_J_KM, n_f=n_f)
        if apply_normalization:
            d *= norm_factor(l)
        grad.append(d)
    return np.array(grad)


def sigma_from_grad(grad, cov_block):
    return float(np.sqrt(grad @ cov_block @ grad))


def gradient_grid(omegas, incs, n_max, apply_normalization=False, n_f=20000):
    """J-only gradient at every (i, omega); shape (len(incs), len(omegas), n_max-1)."""
    G = np.empty((len(incs), len(omegas), n_max - 1))
    for a_idx, ii in enumerate(incs):
        for b_idx, om in enumerate(omegas):
            g = gradient(ii, om, n_max, apply_normalization=apply_normalization, n_f=n_f)
            G[a_idx, b_idx] = g[1:]  # drop GM
    return G


def sigma_grid(G, cov_block):
    """sigma_omega on the grid for a covariance block, from a precomputed gradient grid."""
    return np.sqrt(np.einsum("ijk,kl,ijl->ij", G, cov_block, G))


def cov_blocks(n_max, cov_new_full):
    """Basis-consistent J2..JN blocks: Durante (normalised), cov_220824 (raw)."""
    cov_dur = covariance_slice(n_max)[1:, 1:]
    cov_new = cov_new_full[: n_max - 1, : n_max - 1]
    return cov_dur, cov_new


# ---------------------------------------------------------------------------
# Plane computation (paper window by default)
# ---------------------------------------------------------------------------
def compute_plane(cov_new_full, n_max=10, n_grid=80,
                  omega_lim=(2.80, 3.10), i_lim=(1.55, 1.75), n_f=20000):
    """Return omegas, incs, sigma_Durante, sigma_cov220824 (basis-consistent)."""
    omegas = np.linspace(*omega_lim, n_grid)
    incs = np.linspace(*i_lim, n_grid)
    cov_dur, cov_new = cov_blocks(n_max, cov_new_full)
    G_norm = gradient_grid(omegas, incs, n_max, apply_normalization=True, n_f=n_f)
    G_raw = gradient_grid(omegas, incs, n_max, apply_normalization=False, n_f=n_f)
    S_dur = sigma_grid(G_norm, cov_dur)
    S_new = sigma_grid(G_raw, cov_new)
    return omegas, incs, S_dur, S_new


# Windows: the paper's Fig. 3-left window, and a wide window covering all PJ01-PJ40.
PAPER_WINDOW = dict(omega_lim=(2.80, 3.10), i_lim=(1.55, 1.75))
WIDE_WINDOW = dict(omega_lim=(2.54, 3.10), i_lim=(1.55, 1.86))


# ---------------------------------------------------------------------------
# Check (a) PRIMARY: ratio map sigma_Durante / sigma_cov220824
# ---------------------------------------------------------------------------
def plot_ratio(omegas, incs, S_dur, S_new, n_max=10, out="sigma_omega_iw_ratio.png"):
    ratio = S_dur / S_new
    fig, ax = plt.subplots(figsize=(7.2, 5.6))
    cf = ax.contourf(omegas, incs, ratio, levels=20, cmap="viridis")
    cs = ax.contour(omegas, incs, ratio, levels=8, colors="white",
                    linewidths=0.7, alpha=0.8)
    ax.clabel(cs, inline=True, fontsize=8, fmt="%.2f")
    cbar = fig.colorbar(cf, ax=ax)
    cbar.set_label(r"$\sigma_\omega^{\rm Durante}\,/\,\sigma_\omega^{\rm cov\_220824}$")

    ow, iw, num = _pj(37)
    ax.scatter(ow, iw, s=22, c="red", edgecolors="k", linewidths=0.4, zorder=5,
               label="Perijoves PJ01-PJ37")
    ax.scatter([PERIJOVES[0, 1]], [PERIJOVES[0, 2]], s=70, marker="*",
               c="gold", edgecolors="k", zorder=6, label="PJ01")

    ax.set_xlabel(r"$\omega$")
    ax.set_ylabel(r"$i$")
    ax.set_title(rf"Basis-consistent improvement factor"
                 rf"  ($N={n_max}$)")
    ax.legend(loc="upper right", framealpha=0.9, fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)
    print(f"saved {out}  (ratio range {ratio.min():.3f}-{ratio.max():.3f})")


# ---------------------------------------------------------------------------
# Check (a) SUPPORTING: two-panel Durante | cov_220824 heatmaps
# ---------------------------------------------------------------------------
def plot_compare(plane_paper, plane_wide, n_max=10,
                 out="sigma_omega_iw_compare.png"):
    """Left: Durante (normalised). Right: cov_220824 (raw). Independent colourbars."""
    om_p, i_p, S_dur_p, _ = plane_paper
    om_w, i_w, _, S_new_w = plane_wide
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.4))

    cf0 = axes[0].contourf(om_p, i_p, S_dur_p / 1e-9, levels=20, cmap="viridis")
    fig.colorbar(cf0, ax=axes[0]).set_label(r"$\sigma_\omega\ (\times 10^{-9})$")
    ow, iw, _ = _pj(17)
    axes[0].scatter(ow, iw, s=20, c="red", edgecolors="k", linewidths=0.4, zorder=5)
    axes[0].scatter([PERIJOVES[0, 1]], [PERIJOVES[0, 2]], s=70, marker="*",
                    c="gold", edgecolors="k", zorder=6)
    axes[0].set_title(rf"Durante 2020 (normalised basis)   ($N={n_max}$)")
    axes[0].set_xlabel(r"$\omega$")
    axes[0].set_ylabel(r"$i$")

    cf1 = axes[1].contourf(om_w, i_w, S_new_w / 1e-9, levels=20, cmap="viridis")
    fig.colorbar(cf1, ax=axes[1]).set_label(r"$\sigma_\omega\ (\times 10^{-9})$")
    ow, iw, _ = _pj(37)
    axes[1].scatter(ow, iw, s=20, c="red", edgecolors="k", linewidths=0.4, zorder=5)
    axes[1].scatter([PERIJOVES[0, 1]], [PERIJOVES[0, 2]], s=70, marker="*",
                    c="gold", edgecolors="k", zorder=6)
    axes[1].set_title(rf"cov_220824 (raw basis)   ($N={n_max}$)")
    axes[1].set_xlabel(r"$\omega$")
    axes[1].set_ylabel(r"$i$")

    fig.suptitle(r"Basis-consistent $\sigma_\omega$ in the $(\omega, i)$ plane")
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)
    print(f"saved {out}")


# ---------------------------------------------------------------------------
# Check (a) companion: sigma_omega at each perijove's own geometry
# ---------------------------------------------------------------------------
def plot_per_perijove(cov_new_full, n_max=10, out="sigma_omega_per_perijove.png"):
    cov_dur, cov_new = cov_blocks(n_max, cov_new_full)
    pjs = PERIJOVES[:, 0].astype(int)
    sig_dur, sig_new = [], []
    for _, om, ii in PERIJOVES:
        g_norm = gradient(ii, om, n_max, apply_normalization=True)[1:]
        g_raw = gradient(ii, om, n_max, apply_normalization=False)[1:]
        sig_dur.append(sigma_from_grad(g_norm, cov_dur))
        sig_new.append(sigma_from_grad(g_raw, cov_new))
    sig_dur = np.array(sig_dur)
    sig_new = np.array(sig_new)

    fig, ax = plt.subplots(figsize=(9.0, 5.0))
    ax.plot(pjs, sig_dur / 1e-9, "o-", color="#1f77b4", label="Durante (normalised)")
    ax.plot(pjs, sig_new / 1e-9, "s-", color="#d62728", label="cov_220824 (raw)")
    ax.set_xlabel("perijove number")
    ax.set_ylabel(r"$\sigma_\omega\ (\times 10^{-9})$")
    ax.set_title(rf"Basis-consistent $\sigma_\omega$ at each perijove   ($N={n_max}$)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)
    print(f"saved {out}")

    print(f"\n{'PJ':>3} {'omega':>8} {'i':>8} {'sig_Dur':>11} {'sig_new':>11} {'impr':>7}")
    for pj, (_, om, ii), sd, sn in zip(pjs, PERIJOVES, sig_dur, sig_new):
        print(f"{pj:>3} {om:>8.4f} {ii:>8.4f} {sd:>11.4e} {sn:>11.4e} {sd/sn:>7.3f}")


# ---------------------------------------------------------------------------
# Check (b): sigma_omega vs N for several real perijoves
# ---------------------------------------------------------------------------
def check_b(cov_new_full, pj_list=(1, 10, 22, 40), n_lo=2, n_hi=30, n_f=20000,
            out="sigma_omega_vs_N_multi.png"):
    Ns = np.arange(n_lo, n_hi + 1)
    lookup = {int(pj): (om, ii) for pj, om, ii in PERIJOVES}

    fig, ax = plt.subplots(figsize=(7.6, 5.6))
    colors = plt.cm.viridis(np.linspace(0.1, 0.85, len(pj_list)))
    print(f"\n{'check (b): basis-consistent sigma_omega vs N':=^70}")
    for pj, col in zip(pj_list, colors):
        om, ii = lookup[pj]
        sig_dur, sig_new = [], []
        for n in Ns:
            cov_dur, cov_new = cov_blocks(n, cov_new_full)
            g_norm = gradient(ii, om, n, apply_normalization=True, n_f=n_f)[1:]
            g_raw = gradient(ii, om, n, apply_normalization=False, n_f=n_f)[1:]
            sig_dur.append(sigma_from_grad(g_norm, cov_dur))
            sig_new.append(sigma_from_grad(g_raw, cov_new))
        sig_dur = np.array(sig_dur)
        sig_new = np.array(sig_new)
        ax.plot(Ns, sig_new / 1e-9, "-", color=col, label=f"PJ{pj:02d} cov_220824")
        ax.plot(Ns, sig_dur / 1e-9, "--", color=col, label=f"PJ{pj:02d} Durante")
        print(f"PJ{pj:02d} (i={ii:.4f}, omega={om:.4f}): "
              f"Durante {sig_dur[0]/1e-9:.3f}->{sig_dur[-1]/1e-9:.3f}, "
              f"cov_220824 {sig_new[0]/1e-9:.3f}->{sig_new[-1]/1e-9:.3f}  (x1e-9)")

    ax.set_xlabel(r"$N$")
    ax.set_ylabel(r"$\sigma_\omega\ (\times 10^{-9})$")
    ax.set_title(r"Basis-consistent $\sigma_\omega$ vs $N$ "
                 r"(solid: cov_220824, dashed: Durante)")
    ax.legend(ncol=2, fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)
    print(f"saved {out}")


def main():
    n_max = 10
    _, cov_new_full = load_cov_220824()
    print(f"{'Building basis-consistent (omega, i) planes at N=%d ...' % n_max:=^70}")
    plane_paper = compute_plane(cov_new_full, n_max=n_max, **PAPER_WINDOW)
    plane_wide = compute_plane(cov_new_full, n_max=n_max, **WIDE_WINDOW)
    om_w, i_w, S_dur_w, S_new_w = plane_wide
    _, _, S_dur_p, _ = plane_paper

    cov_dur, cov_new = cov_blocks(n_max, cov_new_full)
    g_pj1_norm = gradient(PERIJOVES[0, 2], PERIJOVES[0, 1], n_max,
                          apply_normalization=True)[1:]
    g_pj1_raw = gradient(PERIJOVES[0, 2], PERIJOVES[0, 1], n_max,
                         apply_normalization=False)[1:]
    sig_pj1_dur = sigma_from_grad(g_pj1_norm, cov_dur)
    sig_pj1_new = sigma_from_grad(g_pj1_raw, cov_new)
    print(f"Durante paper-window span : {S_dur_p.min():.3e} - {S_dur_p.max():.3e}")
    print(f"Durante at PJ01           : {sig_pj1_dur:.3e}   [plateau ~2.3e-9]")
    print(f"cov_220824 at PJ01        : {sig_pj1_new:.3e}   [plateau ~1.74e-9]")
    print(f"basis-consistent improvement at PJ01 = {sig_pj1_dur / sig_pj1_new:.2f}x")

    plot_ratio(om_w, i_w, S_dur_w, S_new_w, n_max=n_max)
    plot_compare(plane_paper, plane_wide, n_max=n_max)
    plot_per_perijove(cov_new_full, n_max=n_max)
    check_b(cov_new_full)


if __name__ == "__main__":
    main()
