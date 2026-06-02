# Juno fifth-force reproduction

Reproduce Singh et al., [JHEP01(2025)098](https://arxiv.org/abs/2409.10616): σ_ω from Durante gravity covariance (eq. 3.5) and fifth-force bounds (eqs. 2.3–2.5, Section 4).

## Setup

```bash
pip install -r requirements.txt
```

## σ_ω (Section 3)

```bash
python -m src.gravity_precession
```

Target: σ_ω ≈ 9×10⁻¹⁰ rad/orbit (N=10, i=1.57, ω=3.08).

## Figure 2 — ⟨Δω⟩ vs α and λ (Section 2)

```bash
python scripts/figure2_precession.py
python scripts/figure2_precession.py --show
python scripts/figure2_precession.py --quick
```

Writes `output/figure2_precession.png`. The right panel scans λ (slow); the left panel uses only two λ values.

## Figure 4 — α vs λ (Section 4)

```bash
python scripts/figure4_constraint.py
python scripts/figure4_constraint.py --show
python scripts/figure4_constraint.py --quick --n-lambda 30
```

Writes `output/figure4_constraint.png`. The scan is slow without `--quick` (each λ point evaluates eqs. 2.3–2.5).
