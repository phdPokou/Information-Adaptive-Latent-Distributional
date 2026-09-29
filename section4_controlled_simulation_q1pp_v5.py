#!/usr/bin/env python3
"""
Section 4 — Controlled Simulation Study
=======================================
Reproducible Q1++ implementation aligned with the locked Sections 2–3.

Core design principles
----------------------
1. S is an observed information state; Xi is future latent severity.
2. State dependence enters through q_{Phi(s)} and Gamma(s).
3. Gamma(s) is decision independent: it NEVER depends on x or C(x).
4. Baseline conditional kernel is Gaussian scale mixture; covariance-matched
   Student kernels are used only in robustness experiments.
5. The robust objective is the finite-dimensional KL variational criterion.
6. All paired comparisons use common random numbers and seed-level outputs.
7. Every figure/table has a CSV backing file; configuration is saved to JSON.
8. PASS/WARN/FAIL diagnostics are written to logs and CSV.

Recommended run
---------------
python section4_controlled_simulation_q1pp_v5.py --output Results_Section4_V5 --use-cuda

Fast smoke test
---------------
python section4_controlled_simulation_q1pp_v5.py --output Results_Section4_V5_Smoke --use-cuda --smoke-test
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import platform
import random
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------
# Global numerical policy
# -----------------------------------------------------------------------------
torch.set_default_dtype(torch.float64)
EPS = 1e-12


@dataclass
class Config:
    # Reproducibility
    seeds: List[int] = field(default_factory=lambda: [
        101, 202, 303, 404, 505, 606, 707, 808, 909, 1010,
        1111, 1212, 1313, 1414, 1515, 1616, 1717, 1818, 1919, 2020,
        2121, 2222, 2323, 2424, 2525, 2626, 2727, 2828, 2929, 3030,
    ])
    dtype: str = "float64"
    use_cuda: bool = True

    # Portfolio universe
    d: int = 12
    n_group: int = 4
    beta: float = 0.95
    weight_cap: float = 0.25
    exposure_green: float = 0.20
    exposure_neutral: float = 0.50
    exposure_brown: float = 0.80

    # State grid
    n_state: int = 21
    s_low: float = 0.10
    s_medium: float = 0.50
    s_high: float = 0.90

    # Latent lognormal q_{Phi(s)}: log Xi ~ N(m0 + a*s, (sigma0+b*s)^2)
    xi_m0: float = -0.18
    xi_a: float = 0.35
    xi_sigma0: float = 0.24
    xi_b: float = 0.10

    # State-dependent KL radius Gamma(s) = gamma0 + gamma1*s
    gamma0: float = 0.015
    gamma1: float = 0.035
    gamma_static: float = 0.0325

    # Synthetic return structure
    annual_mu: float = 0.060
    annual_vol: float = 0.180
    # Cross-group heterogeneity creates a genuine mean--tail-risk trade-off
    # without making the ambiguity radius decision dependent.
    group_mu_annual: Tuple[float, float, float] = (0.045, 0.060, 0.082)
    group_vol_annual: Tuple[float, float, float] = (0.135, 0.180, 0.245)
    corr_within: float = 0.35
    corr_cross: float = 0.18
    periods_per_year: int = 12

    # Fixed latent scenarios (CRN) used for robust objective
    n_latent: int = 4096
    # Out-of-sample true-return draws used for misspecification/robustness
    n_eval: int = 12000

    # Portfolio optimization
    x_steps: int = 280
    x_lr: float = 0.055
    x_tol: float = 2e-8
    x_patience: int = 45
    x_restarts: int = 2
    polish_steps: int = 100
    kkt_tol: float = 5e-6
    oracle_steps: int = 900
    oracle_restarts: int = 3

    # Inner alpha/rho optimization
    inner_steps: int = 180
    inner_lr: float = 0.055
    inner_tol: float = 2e-9
    inner_patience: int = 35
    rho_floor: float = 1e-5

    # Finite differences
    fd_h: float = 2e-3
    fd_h_grid: Tuple[float, ...] = (1e-2, 5e-3, 2e-3, 1e-3)

    # VoSC
    kappa_grid: List[float] = field(default_factory=lambda: [0.0, 0.25, 0.50, 0.75, 1.0])
    pi_balanced: List[float] = field(default_factory=lambda: [1/3, 1/3, 1/3])
    pi_low: List[float] = field(default_factory=lambda: [0.60, 0.30, 0.10])
    pi_high: List[float] = field(default_factory=lambda: [0.10, 0.30, 0.60])

    # Misspecification
    mean_shift: float = 0.018 / 12.0
    corr_shift_max: float = 0.28
    misspec_intensity: List[float] = field(default_factory=lambda: [0.0, 0.25, 0.50, 0.75, 1.0])

    # Kernel robustness
    student_nu: List[int] = field(default_factory=lambda: [15, 8, 5])

    # Signal contamination
    contamination_grid: List[float] = field(default_factory=lambda: [0.0, 0.15, 0.30, 0.45, 0.60, 0.75, 0.90, 1.0])

    # PASS thresholds (numerical diagnostics, not statistical claims)
    pass_grad_mae: float = 2.5e-3
    pass_policy_deriv_mae: float = 1.5e-2
    pass_policy_deriv_rel_mae: float = 0.10
    min_policy_deriv_signal: float = 1e-4
    pass_simplex_error: float = 1e-8
    pass_cap_error: float = 1e-8
    pass_solver_rate: float = 0.95
    pass_vosc_zero_tol: float = 1e-5
    pass_regret_tol: float = 2e-5

    # Plot/export
    dpi: int = 320


# -----------------------------------------------------------------------------
# Logging, IO, reproducibility
# -----------------------------------------------------------------------------
def setup_logger(out: Path) -> logging.Logger:
    logger = logging.getLogger("section4")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%Y-%m-%d %H:%M:%S")
    fh = logging.FileHandler(out / "diagnostics" / "section4_run.log", mode="w", encoding="utf-8")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def select_device(cfg: Config, force_cpu: bool = False) -> torch.device:
    if cfg.use_cuda and (not force_cpu) and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def save_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)

def output_dirs(out: Path) -> Dict[str, Path]:
    dirs = {
        "figures": out / "figures",
        "figure_csv": out / "figure_csv",
        "tables": out / "tables",
        "seed_level": out / "seed_level",
        "diagnostics": out / "diagnostics",
        "config": out / "config",
    }
    for d in dirs.values(): d.mkdir(parents=True, exist_ok=True)
    return dirs


def ci95(x: Sequence[float]) -> Tuple[float, float, float]:
    a = np.asarray(x, dtype=float)
    a = a[np.isfinite(a)]
    if len(a) == 0:
        return np.nan, np.nan, np.nan
    m = float(a.mean())
    if len(a) == 1:
        return m, m, m
    se = float(a.std(ddof=1) / math.sqrt(len(a)))
    z = 1.959963984540054
    return m, m - z * se, m + z * se


def summarize(df: pd.DataFrame, group: List[str], values: List[str]) -> pd.DataFrame:
    rows = []
    for key, g in df.groupby(group, dropna=False):
        if not isinstance(key, tuple):
            key = (key,)
        row = dict(zip(group, key))
        for v in values:
            m, lo, hi = ci95(g[v].values)
            row[f"{v}_mean"] = m
            row[f"{v}_ci_low"] = lo
            row[f"{v}_ci_high"] = hi
            vals_f = np.asarray(g[v].values, dtype=float); vals_f = vals_f[np.isfinite(vals_f)]
            row[f"{v}_sd"] = float(np.std(vals_f, ddof=1)) if len(vals_f) > 1 else (0.0 if len(vals_f)==1 else np.nan)
        row["n_seed"] = int(g["seed"].nunique()) if "seed" in g else len(g)
        rows.append(row)
    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# Synthetic universe and state maps
# -----------------------------------------------------------------------------
def build_universe(cfg: Config, device: torch.device) -> Dict[str, torch.Tensor]:
    d = cfg.d
    assert d == 3 * cfg.n_group, "d must equal 3*n_group in this design."
    c = torch.tensor([cfg.exposure_green] * cfg.n_group + [cfg.exposure_neutral] * cfg.n_group + [cfg.exposure_brown] * cfg.n_group, device=device)

    # Heterogeneous but climate-neutral investment opportunities. Brown-labelled
    # assets have a higher unconditional mean and volatility; green-labelled
    # assets have a lower mean and volatility. State dependence still enters ONLY
    # through q_{Phi(s)} and Gamma(s), exactly as in Sections 2--3.
    offsets = torch.tensor([-0.00018, -0.00006, 0.00006, 0.00018], device=device)
    mu = torch.cat([torch.full((cfg.n_group,), m/cfg.periods_per_year, device=device) + offsets for m in cfg.group_mu_annual])
    vols = torch.cat([torch.full((cfg.n_group,), v/math.sqrt(cfg.periods_per_year), device=device) for v in cfg.group_vol_annual])
    Sigma = torch.empty((d, d), device=device)
    for i in range(d):
        gi = i // cfg.n_group
        for j in range(d):
            gj = j // cfg.n_group
            corr = 1.0 if i == j else (cfg.corr_within if gi == gj else cfg.corr_cross)
            Sigma[i, j] = corr * vols[i] * vols[j]
    eigmin = torch.linalg.eigvalsh(Sigma).min().item()
    if eigmin <= 0:
        Sigma += (abs(eigmin) + 1e-10) * torch.eye(d, device=device)
    A = torch.linalg.cholesky(Sigma)
    return {"mu": mu, "Sigma": Sigma, "A": A, "c": c}


def phi_params(s: torch.Tensor, cfg: Config, kappa: float = 1.0, state_center: float = 0.5) -> Tuple[torch.Tensor, torch.Tensor]:
    # kappa=0 removes cross-state heterogeneity while preserving a common center.
    seff = state_center + kappa * (s - state_center)
    m = cfg.xi_m0 + cfg.xi_a * seff
    sig = cfg.xi_sigma0 + cfg.xi_b * seff
    return m, sig.clamp_min(0.03)


def gamma_state(s: torch.Tensor, cfg: Config, kappa: float = 1.0, static: bool = False) -> torch.Tensor:
    if static:
        return torch.as_tensor(cfg.gamma_static, dtype=s.dtype, device=s.device)
    seff = 0.5 + kappa * (s - 0.5)
    return (cfg.gamma0 + cfg.gamma1 * seff).clamp_min(0.0)


def gamma_prime(cfg: Config, kappa: float = 1.0, static: bool = False) -> float:
    return 0.0 if static else cfg.gamma1 * kappa


def fixed_latent_base(seed: int, n: int, device: torch.device) -> torch.Tensor:
    # Antithetic CRN substantially stabilizes state derivatives.
    g = torch.Generator(device="cpu")
    g.manual_seed(seed)
    n2 = (n + 1) // 2
    z = torch.randn(n2, generator=g, dtype=torch.float64)
    z = torch.cat([z, -z], dim=0)[:n]
    return z.to(device)


def xi_from_base(z: torch.Tensor, s: torch.Tensor, cfg: Config, kappa: float = 1.0) -> torch.Tensor:
    m, sig = phi_params(s, cfg, kappa=kappa)
    return torch.exp(m + sig * z)


# -----------------------------------------------------------------------------
# Feasible set and portfolio helpers
# -----------------------------------------------------------------------------
def project_capped_simplex(v: torch.Tensor, cap: float, total: float = 1.0, iters: int = 70) -> torch.Tensor:
    """Euclidean projection onto {x: sum x=total, 0<=x<=cap}."""
    if cap * v.numel() < total - 1e-12:
        raise ValueError("Infeasible cap: d*cap < 1.")
    lo = (v - cap).min().item() - total
    hi = v.max().item()
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        x = torch.clamp(v - mid, min=0.0, max=cap)
        if x.sum().item() > total:
            lo = mid
        else:
            hi = mid
    x = torch.clamp(v - 0.5 * (lo + hi), min=0.0, max=cap)
    # Tiny correction via another projection-like normalization; normally <1e-12.
    resid = total - x.sum()
    free = (x > 1e-12) & (x < cap - 1e-12)
    if free.any():
        x = x.clone()
        x[free] += resid / free.sum()
    return x


def feasible_error(x: torch.Tensor, cap: float) -> Tuple[float, float, float]:
    return (
        abs(float(x.sum().item()) - 1.0),
        max(0.0, -float(x.min().item())),
        max(0.0, float(x.max().item()) - cap),
    )


def exposure(x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    return c @ x


def turnover(x: torch.Tensor, x0: torch.Tensor) -> torch.Tensor:
    return torch.sum(torch.abs(x - x0))


def group_weights(x: torch.Tensor, cfg: Config) -> Tuple[float, float, float]:
    n = cfg.n_group
    return tuple(float(x[k*n:(k+1)*n].sum().item()) for k in range(3))


def reference_portfolios(cfg: Config, device: torch.device) -> Dict[str, torch.Tensor]:
    # Same budget; all interior to the cap. Exposure differs by group tilt.
    n = cfg.n_group
    def make(g, ntr, b):
        w = torch.cat([
            torch.full((n,), g/n, device=device),
            torch.full((n,), ntr/n, device=device),
            torch.full((n,), b/n, device=device),
        ])
        return w
    return {
        "Green": make(0.55, 0.30, 0.15),
        "Neutral": make(1/3, 1/3, 1/3),
        "Brown": make(0.15, 0.30, 0.55),
    }


# -----------------------------------------------------------------------------
# Robust tail-risk objective from Section 3.1
# -----------------------------------------------------------------------------
def normal_stop_loss(x: torch.Tensor, alpha: torch.Tensor, xi: torch.Tensor, mu: torch.Tensor, Sigma: torch.Tensor) -> torch.Tensor:
    m = -(x @ mu)
    var0 = x @ Sigma @ x
    sigma = torch.sqrt(torch.clamp(xi * var0, min=1e-18))
    z = (m - alpha) / sigma
    pdf = torch.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi)
    cdf = 0.5 * (1.0 + torch.erf(z / math.sqrt(2.0)))
    return sigma * pdf + (m - alpha) * cdf


def robust_objective_given_inner(
    x: torch.Tensor,
    s: torch.Tensor,
    z_latent: torch.Tensor,
    alpha: torch.Tensor,
    raw_rho: torch.Tensor,
    cfg: Config,
    mu: torch.Tensor,
    Sigma: torch.Tensor,
    kappa: float = 1.0,
    gamma_static_flag: bool = False,
    gamma_zero: bool = False,
) -> torch.Tensor:
    xi = xi_from_base(z_latent, s, cfg, kappa=kappa)
    h = normal_stop_loss(x, alpha, xi, mu, Sigma)
    if gamma_zero:
        return alpha + h.mean() / (1.0 - cfg.beta)
    rho = torch.nn.functional.softplus(raw_rho) + cfg.rho_floor
    gam = gamma_state(s, cfg, kappa=kappa, static=gamma_static_flag)
    # Stable log E exp(h/rho)
    lme = torch.logsumexp(h / rho, dim=0) - math.log(h.numel())
    return alpha + rho * (gam + lme) / (1.0 - cfg.beta)


def solve_inner(
    x: torch.Tensor,
    s: torch.Tensor,
    z_latent: torch.Tensor,
    cfg: Config,
    mu: torch.Tensor,
    Sigma: torch.Tensor,
    kappa: float = 1.0,
    gamma_static_flag: bool = False,
    gamma_zero: bool = False,
    create_graph: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, bool, int]:
    # Detach x only when no outer gradient is required.
    xx = x if create_graph else x.detach()
    # Sensible alpha initialization from mean loss; raw rho around 0.03.
    alpha = torch.tensor(float(-(xx @ mu).detach().item()), device=x.device, requires_grad=True)
    if gamma_zero:
        params = [alpha]
        raw_rho = torch.tensor(-3.0, device=x.device, requires_grad=False)
    else:
        raw_rho = torch.tensor(-3.0, device=x.device, requires_grad=True)
        params = [alpha, raw_rho]
    opt = torch.optim.Adam(params, lr=cfg.inner_lr)
    best = float("inf")
    best_state = None
    stale = 0
    converged = False
    for it in range(cfg.inner_steps):
        opt.zero_grad(set_to_none=True)
        val = robust_objective_given_inner(xx, s, z_latent, alpha, raw_rho, cfg, mu, Sigma,
                                           kappa=kappa, gamma_static_flag=gamma_static_flag,
                                           gamma_zero=gamma_zero)
        if not torch.isfinite(val):
            break
        val.backward(create_graph=False)
        torch.nn.utils.clip_grad_norm_(params, 50.0)
        opt.step()
        cur = float(val.detach().item())
        if cur < best - cfg.inner_tol:
            best = cur
            best_state = (alpha.detach().clone(), raw_rho.detach().clone())
            stale = 0
        else:
            stale += 1
        if stale >= cfg.inner_patience:
            converged = True
            break
    if best_state is None:
        best_state = (alpha.detach().clone(), raw_rho.detach().clone())
    a_star, rr_star = best_state
    # Deterministic LBFGS polish: derivative experiments require the variational
    # auxiliaries to satisfy stationarity much more tightly than Adam stagnation.
    aa = a_star.detach().clone().requires_grad_(True)
    rr = rr_star.detach().clone().requires_grad_(not gamma_zero)
    pars2 = [aa] if gamma_zero else [aa, rr]
    lb = torch.optim.LBFGS(pars2, lr=0.8, max_iter=max(30,cfg.polish_steps//2),
                           tolerance_grad=1e-11, tolerance_change=1e-14,
                           history_size=30, line_search_fn="strong_wolfe")
    def cl2():
        lb.zero_grad(set_to_none=True)
        vv = robust_objective_given_inner(xx, s, z_latent, aa, rr, cfg, mu, Sigma,
                                          kappa=kappa, gamma_static_flag=gamma_static_flag,
                                          gamma_zero=gamma_zero)
        vv.backward(); return vv
    try: lb.step(cl2)
    except RuntimeError: pass
    a_star, rr_star = aa.detach(), rr.detach()
    # Re-evaluate with fixed minimizers. Envelope derivatives in x,s can be
    # obtained by differentiating this objective while holding inner optima fixed.
    val_star = robust_objective_given_inner(x, s, z_latent, a_star, rr_star, cfg, mu, Sigma,
                                            kappa=kappa, gamma_static_flag=gamma_static_flag,
                                            gamma_zero=gamma_zero)
    rho_star = torch.nn.functional.softplus(rr_star) + cfg.rho_floor
    return val_star, a_star, rho_star, converged, it + 1


def risk_value(x, s_float, z_latent, cfg, mu, Sigma, **kwargs) -> Tuple[float, float, float, bool, int]:
    s = torch.tensor(float(s_float), device=x.device)
    val, a, rho, conv, nit = solve_inner(x, s, z_latent, cfg, mu, Sigma, **kwargs)
    return float(val.detach().item()), float(a.item()), float(rho.item()), conv, nit


def _rawrho_from_rho(rho: torch.Tensor, floor: float) -> torch.Tensor:
    y = torch.clamp(rho - floor, min=1e-12)
    return torch.where(y > 20.0, y, torch.log(torch.expm1(y)))


def polish_joint_interior(x0, s, z_latent, cfg, mu, Sigma, kappa=1.0,
                          gamma_static_flag=False, gamma_zero=False,
                          alpha0=None, rho0=None):
    """High-accuracy reduced-space LBFGS polish for an interior capped-simplex solution."""
    B = nullspace_budget(cfg.d, x0.device)
    zc = torch.zeros(B.shape[1], device=x0.device, requires_grad=True)
    if alpha0 is None or (rho0 is None and not gamma_zero):
        _, aa0, rrho0, _, _ = solve_inner(x0.detach(), s.detach(), z_latent, cfg, mu, Sigma,
                                           kappa=kappa, gamma_static_flag=gamma_static_flag,
                                           gamma_zero=gamma_zero)
    else:
        aa0 = torch.as_tensor(alpha0, dtype=x0.dtype, device=x0.device)
        rrho0 = torch.as_tensor(0.03 if rho0 is None else rho0, dtype=x0.dtype, device=x0.device)
    aa = aa0.detach().clone().requires_grad_(True)
    if gamma_zero:
        rr = torch.tensor(-3.0, dtype=x0.dtype, device=x0.device)
        pars = [zc, aa]
    else:
        rr = _rawrho_from_rho(rrho0.detach(), cfg.rho_floor).clone().requires_grad_(True)
        pars = [zc, aa, rr]
    opt = torch.optim.LBFGS(pars, lr=0.8, max_iter=cfg.polish_steps,
                            tolerance_grad=1e-10, tolerance_change=1e-13,
                            history_size=50, line_search_fn="strong_wolfe")
    def closure():
        opt.zero_grad(set_to_none=True)
        xx = x0.detach() + B @ zc
        val = robust_objective_given_inner(xx, s, z_latent, aa, rr, cfg, mu, Sigma,
                                           kappa=kappa, gamma_static_flag=gamma_static_flag,
                                           gamma_zero=gamma_zero)
        val.backward()
        return val
    try:
        opt.step(closure)
    except RuntimeError:
        pass
    with torch.no_grad():
        xx = x0.detach() + B @ zc.detach()
        # Only accept the unconstrained reduced-space polish if box constraints remain feasible.
        if xx.min().item() < -1e-9 or xx.max().item() > cfg.weight_cap + 1e-9:
            return x0.detach().clone(), aa0.detach().clone(), rrho0.detach().clone(), float("inf")
        xx = project_capped_simplex(xx, cfg.weight_cap)
    aa_f = aa.detach()
    rho_f = torch.tensor(0.0, device=x0.device) if gamma_zero else (torch.nn.functional.softplus(rr.detach()) + cfg.rho_floor)
    # Reduced joint KKT residual.
    zz = torch.zeros(B.shape[1], device=x0.device, requires_grad=True)
    aaa = aa_f.clone().requires_grad_(True)
    if gamma_zero:
        rrr = torch.tensor(-3.0, device=x0.device)
        yy = [zz, aaa]
    else:
        rrr = _rawrho_from_rho(rho_f, cfg.rho_floor).clone().requires_grad_(True)
        yy = [zz, aaa, rrr]
    vv = robust_objective_given_inner(xx.detach()+B@zz, s, z_latent, aaa, rrr, cfg, mu, Sigma,
                                      kappa=kappa, gamma_static_flag=gamma_static_flag,
                                      gamma_zero=gamma_zero)
    gg = torch.autograd.grad(vv, yy)
    kres = math.sqrt(sum(float((g.detach()**2).sum().item()) for g in gg))
    return xx.detach(), aa_f, rho_f.detach(), kres


def optimize_portfolio(
    s_float: float,
    z_latent: torch.Tensor,
    cfg: Config,
    mu: torch.Tensor,
    Sigma: torch.Tensor,
    x_init: Optional[torch.Tensor] = None,
    kappa: float = 1.0,
    gamma_static_flag: bool = False,
    gamma_zero: bool = False,
) -> Tuple[torch.Tensor, float, Dict[str, float]]:
    """Jointly minimize over x and the variational auxiliaries.

    This is mathematically equivalent to min_x inf_{alpha,rho} Psi(x,alpha,rho;s)
    and is dramatically faster than a nested inner solve at every outer step.
    Projection enforces the capped simplex after every Adam step.
    """
    device = mu.device
    starts = []
    eq = torch.full((cfg.d,), 1.0/cfg.d, device=device)
    starts.append(eq if x_init is None else x_init.detach().clone())
    if cfg.x_restarts > 1:
        starts.append(project_capped_simplex(starts[0] + torch.linspace(-0.01,0.01,cfg.d,device=device), cfg.weight_cap))

    best_x, best_val, best_diag = None, float("inf"), None
    s = torch.tensor(float(s_float), device=device)
    for r, start in enumerate(starts[:cfg.x_restarts]):
        x = start.clone().detach().requires_grad_(True)
        alpha = torch.tensor(float(-(start @ mu).item()), device=device, requires_grad=True)
        raw_rho = torch.tensor(-3.0, device=device, requires_grad=not gamma_zero)
        params = [x, alpha] + ([] if gamma_zero else [raw_rho])
        opt = torch.optim.Adam(params, lr=cfg.x_lr)
        stale, best_local, best_state = 0, float("inf"), None
        for it in range(cfg.x_steps):
            opt.zero_grad(set_to_none=True)
            val = robust_objective_given_inner(x, s, z_latent, alpha, raw_rho, cfg, mu, Sigma,
                                               kappa=kappa, gamma_static_flag=gamma_static_flag,
                                               gamma_zero=gamma_zero)
            if not torch.isfinite(val): break
            val.backward()
            torch.nn.utils.clip_grad_norm_(params, 20.0)
            opt.step()
            with torch.no_grad(): x.copy_(project_capped_simplex(x, cfg.weight_cap))
            cur=float(val.detach().item())
            if cur < best_local-cfg.x_tol:
                best_local=cur; stale=0
                best_state=(x.detach().clone(),alpha.detach().clone(),raw_rho.detach().clone())
            else: stale+=1
            if stale>=cfg.x_patience: break
        if best_state is None: best_state=(x.detach(),alpha.detach(),raw_rho.detach())
        bx,ba,brr=best_state
        rho_adam = torch.tensor(0.0,device=device) if gamma_zero else (torch.nn.functional.softplus(brr)+cfg.rho_floor)
        bx_p,ba_p,rho_p,kkt = polish_joint_interior(bx,s,z_latent,cfg,mu,Sigma,kappa=kappa,
                                                    gamma_static_flag=gamma_static_flag,gamma_zero=gamma_zero,
                                                    alpha0=ba,rho0=rho_adam)
        if math.isfinite(kkt):
            bx,ba=bx_p,ba_p
            brr=torch.tensor(-3.0,device=device) if gamma_zero else _rawrho_from_rho(rho_p,cfg.rho_floor)
        final=float(robust_objective_given_inner(bx,s,z_latent,ba,brr,cfg,mu,Sigma,kappa=kappa,
                                                 gamma_static_flag=gamma_static_flag,gamma_zero=gamma_zero).item())
        se,ne,ce=feasible_error(bx,cfg.weight_cap)
        diag={"restart":r,"outer_iterations":it+1,"inner_iterations_total":0,
              "inner_final_converged":1.0,"outer_converged":float(math.isfinite(final) and se<=cfg.pass_simplex_error and ce<=cfg.pass_cap_error and kkt<=cfg.kkt_tol),
              "simplex_error":se,"negative_error":ne,"cap_error":ce,"kkt_residual":kkt,
              "alpha_star":float(ba.item()),"rho_star":float((torch.nn.functional.softplus(brr)+cfg.rho_floor).item()) if not gamma_zero else 0.0}
        if final<best_val: best_x,best_val,best_diag=bx.clone(),final,diag
    return best_x,best_val,best_diag


# -----------------------------------------------------------------------------
# Derivatives for Experiments I–II
# -----------------------------------------------------------------------------
def risk_state_derivative(
    x: torch.Tensor, s_float: float, z_latent: torch.Tensor, cfg: Config,
    mu: torch.Tensor, Sigma: torch.Tensor, kappa: float = 1.0,
    gamma_static_flag: bool = False,
) -> Dict[str, float]:
    s = torch.tensor(float(s_float), device=x.device, requires_grad=True)
    val, alpha, rho, _, _ = solve_inner(x, s, z_latent, cfg, mu, Sigma,
                                        kappa=kappa, gamma_static_flag=gamma_static_flag)
    # Total envelope derivative via autograd at fixed inner minimizers.
    total = torch.autograd.grad(val, s, retain_graph=False)[0]
    # Radius channel is exact from theorem.
    g_gamma = float(rho.item()) * gamma_prime(cfg, kappa=kappa, static=gamma_static_flag) / (1.0-cfg.beta)
    g_total = float(total.detach().item())
    g_phi = g_total - g_gamma
    fd_vals=[]
    for h in cfg.fd_h_grid:
        lo, *_ = risk_value(x, max(0.0, s_float-h), z_latent, cfg, mu, Sigma, kappa=kappa, gamma_static_flag=gamma_static_flag)
        hi, *_ = risk_value(x, min(1.0, s_float+h), z_latent, cfg, mu, Sigma, kappa=kappa, gamma_static_flag=gamma_static_flag)
        denom = min(1.0, s_float+h) - max(0.0, s_float-h)
        fd_vals.append((hi-lo)/denom)
    g_fd=float(np.median(fd_vals))
    return {
        "risk": float(val.detach().item()), "alpha_star": float(alpha.item()), "rho_star": float(rho.item()),
        "g_gamma": g_gamma, "g_phi": g_phi, "g_total": g_total, "g_fd": g_fd,
        "fd_stability_range": float(max(fd_vals)-min(fd_vals)),
        "grad_abs_error": abs(g_fd-g_total),
    }


def nullspace_budget(d: int, device: torch.device) -> torch.Tensor:
    # Orthonormal basis of {v: 1'v=0} via QR.
    one = torch.ones((d,1), device=device) / math.sqrt(d)
    M = torch.cat([one, torch.eye(d, device=device)], dim=1)
    Q, _ = torch.linalg.qr(M)
    # Q[:,0] spans ones; remaining d-1 columns span the nullspace.
    return Q[:,1:d]


def local_policy_derivative(
    xstar: torch.Tensor, s_float: float, z_latent: torch.Tensor, cfg: Config,
    mu: torch.Tensor, Sigma: torch.Tensor,
) -> Tuple[torch.Tensor, float, float, bool]:
    """Implicit derivative from the full reduced KKT/FOC system.

    We differentiate the joint stationarity conditions in the budget-nullspace
    coordinates and in the variational auxiliaries (alpha, raw_rho).  This is
    algebraically equivalent to profiling by a Schur complement, but is much
    better conditioned numerically and avoids subtracting nearly singular
    second-order blocks.  The calculation is used only when box constraints
    are locally inactive, exactly matching the local theorem.
    """
    active = bool(((xstar < 2e-4) | (xstar > cfg.weight_cap-2e-4)).any().item())
    B = nullspace_budget(cfg.d, xstar.device)
    s0 = torch.tensor(float(s_float), device=xstar.device)
    _, a0, rho0, _, _ = solve_inner(xstar, s0, z_latent, cfg, mu, Sigma)
    rr0 = torch.log(torch.expm1(torch.clamp(rho0 - cfg.rho_floor, min=1e-10)))
    # y=(z,alpha,raw_rho), with x=xstar+Bz and z=0 at the evaluation point.
    y0 = torch.cat([torch.zeros(B.shape[1], device=xstar.device),
                    a0.reshape(1), rr0.reshape(1)]).requires_grad_(True)
    ss0 = s0.detach().clone().requires_grad_(True)

    def f_y_s(y, ss):
        zc=y[:B.shape[1]]; aa=y[-2]; rr=y[-1]
        xx=xstar.detach()+B@zc
        return robust_objective_given_inner(xx,ss,z_latent,aa,rr,cfg,mu,Sigma)

    def grad_y(y, ss):
        val=f_y_s(y,ss)
        return torch.autograd.grad(val,y,create_graph=True)[0]

    try:
        Hyy=torch.autograd.functional.jacobian(lambda yy: grad_y(yy,ss0),y0,vectorize=True)
        Hys=torch.autograd.functional.jacobian(lambda st: grad_y(y0,st),ss0,vectorize=True)
        Hyy=0.5*(Hyy+Hyy.T)
        # The reduced portfolio block supplies the theorem's curvature audit.
        Hred=Hyy[:B.shape[1],:B.shape[1]]
        eigmin=float(torch.linalg.eigvalsh(Hred).min().item())
        # Mild diagonal regularization is numerical only and scale adaptive.
        scale=max(float(torch.linalg.norm(Hyy,ord=2).item()),1.0)
        reg=1e-10*scale
        dy=-torch.linalg.solve(Hyy+reg*torch.eye(Hyy.shape[0],device=Hyy.device),Hys)
        dx=B@dy[:B.shape[1]]
        ok=(eigmin>1e-9) and (not active) and bool(torch.all(torch.isfinite(dx)).item())
        station=float(torch.linalg.norm(grad_y(y0,ss0).detach()).item())
    except RuntimeError:
        dx=torch.full_like(xstar,float("nan")); eigmin=float("nan"); station=float("nan"); ok=False
    return dx.detach(), station, eigmin, ok

# -----------------------------------------------------------------------------
# True DGP sampling and empirical CVaR
# -----------------------------------------------------------------------------
def make_corr_shifted_sigma(base: torch.Tensor, cfg: Config, intensity: float) -> torch.Tensor:
    d = cfg.d
    vols = torch.sqrt(torch.diag(base))
    corr = base / (vols[:,None]*vols[None,:])
    target = corr.clone()
    off = ~torch.eye(d, dtype=torch.bool, device=base.device)
    target[off] = torch.clamp(target[off] + intensity*cfg.corr_shift_max, max=0.88)
    target.fill_diagonal_(1.0)
    # Ensure PSD by eigenvalue clipping.
    ev, Q = torch.linalg.eigh(target)
    target = Q @ torch.diag(ev.clamp_min(1e-5)) @ Q.T
    D = torch.diag(vols)
    return D @ target @ D


def sample_returns(
    seed: int, n: int, s: float, cfg: Config, mu: torch.Tensor, Sigma: torch.Tensor,
    z_latent: torch.Tensor, dgp: str = "radial", intensity: float = 1.0,
    student_nu: Optional[int] = None,
) -> torch.Tensor:
    device = mu.device
    # CRN: same Gaussian base for all DGPs at same seed.
    g = torch.Generator(device="cpu"); g.manual_seed(seed + 700_000)
    Z = torch.randn((n, cfg.d), generator=g, dtype=torch.float64).to(device)
    s_t = torch.tensor(float(s), device=device)
    xi = xi_from_base(z_latent[:n], s_t, cfg)
    mu_true = mu.clone()
    Sig_true = Sigma.clone()
    if dgp in ("mean", "combined"):
        # Mean repricing is exposure-linked in the DGP only; ambiguity radius remains decision independent.
        c = build_universe(cfg, device)["c"]
        mu_true = mu_true - intensity * cfg.mean_shift * s * c
    if dgp in ("correlation", "combined"):
        Sig_true = make_corr_shifted_sigma(Sigma, cfg, intensity*s)
    A = torch.linalg.cholesky(Sig_true)
    if student_nu is None:
        U = Z
    else:
        # covariance-matched multivariate t: sqrt((nu-2)/chi2_nu) * Z
        rng = np.random.default_rng(seed + 710_000 + int(student_nu))
        chi = torch.as_tensor(rng.chisquare(float(student_nu), size=n), dtype=torch.float64, device=device)
        U = Z * torch.sqrt((student_nu-2.0)/chi).unsqueeze(1)
    return mu_true + torch.sqrt(xi).unsqueeze(1) * (U @ A.T)


def empirical_cvar(loss: torch.Tensor, beta: float) -> float:
    q = torch.quantile(loss, beta)
    tail = loss[loss >= q]
    return float(tail.mean().item()) if tail.numel() else float(q.item())


def empirical_var(loss: torch.Tensor, beta: float) -> float:
    return float(torch.quantile(loss, beta).item())


# -----------------------------------------------------------------------------
# Experiments
# -----------------------------------------------------------------------------
def exp1_mechanism(cfg, device, uni, out, logger):
    logger.info("[EXP 1/6] Mechanism recovery: nominal-law and ambiguity-radius channels")
    rows = []
    states = np.linspace(0.05, 0.95, cfg.n_state)
    xref = reference_portfolios(cfg, device)["Neutral"]
    designs = [
        ("D0", 0.0, True),
        ("D_phi", 1.0, True),
        ("D_gamma", 0.0, False),
        ("D_phi+gamma", 1.0, False),
    ]
    for seed in cfg.seeds:
        z = fixed_latent_base(seed, cfg.n_latent, device)
        for name, kappa, gstatic in designs:
            for s in states:
                r = risk_state_derivative(xref, float(s), z, cfg, uni["mu"], uni["Sigma"],
                                          kappa=kappa, gamma_static_flag=gstatic)
                rows.append({"seed":seed,"design":name,"state":s,**r})
    df = pd.DataFrame(rows); save_csv(df, out/"seed_level"/"seed_level_risk_decomposition.csv")
    figdf = summarize(df, ["design","state"], ["g_gamma","g_phi","g_total","g_fd","fd_stability_range","grad_abs_error"])
    save_csv(figdf, out/"figure_csv"/"figure_1_data.csv")
    # Plot combined design for interpretability + all-design validation scatter.
    comb = figdf[figdf.design=="D_phi+gamma"]
    fig, ax = plt.subplots(1,2,figsize=(10.8,4.1))
    for v,lab in [("g_gamma","Radius channel"),("g_phi","Nominal-law channel"),("g_total","Total")]:
        ax[0].plot(comb.state, comb[f"{v}_mean"], label=lab)
        ax[0].fill_between(comb.state, comb[f"{v}_ci_low"], comb[f"{v}_ci_high"], alpha=.12)
    ax[0].axhline(0,lw=.8); ax[0].set_xlabel("Information state $s$"); ax[0].set_ylabel("State derivative")
    ax[0].set_title("(a) Two-channel decomposition"); ax[0].legend(frameon=False,fontsize=8)
    scat = df.sample(min(len(df), 4000), random_state=1)
    ax[1].scatter(scat.g_total, scat.g_fd, s=8, alpha=.35)
    mn=min(scat.g_total.min(),scat.g_fd.min()); mx=max(scat.g_total.max(),scat.g_fd.max())
    ax[1].plot([mn,mx],[mn,mx],lw=1); ax[1].set_xlabel("Analytical decomposition"); ax[1].set_ylabel("Finite difference")
    ax[1].set_title("(b) Derivative recovery")
    fig.tight_layout(); fig.savefig(out/"figures"/"figure_1_state_sensitivity.png",dpi=cfg.dpi,bbox_inches="tight"); plt.close(fig)
    return df


def exp2_policy(cfg, device, uni, out, logger):
    logger.info("[EXP 2/6] Optimal-policy and transition-exposure comparative statics")
    rows=[]; diag=[]
    states=np.linspace(0.08,0.92,max(11,cfg.n_state//2+1))
    B=nullspace_budget(cfg.d,device)
    for seed in cfg.seeds:
        z=fixed_latent_base(seed,cfg.n_latent,device)
        prev=None
        cache={}
        for s in states:
            x,val,d=optimize_portfolio(float(s),z,cfg,uni["mu"],uni["Sigma"],x_init=prev)
            prev=x; cache[float(s)]=(x,val)
            wg,wn,wb=group_weights(x,cfg)
            rows.append({"seed":seed,"state":s,"risk":val,"exposure":float(exposure(x,uni["c"]).item()),
                         "w_green":wg,"w_neutral":wn,"w_brown":wb, **{f"x{i+1}":float(x[i].item()) for i in range(cfg.d)}})
            diag.append({"experiment":"policy","seed":seed,"state":s,**d})
        # Validate Theorem 3.4 locally. Analytical sensitivity uses the profiled
        # (Schur-complement) Hessian; the benchmark re-solves the portfolio at s±h.
        for j in range(1,len(states)-1):
            s=float(states[j]); x,val=cache[s]
            h=max(cfg.fd_h, 2e-3)
            sm=max(0.0,s-h); sp=min(1.0,s+h)
            xm,_,_=optimize_portfolio(sm,z,cfg,uni["mu"],uni["Sigma"],x_init=x)
            xp,_,_=optimize_portfolio(sp,z,cfg,uni["mu"],uni["Sigma"],x_init=x)
            dx_fd=(xp-xm)/(sp-sm)
            dx_an,_,eigmin,ok=local_policy_derivative(x,s,z,cfg,uni["mu"],uni["Sigma"])
            active=bool(((x<2e-4)|(x>cfg.weight_cap-2e-4)).any().item())
            if ok:
                dc_an=float((uni["c"]@dx_an).item()); dc_fd=float((uni["c"]@dx_fd).item())
                mae=float(torch.mean(torch.abs(dx_an-dx_fd)).item())
            else:
                dc_an=dc_fd=mae=np.nan
            idx=len(rows)-len(states)+j
            rows[idx].update({"dc_analytical":dc_an,"dc_fd":dc_fd,"policy_deriv_mae":mae,
                              "hred_eigmin":eigmin,"inequality_active":int(active),"implicit_ok":int(ok)})
    df=pd.DataFrame(rows); save_csv(df,out/"seed_level"/"seed_level_policy_sensitivity.csv")
    save_csv(pd.DataFrame(diag),out/"diagnostics"/"optimization_diagnostics_policy.csv")
    vals=["risk","exposure","w_green","w_neutral","w_brown","dc_analytical","dc_fd","policy_deriv_mae"]
    figdf=summarize(df,["state"],vals); save_csv(figdf,out/"figure_csv"/"figure_2_data.csv")
    fig,ax=plt.subplots(1,2,figsize=(10.8,4.1))
    ax[0].plot(figdf.state,figdf.exposure_mean,label="$C(x^*(s))$")
    ax[0].fill_between(figdf.state,figdf.exposure_ci_low,figdf.exposure_ci_high,alpha=.15)
    ax[0].plot(figdf.state,figdf.w_green_mean,ls="--",label="Green weight")
    ax[0].plot(figdf.state,figdf.w_brown_mean,ls="--",label="Brown weight")
    ax[0].set_xlabel("Information state $s$"); ax[0].set_title("(a) Optimal allocation and exposure"); ax[0].legend(frameon=False,fontsize=8)
    q=df.dropna(subset=["dc_analytical","dc_fd"])
    ax[1].scatter(q.dc_analytical,q.dc_fd,s=12,alpha=.45)
    if len(q):
        mn=min(q.dc_analytical.min(),q.dc_fd.min()); mx=max(q.dc_analytical.max(),q.dc_fd.max()); ax[1].plot([mn,mx],[mn,mx],lw=1)
    ax[1].set_xlabel("Analytical $D_s C$"); ax[1].set_ylabel("Finite-difference $D_s C$"); ax[1].set_title("(b) Local comparative-static validation")
    fig.tight_layout(); fig.savefig(out/"figures"/"figure_2_policy_sensitivity.png",dpi=cfg.dpi,bbox_inches="tight"); plt.close(fig)
    return df,pd.DataFrame(diag)


def exp3_heterogeneity(cfg,device,uni,out,logger):
    logger.info("[EXP 3/6] Heterogeneous Green/Neutral/Brown portfolio responses")
    refs=reference_portfolios(cfg,device); rows=[]
    states=[cfg.s_low,cfg.s_medium,cfg.s_high]
    for seed in cfg.seeds:
        z=fixed_latent_base(seed,cfg.n_latent,device)
        for name,x0 in refs.items():
            base_r,*_=risk_value(x0,cfg.s_low,z,cfg,uni["mu"],uni["Sigma"])
            c0=float(exposure(x0,uni["c"]).item())
            for s in states:
                x,val,_=optimize_portfolio(s,z,cfg,uni["mu"],uni["Sigma"],x_init=x0)
                rows.append({"seed":seed,"portfolio":name,"state":s,"c_ref":c0,"risk_ref_low":base_r,
                             "risk_opt":val,"delta_risk":val-base_r,"c_opt":float(exposure(x,uni["c"]).item()),
                             "delta_c":float(exposure(x,uni["c"]).item())-c0,
                             "delta_x_l1":float(torch.sum(torch.abs(x-x0)).item()),
                             "turnover":float(turnover(x,x0).item())})
    df=pd.DataFrame(rows); save_csv(df,out/"seed_level"/"seed_level_portfolio_heterogeneity.csv")
    figdf=summarize(df,["portfolio","state"],["delta_risk","delta_c","delta_x_l1","turnover","risk_opt","c_opt"])
    save_csv(figdf,out/"figure_csv"/"figure_3_data.csv")
    high=df[np.isclose(df.state,cfg.s_high)]
    tab=summarize(high,["portfolio"],["c_ref","risk_ref_low","risk_opt","delta_c","turnover","delta_x_l1"])
    save_csv(tab,out/"tables"/"table_2_heterogeneity.csv")
    fig,ax=plt.subplots(figsize=(6.3,4.7))
    for name,g in figdf.groupby("portfolio"):
        ax.plot(g.delta_c_mean,g.delta_risk_mean,marker="o",label=name)
        for _,r in g.iterrows(): ax.annotate(f"s={r.state:.1f}",(r.delta_c_mean,r.delta_risk_mean),fontsize=7,xytext=(3,3),textcoords="offset points")
    ax.axhline(0,lw=.7); ax.axvline(0,lw=.7); ax.set_xlabel("Change in transition exposure $\\Delta C$"); ax.set_ylabel("Change in robust tail risk $\\Delta \\mathcal{R}$")
    ax.set_title("State-dependent heterogeneous portfolio adjustments"); ax.legend(frameon=False); fig.tight_layout(); fig.savefig(out/"figures"/"figure_3_heterogeneity.png",dpi=cfg.dpi,bbox_inches="tight"); plt.close(fig)
    return df


def optimize_static_across_states(states, probs, z, cfg, mu, Sigma, model, kappa):
    """Joint static allocation across states with state-specific variational auxiliaries."""
    device=mu.device
    x=torch.full((cfg.d,),1/cfg.d,device=device,requires_grad=True)
    alphas=torch.zeros(len(states),device=device,requires_grad=True)
    rawrhos=torch.full((len(states),),-3.0,device=device,requires_grad=True)
    params=[x,alphas] if model=="M0" else [x,alphas,rawrhos]
    opt=torch.optim.Adam(params,lr=cfg.x_lr)
    stale=0; best=float("inf"); bestx=x.detach().clone()
    for it in range(cfg.x_steps):
        opt.zero_grad(set_to_none=True); total=torch.tensor(0.0,device=device)
        for j,(s,p) in enumerate(zip(states,probs)):
            gz=(model=="M0"); gs=(model in ("M1","M2")); kap=0.0 if model in ("M0","M1") else kappa
            total=total+float(p)*robust_objective_given_inner(x,torch.tensor(float(s),device=device),z,alphas[j],rawrhos[j],cfg,mu,Sigma,
                                                              kappa=kap,gamma_static_flag=gs,gamma_zero=gz)
        total.backward(); torch.nn.utils.clip_grad_norm_(params,20.0); opt.step()
        with torch.no_grad(): x.copy_(project_capped_simplex(x,cfg.weight_cap))
        cur=float(total.detach().item())
        if cur<best-cfg.x_tol: best=cur; stale=0; bestx=x.detach().clone()
        else: stale+=1
        if stale>=cfg.x_patience: break
    # High-accuracy joint reduced-space polish for the common static decision.
    B=nullspace_budget(cfg.d,device); zc=torch.zeros(B.shape[1],device=device,requires_grad=True)
    aa=alphas.detach().clone().requires_grad_(True); rr=rawrhos.detach().clone().requires_grad_(model!="M0")
    pars=[zc,aa] if model=="M0" else [zc,aa,rr]
    lb=torch.optim.LBFGS(pars,lr=0.8,max_iter=cfg.polish_steps,tolerance_grad=1e-10,tolerance_change=1e-13,history_size=50,line_search_fn="strong_wolfe")
    def cls():
        lb.zero_grad(set_to_none=True); xx=bestx.detach()+B@zc; tot=torch.tensor(0.0,device=device)
        for jj,(ss,pp) in enumerate(zip(states,probs)):
            gz=(model=="M0"); gs=(model in ("M1","M2")); kap=0.0 if model in ("M0","M1") else kappa
            tot=tot+float(pp)*robust_objective_given_inner(xx,torch.tensor(float(ss),device=device),z,aa[jj],rr[jj],cfg,mu,Sigma,kappa=kap,gamma_static_flag=gs,gamma_zero=gz)
        tot.backward(); return tot
    try: lb.step(cls)
    except RuntimeError: pass
    cand=bestx.detach()+B@zc.detach()
    if cand.min().item()>=-1e-9 and cand.max().item()<=cfg.weight_cap+1e-9:
        bestx=project_capped_simplex(cand,cfg.weight_cap)
    vals=[]
    for s,p in zip(states,probs):
        gz=(model=="M0"); gs=(model in ("M1","M2")); kap=0.0 if model in ("M0","M1") else kappa
        v,*_=risk_value(bestx,float(s),z,cfg,mu,Sigma,kappa=kap,gamma_static_flag=gs,gamma_zero=gz); vals.append(v)
    return bestx,float(np.dot(probs,vals))


def exp4_vosc(cfg,device,uni,out,logger):
    logger.info("[EXP 4/6] Value of State Conditioning and nested ablation")
    states=[cfg.s_low,cfg.s_medium,cfg.s_high]
    pis={"balanced":cfg.pi_balanced,"low":cfg.pi_low,"high":cfg.pi_high}
    models=["M0","M1","M2","M3"]; rows=[]
    for seed in cfg.seeds:
        z=fixed_latent_base(seed,cfg.n_latent,device)
        for kap in cfg.kappa_grid:
            for model in models:
                gz=(model=="M0"); gs=(model in ("M1","M2")); kk=0.0 if model in ("M0","M1") else kap
                xs=[]; vs=[]
                for s in states:
                    x,v,_=optimize_portfolio(s,z,cfg,uni["mu"],uni["Sigma"],kappa=kk,gamma_static_flag=gs,gamma_zero=gz)
                    # Canonical high-accuracy value evaluation at the returned decision.
                    v,*_=risk_value(x,s,z,cfg,uni["mu"],uni["Sigma"],kappa=kk,gamma_static_flag=gs,gamma_zero=gz)
                    xs.append(x); vs.append(v)
                for piname,pi in pis.items():
                    if model in ("M0","M1") or abs(kap)<1e-14:
                        xstatic=xs[0].detach().clone(); vstatic=float(np.dot(pi,vs))
                    else:
                        xstatic,_=optimize_static_across_states(states,pi,z,cfg,uni["mu"],uni["Sigma"],model,kap)
                        # Crucial V5 safeguard: the static solution is a feasible candidate
                        # for every state problem. Re-solve each state from that common
                        # candidate and retain the lower canonical value. This enforces
                        # optimizer consistency without clipping or altering the VoSC.
                        for j,s in enumerate(states):
                            xr,_,_=optimize_portfolio(s,z,cfg,uni["mu"],uni["Sigma"],x_init=xstatic,kappa=kk,gamma_static_flag=gs,gamma_zero=gz)
                            vr,*_=risk_value(xr,s,z,cfg,uni["mu"],uni["Sigma"],kappa=kk,gamma_static_flag=gs,gamma_zero=gz)
                            if vr < vs[j]: xs[j],vs[j]=xr,vr
                        # Evaluate the static decision with exactly the same statewise
                        # risk evaluator used above; no joint-optimizer objective is reused.
                        sv=[]
                        for s in states:
                            vv,*_=risk_value(xstatic,s,z,cfg,uni["mu"],uni["Sigma"],kappa=kk,gamma_static_flag=gs,gamma_zero=gz); sv.append(vv)
                        vstatic=float(np.dot(pi,sv))
                        # Candidate safeguard for the static problem: each state-optimal
                        # decision is also feasible for the static problem.
                        for xcand in xs:
                            cv=[]
                            for s in states:
                                vv,*_=risk_value(xcand,s,z,cfg,uni["mu"],uni["Sigma"],kappa=kk,gamma_static_flag=gs,gamma_zero=gz); cv.append(vv)
                            vv=float(np.dot(pi,cv))
                            if vv<vstatic: xstatic=xcand.detach().clone(); vstatic=vv
                    vstate=float(np.dot(pi,vs))
                    cstate=float(np.dot(pi,[float(exposure(x,uni["c"]).item()) for x in xs]))
                    rows.append({"seed":seed,"kappa":kap,"model":model,"pi":piname,"v_state":vstate,"v_static":vstatic,
                                 "vosc":vstatic-vstate,"c_state":cstate,"c_static":float(exposure(xstatic,uni["c"]).item()),
                                 "turnover_state":float(np.dot(pi,[float(turnover(x,xstatic).item()) for x in xs]))})
    df=pd.DataFrame(rows); save_csv(df,out/"seed_level"/"seed_level_vosc_ablation.csv")
    figdf=summarize(df,["model","pi","kappa"],["v_state","v_static","vosc","c_state","c_static","turnover_state"]); save_csv(figdf,out/"figure_csv"/"figure_4_data.csv")
    tab=summarize(df[np.isclose(df.kappa,1.0)],["model","pi"],["v_state","v_static","vosc","c_state","turnover_state"]); save_csv(tab,out/"tables"/"table_3_ablation.csv")
    fig,ax=plt.subplots(1,2,figsize=(10.8,4.1))
    q=figdf[(figdf.model=="M3") & (figdf.pi=="balanced")]
    ax[0].plot(q.kappa,q.v_state_mean,marker="o",label="$V^{state}$"); ax[0].plot(q.kappa,q.v_static_mean,marker="s",label="$V^{static}$")
    ax[0].set_xlabel("Inter-state heterogeneity $\\kappa$"); ax[0].set_title("(a) State-conditioned vs static value"); ax[0].legend(frameon=False)
    for piname,g in figdf[figdf.model=="M3"].groupby("pi"):
        ax[1].plot(g.kappa,g.vosc_mean,marker="o",label=piname)
    ax[1].axhline(0,lw=.8); ax[1].set_xlabel("Inter-state heterogeneity $\\kappa$"); ax[1].set_ylabel("VoSC"); ax[1].set_title("(b) Value of State Conditioning"); ax[1].legend(frameon=False)
    fig.tight_layout(); fig.savefig(out/"figures"/"figure_4_vosc.png",dpi=cfg.dpi,bbox_inches="tight"); plt.close(fig)
    return df


def optimize_empirical_cvar(Y: torch.Tensor, cfg: Config, x_init: Optional[torch.Tensor] = None, steps: Optional[int] = None) -> Tuple[torch.Tensor, float, float]:
    """Convex empirical CVaR optimization with multi-start and candidate safeguard.

    The returned oracle can never be worse (on the same empirical sample) than
    the supplied feasible x_init. This prevents optimizer noise from generating
    mechanically negative regret.
    """
    device=Y.device; steps=int(cfg.oracle_steps if steps is None else steps)
    eq=torch.full((cfg.d,),1/cfg.d,device=device)
    starts=[eq if x_init is None else x_init.detach().clone(), eq]
    if cfg.oracle_restarts>2:
        starts.append(project_capped_simplex(eq+torch.linspace(-0.02,0.02,cfg.d,device=device),cfg.weight_cap))
    candidates=[]
    if x_init is not None:
        Lin=-(Y@x_init.detach()); candidates.append((x_init.detach().clone(),empirical_cvar(Lin,cfg.beta),empirical_var(Lin,cfg.beta)))
    for st in starts[:cfg.oracle_restarts]:
        x=st.clone().detach().requires_grad_(True)
        with torch.no_grad(): a0=torch.quantile(-(Y@x.detach()),cfg.beta)
        alpha=a0.detach().clone().requires_grad_(True)
        opt=torch.optim.Adam([x,alpha],lr=0.015)
        best=float("inf"); bestx=x.detach().clone(); stale=0
        for _ in range(steps):
            opt.zero_grad(set_to_none=True)
            L=-(Y@x); obj=alpha+torch.relu(L-alpha).mean()/(1-cfg.beta)
            obj.backward(); torch.nn.utils.clip_grad_norm_([x,alpha],20.0); opt.step()
            with torch.no_grad(): x.copy_(project_capped_simplex(x,cfg.weight_cap))
            cur=float(obj.detach().item())
            if cur<best-1e-11: best=cur; bestx=x.detach().clone(); stale=0
            else: stale+=1
            if stale>=140: break
        L=-(Y@bestx); candidates.append((bestx,empirical_cvar(L,cfg.beta),empirical_var(L,cfg.beta)))
    return min(candidates,key=lambda q:q[1])

def exp5_misspec(cfg,device,uni,out,logger):
    logger.info("[EXP 5/6] Structural misspecification boundary")
    rows=[]; dgp_names=["radial","mean","correlation","combined"]; s=cfg.s_high
    for seed in cfg.seeds:
        z=fixed_latent_base(seed,max(cfg.n_latent,cfg.n_eval),device)
        x_model,_,_=optimize_portfolio(s,z[:cfg.n_latent],cfg,uni["mu"],uni["Sigma"])
        # Model-implied nominal benchmark, evaluated with the SAME CRN. This is the
        # correct reference for specification error; robust risk itself includes a
        # deliberate ambiguity premium and is not a forecast error.
        Y0=sample_returns(seed,cfg.n_eval,s,cfg,uni["mu"],uni["Sigma"],z,dgp="radial",intensity=0.0)
        L0=-(Y0@x_model); pred_cvar=empirical_cvar(L0,cfg.beta); pred_var=empirical_var(L0,cfg.beta)
        for dgp in dgp_names:
            for inten in cfg.misspec_intensity:
                Y=sample_returns(seed,cfg.n_eval,s,cfg,uni["mu"],uni["Sigma"],z,dgp=dgp,intensity=float(inten))
                loss_model=-(Y@x_model); true_cvar=empirical_cvar(loss_model,cfg.beta)
                xo,oracle_cvar,_=optimize_empirical_cvar(Y,cfg,x_init=x_model)
                regret=true_cvar-oracle_cvar
                rows.append({"seed":seed,"dgp":dgp,"intensity":inten,"pred_cvar":pred_cvar,"pred_var":pred_var,"true_cvar":true_cvar,
                             "cvar_bias":pred_cvar-true_cvar,"cvar_rel_error":abs(pred_cvar-true_cvar)/max(abs(true_cvar),1e-10),
                             "tail_exceedance":float((loss_model>pred_var).double().mean().item()),
                             "exceedance_error":float((loss_model>pred_var).double().mean().item())-(1-cfg.beta),
                             "decision_regret":regret,"normalized_regret":regret/max(abs(oracle_cvar),1e-10),
                             "delta_c":float(exposure(x_model,uni["c"]).item()-exposure(xo,uni["c"]).item())})
    df=pd.DataFrame(rows); save_csv(df,out/"seed_level"/"seed_level_misspecification.csv")
    figdf=summarize(df,["dgp","intensity"],["cvar_rel_error","decision_regret","normalized_regret","tail_exceedance","exceedance_error","delta_c"]); save_csv(figdf,out/"figure_csv"/"figure_5_data.csv")
    tab=summarize(df[np.isclose(df.intensity,1.0)],["dgp"],["cvar_rel_error","tail_exceedance","exceedance_error","decision_regret","normalized_regret","delta_c"]); save_csv(tab,out/"tables"/"table_4_misspecification.csv")
    piv=figdf.pivot(index="dgp",columns="intensity",values="normalized_regret_mean")
    fig,ax=plt.subplots(1,2,figsize=(10.8,4.2))
    im=ax[0].imshow(piv.values,aspect="auto"); ax[0].set_xticks(range(len(piv.columns)),[f"{v:.2f}" for v in piv.columns]); ax[0].set_yticks(range(len(piv.index)),piv.index)
    ax[0].set_xlabel("Misspecification intensity"); ax[0].set_title("(a) Normalized decision regret"); fig.colorbar(im,ax=ax[0],fraction=.046)
    piv2=figdf.pivot(index="dgp",columns="intensity",values="cvar_rel_error_mean"); im2=ax[1].imshow(piv2.values,aspect="auto"); ax[1].set_xticks(range(len(piv2.columns)),[f"{v:.2f}" for v in piv2.columns]); ax[1].set_yticks(range(len(piv2.index)),piv2.index)
    ax[1].set_xlabel("Misspecification intensity"); ax[1].set_title("(b) Relative CVaR specification error"); fig.colorbar(im2,ax=ax[1],fraction=.046)
    fig.tight_layout(); fig.savefig(out/"figures"/"figure_5_misspecification.png",dpi=cfg.dpi,bbox_inches="tight"); plt.close(fig)
    return df


def exp6_robustness(cfg,device,uni,out,logger):
    logger.info("[EXP 6/6] Conditional-kernel robustness and signal contamination")
    krows=[]; crows=[]; s=cfg.s_high
    for seed in cfg.seeds:
        z=fixed_latent_base(seed,max(cfg.n_latent,cfg.n_eval),device)
        xg,_,_=optimize_portfolio(s,z[:cfg.n_latent],cfg,uni["mu"],uni["Sigma"])
        # Gaussian and covariance-matched Student true kernels.
        for label,nu in [("Gaussian",None)]+[(f"t{v}",v) for v in cfg.student_nu]:
            Y=sample_returns(seed,cfg.n_eval,s,cfg,uni["mu"],uni["Sigma"],z,dgp="radial",student_nu=nu)
            cvar_model=empirical_cvar(-(Y@xg),cfg.beta)
            xo,cvar_oracle,_=optimize_empirical_cvar(Y,cfg,x_init=xg)
            krows.append({"seed":seed,"kernel":label,"nu":np.nan if nu is None else nu,"decision_regret":cvar_model-cvar_oracle,
                          "delta_x_l1":float(torch.sum(torch.abs(xo-xg)).item()),
                          "delta_c":float(exposure(xo,uni["c"]).item()-exposure(xg,uni["c"]).item()),
                          "true_cvar_model":cvar_model,"true_cvar_oracle":cvar_oracle})
        # Signal contamination: normalized mixture, true state s_ct and generic M.
        g=np.random.default_rng(seed+990_000); sct=float(g.normal(0,1)); market=float(g.normal(0,1))
        # map standardized signal through normal CDF-like logistic to [0,1]
        true_state=1/(1+math.exp(-sct))
        xtrue,_,_=optimize_portfolio(true_state,z[:cfg.n_latent],cfg,uni["mu"],uni["Sigma"])
        # Evaluate decisions under true DGP for regret.
        Ytrue=sample_returns(seed,cfg.n_eval,true_state,cfg,uni["mu"],uni["Sigma"],z,dgp="radial")
        xoracle,oracle,_=optimize_empirical_cvar(Ytrue,cfg,x_init=xtrue)
        for lam in cfg.contamination_grid:
            sobs=math.sqrt(max(0.0,1-lam*lam))*sct+lam*market
            obs_state=1/(1+math.exp(-sobs))
            xobs,_,_=optimize_portfolio(obs_state,z[:cfg.n_latent],cfg,uni["mu"],uni["Sigma"])
            cv=empirical_cvar(-(Ytrue@xobs),cfg.beta)
            # 3-bin state classification
            def cls(v): return 0 if v<1/3 else (1 if v<2/3 else 2)
            crows.append({"seed":seed,"lambda":lam,"s_true":true_state,"s_obs":obs_state,"state_correct":int(cls(true_state)==cls(obs_state)),
                          "state_abs_error":abs(obs_state-true_state),"decision_regret":cv-oracle,
                          "delta_x_l1":float(torch.sum(torch.abs(xobs-xtrue)).item()),
                          "exposure_abs_error":abs(float(exposure(xobs,uni["c"]).item()-exposure(xtrue,uni["c"]).item()))})
    kdf=pd.DataFrame(krows); cdf=pd.DataFrame(crows); save_csv(kdf,out/"seed_level"/"seed_level_kernel_robustness.csv"); save_csv(cdf,out/"seed_level"/"seed_level_signal_contamination.csv")
    ksum=summarize(kdf,["kernel"],["decision_regret","delta_x_l1","delta_c","true_cvar_model"])
    csum=summarize(cdf,["lambda"],["decision_regret","delta_x_l1","exposure_abs_error","state_abs_error","state_correct"])
    ksum["panel"]="kernel"; csum["panel"]="contamination"
    figdf=pd.concat([ksum,csum],ignore_index=True,sort=False); save_csv(figdf,out/"figure_csv"/"figure_6_data.csv")
    fig,ax=plt.subplots(1,2,figsize=(10.8,4.1))
    order=["Gaussian"]+[f"t{v}" for v in cfg.student_nu]; q=ksum.set_index("kernel").loc[order].reset_index()
    ax[0].errorbar(range(len(q)),q.decision_regret_mean,yerr=[q.decision_regret_mean-q.decision_regret_ci_low,q.decision_regret_ci_high-q.decision_regret_mean],fmt="o",capsize=3)
    ax[0].set_xticks(range(len(q)),q.kernel); ax[0].set_ylabel("Decision regret"); ax[0].set_title("(a) Conditional-kernel robustness")
    ax[1].plot(csum["lambda"],csum.decision_regret_mean,marker="o"); ax[1].fill_between(csum["lambda"],csum.decision_regret_ci_low,csum.decision_regret_ci_high,alpha=.15)
    ax[1].set_xlabel("Signal contamination $\\lambda$"); ax[1].set_ylabel("Decision regret"); ax[1].set_title("(b) Signal contamination")
    fig.tight_layout(); fig.savefig(out/"figures"/"figure_6_robustness.png",dpi=cfg.dpi,bbox_inches="tight"); plt.close(fig)
    return kdf,cdf


# -----------------------------------------------------------------------------
# Tables, diagnostics and audit
# -----------------------------------------------------------------------------
def parameter_table(cfg: Config, device: torch.device) -> pd.DataFrame:
    entries=[
        ("DGP","Portfolio dimension d",cfg.d,"Common universe","No"),
        ("DGP","State grid size",cfg.n_state,"Comparative statics","No"),
        ("DGP","Latent family","LogNormal","q_{Phi(s)}","Yes"),
        ("DGP","xi_m0",cfg.xi_m0,"Latent center","Yes"),("DGP","xi_a",cfg.xi_a,"State slope of latent center","Yes"),
        ("DGP","xi_sigma0",cfg.xi_sigma0,"Latent dispersion","Yes"),("DGP","xi_b",cfg.xi_b,"State slope of latent dispersion","Yes"),
        ("Decision","CVaR beta",cfg.beta,"Tail probability level","No"),("Decision","Gamma(s)",f"{cfg.gamma0}+{cfg.gamma1} s","Decision-independent KL radius","Yes"),
        ("Decision","Weight cap",cfg.weight_cap,"Feasible set","No"),("Decision","Exposure scores",f"G={cfg.exposure_green}, N={cfg.exposure_neutral}, B={cfg.exposure_brown}","C(x)=c'x","No"),
        ("Numerical","Independent seeds",len(cfg.seeds),"Replicated paired evaluation","No"),("Numerical","Latent CRN draws",cfg.n_latent,"Variational expectation","No"),
        ("Numerical","Evaluation draws",cfg.n_eval,"True-DGP evaluation","No"),("Numerical","Outer steps",cfg.x_steps,"Portfolio optimization","No"),
        ("Numerical","Inner steps",cfg.inner_steps,"alpha/rho optimization","No"),("Numerical","Finite-difference h",cfg.fd_h,"Derivative audit","Yes"),
        ("Numerical","dtype",cfg.dtype,"Numerical precision","No"),("Numerical","device",str(device),"Compute backend","No"),
    ]
    return pd.DataFrame(entries,columns=["Category","Parameter","Configuration","Role","Varied_in_robustness"])


def build_diagnostics(cfg, policy_df, policy_diag, exp1_df, vosc_df, misspec_df, out, logger):
    checks=[]
    def add(name,value,threshold,rule,interpretation):
        if rule=="le": status="PASS" if value<=threshold else "FAIL"
        elif rule=="ge": status="PASS" if value>=threshold else "FAIL"
        else: status="WARN"
        checks.append({"Check":name,"Value":value,"Threshold":threshold,"Rule":rule,"Status":status,"Interpretation":interpretation})
        logger.info("CHECK %-38s %s | value=%s threshold=%s",name,status,f"{value:.6g}" if isinstance(value,(int,float,np.floating)) else value,threshold)
    add("State-gradient MAE",float(exp1_df.grad_abs_error.mean()),cfg.pass_grad_mae,"le","Analytical decomposition vs central finite difference")
    grad_err=exp1_df.g_total-exp1_df.g_fd
    grad_rmse=float(np.sqrt(np.mean(np.square(grad_err))))
    grad_max=float(np.max(np.abs(grad_err)))
    grad_r2=float(1.0-np.sum(np.square(grad_err))/max(np.sum(np.square(exp1_df.g_fd-exp1_df.g_fd.mean())),1e-15))
    checks.extend([{ "Check":"State-gradient RMSE","Value":grad_rmse,"Threshold":np.nan,"Rule":"audit","Status":"AUDIT","Interpretation":"Supplementary derivative-recovery metric"}, {"Check":"State-gradient max abs error","Value":grad_max,"Threshold":np.nan,"Rule":"audit","Status":"AUDIT","Interpretation":"Worst derivative-recovery discrepancy"}, {"Check":"State-gradient R2","Value":grad_r2,"Threshold":np.nan,"Rule":"audit","Status":"AUDIT","Interpretation":"Agreement of analytical and finite-difference derivatives"}])
    q=policy_df[(policy_df.get("inequality_active",1)==0) & policy_df.policy_deriv_mae.notna()]
    pol_mae=float(q.policy_deriv_mae.mean()) if len(q) else np.inf
    pol_signal=float(np.mean(np.abs(q.dc_fd))) if len(q) else 0.0
    pol_rel=float(np.mean(np.abs(q.dc_analytical-q.dc_fd))/max(np.mean(np.abs(q.dc_fd)),1e-12)) if len(q) else np.inf
    add("Policy derivative signal",pol_signal,cfg.min_policy_deriv_signal,"ge","Comparative-static experiment must excite a non-degenerate policy response")
    add("Policy-derivative MAE",pol_mae,cfg.pass_policy_deriv_mae,"le","Theorem 3.4 evaluated only where box constraints are inactive")
    add("Policy-derivative relative MAE",pol_rel,cfg.pass_policy_deriv_rel_mae,"le","Scale-free analytical versus finite-difference policy sensitivity")
    if len(policy_diag):
        add("Max simplex error",float(policy_diag.simplex_error.max()),cfg.pass_simplex_error,"le","Budget feasibility")
        add("Max cap violation",float(policy_diag.cap_error.max()),cfg.pass_cap_error,"le","Upper-bound feasibility")
        rate=float(policy_diag.outer_converged.mean()); add("Portfolio solver convergence rate",rate,cfg.pass_solver_rate,"ge","Reduced joint KKT residual plus feasibility")
    q0=vosc_df[(vosc_df.model=="M3") & np.isclose(vosc_df.kappa,0.0)]
    add("VoSC at zero heterogeneity",float(np.abs(q0.vosc).mean()),cfg.pass_vosc_zero_tol,"le","Zero-heterogeneity numerical identity")
    # Theoretical inequality should hold up to numerical tolerance.
    add("Minimum VoSC",float(vosc_df.vosc.min()),-cfg.pass_vosc_zero_tol,"ge","V_static >= V_state up to optimization tolerance")
    radial0=misspec_df[(misspec_df.dgp=="radial") & np.isclose(misspec_df.intensity,0.0)]
    add("Radial baseline CVaR error",float(radial0.cvar_rel_error.mean()),1e-10,"le","Correctly specified radial DGP under common random numbers")
    add("Minimum decision regret",float(misspec_df.decision_regret.min()),-cfg.pass_regret_tol,"ge","Oracle dominance up to Monte Carlo/optimization tolerance")
    d=pd.DataFrame(checks); save_csv(d,out/"tables"/"table_5_diagnostics.csv")
    return d


def save_state_grid(cfg,out):
    s=np.linspace(0,1,cfg.n_state)
    pd.DataFrame({"state":s,"label":["Low" if abs(x-cfg.s_low)<.03 else "Medium" if abs(x-cfg.s_medium)<.03 else "High" if abs(x-cfg.s_high)<.03 else "" for x in s]}).to_csv(out/"config"/"state_grid.csv",index=False)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def parse_args():
    p=argparse.ArgumentParser(description="Controlled simulation study for state-dependent latent KL-DRO")
    p.add_argument("--output",default="Results_Section4_Q1PP")
    p.add_argument("--use-cuda",action="store_true",help="Use CUDA when available")
    p.add_argument("--cpu",action="store_true",help="Force CPU")
    p.add_argument("--smoke-test",action="store_true",help="Small run for implementation checks")
    return p.parse_args()


def main():
    args=parse_args(); out=Path(args.output); out.mkdir(parents=True,exist_ok=True); output_dirs(out)
    cfg=Config(); cfg.use_cuda=bool(args.use_cuda) or cfg.use_cuda
    if args.smoke_test:
        cfg.seeds=cfg.seeds[:2]; cfg.n_state=7; cfg.n_latent=512; cfg.n_eval=1200; cfg.x_steps=110; cfg.inner_steps=100; cfg.polish_steps=80; cfg.x_restarts=2; cfg.oracle_steps=350; cfg.oracle_restarts=2; cfg.kappa_grid=[0.0,1.0]; cfg.misspec_intensity=[0.0,1.0]; cfg.contamination_grid=[0.0,0.6,1.0]
    logger=setup_logger(out); device=select_device(cfg,force_cpu=args.cpu)
    seed_all(cfg.seeds[0])
    logger.info("Section 4 controlled simulation V5 started")
    logger.info("Device=%s | CUDA available=%s | torch=%s",device,torch.cuda.is_available(),torch.__version__)
    if device.type=="cuda": logger.info("GPU=%s",torch.cuda.get_device_name(0))
    logger.info("Design invariant: Gamma(s) is decision independent; no Gamma(s,C(x)) is implemented.")

    meta={"config":asdict(cfg),"environment":{"python":sys.version,"platform":platform.platform(),"torch":torch.__version__,"numpy":np.__version__,"pandas":pd.__version__,"device":str(device),"cuda_available":torch.cuda.is_available(),"gpu":torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}}
    (out/"config"/"simulation_config.json").write_text(json.dumps(meta,indent=2),encoding="utf-8")
    # CSV mirror requested by the study protocol.
    pd.json_normalize(meta["config"],sep=".").T.reset_index().rename(columns={"index":"parameter",0:"value"}).to_csv(out/"config"/"simulation_config.csv",index=False)
    save_state_grid(cfg,out)
    ptab=parameter_table(cfg,device); save_csv(ptab,out/"tables"/"table_1_parameters.csv")

    uni=build_universe(cfg,device)
    t0=time.time()
    exp1=exp1_mechanism(cfg,device,uni,out,logger)
    pol,pdiag=exp2_policy(cfg,device,uni,out,logger)
    exp3_heterogeneity(cfg,device,uni,out,logger)
    vosc=exp4_vosc(cfg,device,uni,out,logger)
    misspec=exp5_misspec(cfg,device,uni,out,logger)
    exp6_robustness(cfg,device,uni,out,logger)

    # Unified optimization diagnostics required by protocol.
    save_csv(pdiag,out/"diagnostics"/"optimization_diagnostics.csv")
    diagnostics=build_diagnostics(cfg,pol,pdiag,exp1,vosc,misspec,out,logger)
    nfail=int((diagnostics.Status=="FAIL").sum())
    logger.info("Completed in %.2f minutes | diagnostics: %d PASS, %d FAIL",(time.time()-t0)/60.0,int((diagnostics.Status=="PASS").sum()),nfail)
    logger.info("Outputs organized in figures/, figure_csv/, tables/, seed_level/, diagnostics/, and config/ under %s",out.resolve())
    return 0 if nfail==0 else 2


if __name__=="__main__":
    raise SystemExit(main())
