#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Section 6 — Q1++ empirical pipeline V7: final identification and inference
Variational Distributional Robustness under Climate Transition Risk

Design principles
-----------------
1. Strict recursive timing: no observation after a decision date enters that decision.
2. S_t conditions the nominal latent law; Xi_{t+1} remains future/latent.
3. KL ambiguity is on the latent scale law, never decision-dependent.
4. Transition exposure C_t(x)=c_t'x is an evaluation object only.
5. State-conditioning ablations isolate nominal-law and ambiguity-radius channels.
6. All figure/table source data are exported to CSV before plotting.

The script is intentionally self-auditing. If an empirical ingredient required by the
locked manuscript is absent (notably CPU_narrow or dated asset transition scores),
it records a NO PASS rather than silently constructing a substitute.

Recommended command
-------------------
python section6_q1pp_empirical_v7_fixed.py --data-dir "C:/Users/fredy/Downloads/Univers" --state-dir "C:/Users/fredy/Downloads/Z_t" --use-cuda

Quick smoke test
----------------
python section6_q1pp_empirical_v7_fixed.py --data-dir "C:/Users/fredy/Downloads/Univers" --state-dir "C:/Users/fredy/Downloads/Z_t" --fast --use-cuda
"""
from __future__ import annotations
import argparse, json, logging, math, os, platform, random, sys, time, warnings
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm, spearmanr, pearsonr
import matplotlib.pyplot as plt

try:
    import torch
except Exception as e:
    raise RuntimeError("PyTorch is required. Install a CUDA-enabled PyTorch build.") from e

warnings.filterwarnings("ignore", category=RuntimeWarning)

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
@dataclass
class Config:
    data_dir: str = r"C:/Users/fredy/Downloads/Univers"
    output_dir: str = "Section6_Q1PP_Results"
    use_cuda: bool = True
    seed: int = 20260926
    dtype: str = "float64"
    beta: float = 0.95
    state_window: int = 120
    state_window_sensitivity: Tuple[int, ...] = (60, 90, 120, 252)
    radial_window: int = 252
    radial_min_obs: int = 120
    covariance_ridge: float = 1e-6
    conditional_bandwidth: float = 0.15
    conditional_min_effective_n: float = 20.0
    bootstrap_B: int = 500
    bootstrap_block: int = 20
    bootstrap_quantile: float = 0.95
    bootstrap_quantile_sensitivity: Tuple[float, ...] = (0.90, 0.95, 0.99)
    rebalance_H: int = 20
    rebalance_sensitivity: Tuple[int, ...] = (10, 20, 40)
    train_end: str = "2019-12-31"
    validation_end: str = "2021-12-31"
    test_end: str = "2026-08-31"
    latent_mc: int = 2048
    return_mc: int = 4096
    outer_steps: int = 220
    inner_steps: int = 100
    lr_weights: float = 0.06
    lr_inner: float = 0.05
    max_weight: Optional[float] = None  # None = long-only fully invested; do not invent a cap.
    transaction_cost_bps: Tuple[int, ...] = (5, 10, 25, 50)
    n_ci_boot: int = 1000
    ci_block: int = 20
    ci_level: float = 0.95
    fast: bool = False
    # File candidates. Script also discovers columns by aliases.
    universe_a_candidates: Tuple[str, ...] = ("Universe_A_log_returns.csv", "Universe_A_log_returns(3).csv")
    universe_b_candidates: Tuple[str, ...] = ("Universe_B_log_returns.csv", "Universe_B_log_returns(3).csv")
    factors_candidates: Tuple[str, ...] = ("Z_factors_daily_common.csv", "Z_factors_daily_common(1).csv", "Z_factors_daily.csv", "Z_factors_daily(1).csv")
    exposure_candidates: Tuple[str, ...] = ("transition_exposure_scores.csv", "asset_transition_exposure.csv", "c_t_scores.csv")

ETF_BLOCKS = {
    "Fossil energy": ["XLE","VDE","XOP","AMLP","FCG"],
    "Clean energy": ["ICLN","TAN","QCLN","PBW","PBD"],
    "Industrials": ["XLI","VIS","PPA"],
    "Utilities": ["XLU","VPU","IDU"],
    "Broad and defensive": ["SPY","QQQ","GLD","TLT"],
}
FIRM_BLOCKS = {
    "Fossil energy": ["XOM","CVX","COP","OXY","SLB"],
    "Transition-oriented": ["FSLR","NEE","TSLA","ALB","ENPH"],
    "Transition-relevant industrials": ["CAT","DE","HON","ETN","EMR"],
    "Electricity and utilities": ["DUK","SO","AEP","EXC","SRE"],
}

# -----------------------------------------------------------------------------
# Logging, checks, reproducibility
# -----------------------------------------------------------------------------
class Audit:
    def __init__(self, logger): self.logger, self.rows = logger, []
    def check(self, name: str, ok: bool, detail: str = "", fatal: bool = False):
        status = "PASS" if ok else "NO PASS"
        self.rows.append({"check": name, "status": status, "detail": detail})
        self.logger.info("CHECK %-7s | %-42s | %s", status, name, detail)
        if fatal and not ok: raise RuntimeError(f"{name}: {detail}")
    def save(self, path: Path): pd.DataFrame(self.rows).to_csv(path, index=False)

def setup_logger(out: Path):
    out.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("section6"); logger.setLevel(logging.INFO); logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%Y-%m-%d %H:%M:%S")
    fh = logging.FileHandler(out/"section6_run.log", encoding="utf-8"); fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stdout); sh.setFormatter(fmt)
    logger.addHandler(fh); logger.addHandler(sh); return logger

def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def resolve_file(root: Path, candidates: Tuple[str,...]) -> Optional[Path]:
    for n in candidates:
        p=root/n
        if p.exists(): return p
    lower={p.name.lower():p for p in root.glob("*.csv")}
    for n in candidates:
        if n.lower() in lower: return lower[n.lower()]
    return None

def pick_col(df, aliases):
    normed={c.lower().replace(" ","").replace("-","").replace("_",""):c for c in df.columns}
    for a in aliases:
        k=a.lower().replace(" ","").replace("-","").replace("_","")
        if k in normed: return normed[k]
    return None

def read_date_csv(path: Path) -> pd.DataFrame:
    d=pd.read_csv(path); dc=pick_col(d,["Date","date"])
    if dc is None: raise ValueError(f"No Date column in {path}")
    d[dc]=pd.to_datetime(d[dc]); d=d.rename(columns={dc:"Date"}).sort_values("Date").drop_duplicates("Date")
    return d.set_index("Date")

# -----------------------------------------------------------------------------
# State construction — locked Section 5 chronology
# -----------------------------------------------------------------------------
def rolling_z(x: pd.Series, W: int) -> pd.Series:
    # includes current observation: nonanticipative, as in locked design
    mu=x.rolling(W, min_periods=W).mean(); sd=x.rolling(W, min_periods=W).std(ddof=1)
    return (x-mu)/sd.replace(0,np.nan)

def rolling_midrank(x: pd.Series, W: int) -> pd.Series:
    a=x.to_numpy(float); out=np.full(len(a),np.nan)
    for i in range(W-1,len(a)):
        win=a[i-W+1:i+1]
        if np.isfinite(win).all():
            z=a[i]; out[i]=(np.sum(win<z)+0.5*np.sum(win==z))/W
    return pd.Series(out,index=x.index)

def construct_state(factors: pd.DataFrame, W: int, audit: Audit) -> pd.DataFrame:
    eua=pick_col(factors,["ZCT_EUA_logret","EUA_logret","Delta_log_EUA"])
    cpu=pick_col(factors,["ZCT_CPU_narrow","CPU_narrow","CPU_narrow_lagged","Climate_Policy_Uncertainty_narrow"])
    audit.check("EUA factor available", eua is not None, str(eua), fatal=True)
    audit.check("CPU narrow factor available", cpu is not None,
                "Required by locked Section 5; no substitute is silently used." if cpu is None else str(cpu), fatal=True)
    z1=rolling_z(factors[eua].astype(float),W); z2=rolling_z(factors[cpu].astype(float),W)
    idx=0.5*(z1+z2); s=rolling_midrank(idx,W)
    return pd.DataFrame({"EUA":factors[eua],"CPU":factors[cpu],"Z_EUA":z1,"Z_CPU":z2,"I_CT":idx,"S_CT":s})

def construct_competing_state(factors: pd.DataFrame, W: int, kind: str) -> pd.Series:
    if kind=="market": aliases=[["ZM_VIX_level"],["ZM_MOVE_level"],["ZM_EPU_UNVALIDATED_level"]]
    elif kind=="energy": aliases=[["ZE_OVX_level"],["ZE_WTI_logret"],["ZE_TTF_logret"]]
    else: raise ValueError(kind)
    cols=[pick_col(factors,a) for a in aliases]; cols=[c for c in cols if c]
    if not cols: return pd.Series(np.nan,index=factors.index)
    zs=pd.concat([rolling_z(factors[c].astype(float),W) for c in cols],axis=1)
    idx=zs.mean(axis=1,skipna=False)
    return rolling_midrank(idx,W)

# -----------------------------------------------------------------------------
# Radial proxy and conditional lognormal q_{Phi_t(S_t)}
# -----------------------------------------------------------------------------
def trailing_radial_proxy(R: pd.DataFrame, cfg: Config) -> pd.Series:
    X=R.to_numpy(float); d=X.shape[1]; out=np.full(len(R),np.nan)
    for i in range(cfg.radial_min_obs,len(R)):
        j=max(0,i-cfg.radial_window); hist=X[j:i]
        hist=hist[np.isfinite(hist).all(axis=1)]
        if len(hist)<cfg.radial_min_obs or not np.isfinite(X[i]).all(): continue
        mu=hist.mean(0); S=np.cov(hist,rowvar=False,ddof=1)
        ridge=max(cfg.covariance_ridge*np.trace(S)/d,1e-12); S=S+ridge*np.eye(d)
        z=X[i]-mu
        try: out[i]=float(z@np.linalg.solve(S,z)/d)
        except np.linalg.LinAlgError: pass
    return pd.Series(out,index=R.index,name="xi_hat")

def weighted_normal_fit(logxi: np.ndarray, s_hist: np.ndarray, s0: float, bw: float):
    ok=np.isfinite(logxi)&np.isfinite(s_hist); y=logxi[ok]; s=s_hist[ok]
    if len(y)<10: return None
    w=np.exp(-0.5*((s-s0)/bw)**2); sw=w.sum()
    if sw<=1e-12: return None
    w=w/sw; neff=1.0/np.sum(w*w)
    m=np.sum(w*y); v=np.sum(w*(y-m)**2); sig=math.sqrt(max(v,1e-8))
    return m,sig,neff

def normal_kl(m1,s1,m0,s0):
    s1=max(float(s1),1e-8); s0=max(float(s0),1e-8)
    return math.log(s0/s1)+(s1*s1+(m1-m0)**2)/(2*s0*s0)-0.5

def circular_block_indices(n:int, block:int, rng:np.random.Generator):
    need=int(math.ceil(n/block)); starts=rng.integers(0,n,size=need)
    return np.concatenate([(np.arange(st,st+block)%n) for st in starts])[:n]

def estimate_q_gamma(s_hist, xi_hist, s0, cfg:Config, rng, B=None):
    ok=np.isfinite(s_hist)&np.isfinite(xi_hist)&(xi_hist>0)
    s=np.asarray(s_hist)[ok]; y=np.log(np.asarray(xi_hist)[ok])
    fit=weighted_normal_fit(y,s,s0,cfg.conditional_bandwidth)
    if fit is None: return None
    m,sig,neff=fit
    B=cfg.bootstrap_B if B is None else B; kls=[]
    for _ in range(B):
        ind=circular_block_indices(len(y),cfg.bootstrap_block,rng)
        fb=weighted_normal_fit(y[ind],s[ind],s0,cfg.conditional_bandwidth)
        if fb is None: continue
        mb,sb,_=fb; val=normal_kl(mb,sb,m,sig)
        if np.isfinite(val): kls.append(val)
    if not kls: return None
    return {"m":m,"sigma":sig,"neff":neff,"gamma":float(np.quantile(kls,cfg.bootstrap_quantile)),
            "B_valid":len(kls),"kl_boot":np.asarray(kls)}

def fit_static_q(xi_hist):
    y=np.log(np.asarray(xi_hist)); y=y[np.isfinite(y)]
    if len(y)<20:return None
    return float(y.mean()),float(max(y.std(ddof=1),1e-6))

# -----------------------------------------------------------------------------
# Portfolio optimizers
# -----------------------------------------------------------------------------
def project_capped_simplex_np(x, cap=None):
    x=np.maximum(np.asarray(x,float),0)
    if cap is None:
        s=x.sum(); return x/s if s>0 else np.ones_like(x)/len(x)
    cap=float(cap)
    if cap*len(x)<1-1e-12: raise ValueError("max_weight infeasible")
    # bisection projection onto {sum=1, 0<=x<=cap}
    lo,hi=x.min()-cap,x.max()
    for _ in range(80):
        mid=(lo+hi)/2; y=np.clip(x-mid,0,cap)
        if y.sum()>1: lo=mid
        else: hi=mid
    y=np.clip(x-hi,0,cap); return y/y.sum()

def nominal_cvar_weights(hist:np.ndarray,beta:float,cap=None):
    n,d=hist.shape
    def f(z):
        w=z[:d]; a=z[d]; loss=-hist@w
        return a+np.maximum(loss-a,0).mean()/(1-beta)
    x0=np.r_[np.ones(d)/d,0.0]; cons={"type":"eq","fun":lambda z:z[:d].sum()-1}
    b=[(0,cap if cap else 1)]*d+[(None,None)]
    res=minimize(f,x0,method="SLSQP",bounds=b,constraints=cons,options={"maxiter":400,"ftol":1e-10})
    return project_capped_simplex_np(res.x[:d] if res.success else x0[:d],cap)

def minvar_weights(hist,cap=None):
    d=hist.shape[1]; S=np.cov(hist,rowvar=False)+1e-10*np.eye(d)
    def f(w):return float(w@S@w)
    res=minimize(f,np.ones(d)/d,method="SLSQP",bounds=[(0,cap if cap else 1)]*d,
                 constraints={"type":"eq","fun":lambda w:w.sum()-1},options={"maxiter":300,"ftol":1e-12})
    return project_capped_simplex_np(res.x,cap)

def risk_parity_weights(hist,cap=None):
    d=hist.shape[1]; S=np.cov(hist,rowvar=False)+1e-10*np.eye(d)
    def f(w):
        v=max(float(w@S@w),1e-16); rc=w*(S@w)/v
        return float(np.sum((rc-1/d)**2))
    res=minimize(f,np.ones(d)/d,method="SLSQP",bounds=[(1e-8,cap if cap else 1)]*d,
                 constraints={"type":"eq","fun":lambda w:w.sum()-1},options={"maxiter":500,"ftol":1e-12})
    return project_capped_simplex_np(res.x,cap)

def torch_latent_portfolio(mu_np,S_np,m_logxi,s_logxi,gamma,cfg:Config,device,seed=0):
    """Minimize latent KL-robust CVaR using the Section-3 entropic dual.
    q is LogNormal(m_logxi,s_logxi^2). gamma=0 gives state-conditioned nominal CVaR.
    """
    dtype=torch.float64 if cfg.dtype=="float64" else torch.float32
    mu=torch.tensor(mu_np,device=device,dtype=dtype); S=torch.tensor(S_np,device=device,dtype=dtype)
    d=len(mu); gen=torch.Generator(device=device); gen.manual_seed(seed)
    z=torch.randn(cfg.latent_mc,device=device,dtype=dtype,generator=gen)
    xi=torch.exp(torch.tensor(m_logxi,device=device,dtype=dtype)+torch.tensor(s_logxi,device=device,dtype=dtype)*z)
    theta=torch.zeros(d,device=device,dtype=dtype,requires_grad=True)
    alpha=torch.tensor(0.,device=device,dtype=dtype,requires_grad=True)
    raw_eta=torch.tensor(-1.,device=device,dtype=dtype,requires_grad=True)
    opt=torch.optim.Adam([theta,alpha,raw_eta],lr=cfg.lr_weights)
    best=(float("inf"),None)
    for step in range(cfg.outer_steps):
        opt.zero_grad()
        w=torch.softmax(theta,0)
        if cfg.max_weight is not None:
            # differentiable soft penalty; exact projection after optimization
            cap_pen=2e3*torch.relu(w-cfg.max_weight).pow(2).sum()
        else: cap_pen=0.0
        mx=-(w@mu); sx=torch.sqrt(torch.clamp(w@S@w,min=1e-14))
        scale=torch.sqrt(torch.clamp(xi,min=1e-12))*sx
        zz=(alpha-mx)/scale
        phi=torch.exp(-0.5*zz*zz)/math.sqrt(2*math.pi)
        Phi=0.5*(1+torch.erf((mx-alpha)/(scale*math.sqrt(2))))
        h=scale*phi+(mx-alpha)*Phi
        if gamma<=1e-14:
            robust_h=h.mean()
        else:
            eta=torch.nn.functional.softplus(raw_eta)+1e-8
            robust_h=eta*gamma+eta*(torch.logsumexp(h/eta,0)-math.log(h.numel()))
        obj=alpha+robust_h/(1-cfg.beta)+cap_pen
        obj.backward(); torch.nn.utils.clip_grad_norm_([theta,alpha,raw_eta],20.0); opt.step()
        val=float(obj.detach().cpu())
        if val<best[0] and np.isfinite(val): best=(val,torch.softmax(theta.detach(),0).cpu().numpy())
    w=best[1] if best[1] is not None else np.ones(d)/d
    return project_capped_simplex_np(w,cfg.max_weight),best[0]

# Finite-sample global KL and Wasserstein benchmarks. They are deliberately
# separate from the proposed latent ambiguity model.
def global_kl_cvar_weights(hist,beta,gamma,cap=None):
    n,d=hist.shape
    def f(z):
        w=z[:d]; a=z[d]; eta=max(z[d+1],1e-8); h=np.maximum(-hist@w-a,0)/(1-beta)
        m=h.max(); lme=m/eta+np.log(np.mean(np.exp((h-m)/eta)))
        return a+eta*gamma+eta*lme
    x0=np.r_[np.ones(d)/d,0.,0.01]; bounds=[(0,cap if cap else 1)]*d+[(None,None),(1e-8,None)]
    res=minimize(f,x0,method="SLSQP",bounds=bounds,constraints={"type":"eq","fun":lambda z:z[:d].sum()-1},
                 options={"maxiter":500,"ftol":1e-9})
    return project_capped_simplex_np(res.x[:d] if res.success else x0[:d],cap)

def wasserstein_cvar_weights(hist,beta,radius,cap=None):
    # 1-Wasserstein Lipschitz upper-envelope for linear loss/CVaR under L2 transport:
    # empirical CVaR + radius*||w||_2/(1-beta). Transparent benchmark, not latent DRO.
    n,d=hist.shape
    def f(z):
        w=z[:d]; a=z[d]; emp=a+np.maximum(-hist@w-a,0).mean()/(1-beta)
        return emp+radius*np.linalg.norm(w)/(1-beta)
    x0=np.r_[np.ones(d)/d,0.]; bounds=[(0,cap if cap else 1)]*d+[(None,None)]
    res=minimize(f,x0,method="SLSQP",bounds=bounds,constraints={"type":"eq","fun":lambda z:z[:d].sum()-1},
                 options={"maxiter":500,"ftol":1e-9})
    return project_capped_simplex_np(res.x[:d] if res.success else x0[:d],cap)

# -----------------------------------------------------------------------------
# Evaluation
# -----------------------------------------------------------------------------
def simple_from_log(R): return np.expm1(R)
def drift_weights(w, simple_returns):
    gross=np.prod(1+simple_returns,axis=0); z=w*gross; return z/z.sum() if z.sum()>0 else w

def block_boot_ci(x,stat=np.mean,B=1000,block=20,level=.95,seed=0):
    x=np.asarray(x,float); x=x[np.isfinite(x)]; n=len(x)
    if n<5:return np.nan,np.nan
    rng=np.random.default_rng(seed); vals=[]
    for _ in range(B): vals.append(stat(x[circular_block_indices(n,min(block,n),rng)]))
    a=(1-level)/2; return float(np.quantile(vals,a)),float(np.quantile(vals,1-a))

def metrics(r,turnover=None):
    r=np.asarray(r,float); r=r[np.isfinite(r)]
    if len(r)==0:return {}
    ann=252; mu=r.mean()*ann; vol=r.std(ddof=1)*math.sqrt(ann); neg=r[r<0]
    sortino=mu/(neg.std(ddof=1)*math.sqrt(ann)) if len(neg)>1 and neg.std(ddof=1)>0 else np.nan
    wealth=np.cumprod(1+r); dd=wealth/np.maximum.accumulate(wealth)-1
    loss=-r; var=np.quantile(loss,.95); es=loss[loss>=var].mean() if np.any(loss>=var) else var
    return {"AnnReturn":mu,"AnnVol":vol,"Sharpe":mu/vol if vol>0 else np.nan,"Sortino":sortino,
            "MaxDD":dd.min(),"VaR95":var,"ES95":es,"Turnover":np.nanmean(turnover) if turnover is not None else np.nan}

# -----------------------------------------------------------------------------
# Exposure scores — never infer c_t from economic blocks
# -----------------------------------------------------------------------------
def load_exposure_scores(root:Path,assets:List[str],dates:pd.DatetimeIndex,candidates,audit:Audit):
    p=resolve_file(root,candidates)
    if p is None:
        audit.check("Transition exposure scores available",False,
                    "No dated c_{i,t} file found. Exposure figures/tables will be marked unavailable; economic blocks are NOT substituted for c_t.")
        return None
    d=read_date_csv(p)
    missing=[a for a in assets if a not in d.columns]
    audit.check("Exposure score asset coverage",not missing,f"missing={missing}")
    if missing:return None
    # carry forward only already observed scores
    return d[assets].reindex(dates).ffill()

# -----------------------------------------------------------------------------
# Main recursive experiment
# -----------------------------------------------------------------------------
def run_universe(name,R,state_df,factors,cfg,device,audit,out,exposure=None,H=None,Wstate=None):
    H=H or cfg.rebalance_H; Wstate=Wstate or cfg.state_window
    idx=R.index.intersection(state_df.index); R=R.loc[idx].dropna(); st=state_df.reindex(R.index)
    xi=trailing_radial_proxy(R,cfg)
    # common decision calendar: first admissible date then every H common return dates
    valid=np.where(np.isfinite(st["S_CT"].to_numpy()) & np.isfinite(xi.shift(1).to_numpy()))[0]
    if len(valid)==0: raise RuntimeError(f"No admissible decision dates for {name}")
    start=max(valid[0],cfg.radial_min_obs+1); positions=np.arange(start,len(R)-1,H)
    positions=positions[R.index[positions]<=pd.Timestamp(cfg.test_end)]
    if cfg.fast: positions=positions[-min(12,len(positions)):]
    rng=np.random.default_rng(cfg.seed+(1 if name=="ETF" else 2))
    rows=[]; weights=[]; daily=[]; prev_post={}
    methods=["EqualWeight","MinVariance","RiskParity","NominalCVaR","GlobalKL","Wasserstein",
             "A1_StaticLatentKL","A2_StateNominal","A3_StateLatentKL","A4_StateQ_MeanGamma","A5_StaticQ_DateGamma"]
    # cache date-specific q/gamma first, enabling mean gamma ablation without look-ahead by using expanding past mean
    gamma_history=[]
    for kk,pos in enumerate(positions):
        date=R.index[pos]; hist=R.iloc[max(0,pos-cfg.radial_window):pos].to_numpy(float)
        s0=float(st.loc[date,"S_CT"])
        admiss=(R.index < date)
        qg=estimate_q_gamma(st.loc[admiss,"S_CT"].to_numpy(),xi.loc[admiss].to_numpy(),s0,cfg,rng,B=(60 if cfg.fast else cfg.bootstrap_B))
        static=fit_static_q(xi.loc[admiss].to_numpy())
        if qg is None or static is None: continue
        gamma_history.append(qg["gamma"]); mean_gamma=float(np.mean(gamma_history[:-1])) if len(gamma_history)>1 else qg["gamma"]
        mu=hist.mean(0); S=np.cov(hist,rowvar=False)+max(cfg.covariance_ridge*np.trace(np.cov(hist,rowvar=False))/hist.shape[1],1e-12)*np.eye(hist.shape[1])
        # Wasserstein radius: recursive scale, fixed rule, not tuned on OOS performance
        wass_radius=0.05*float(np.median(np.linalg.norm(hist-hist.mean(0),axis=1)))/math.sqrt(len(hist))
        global_gamma=qg["gamma"]
        wm={}
        wm["EqualWeight"]=np.ones(R.shape[1])/R.shape[1]
        wm["MinVariance"]=minvar_weights(hist,cfg.max_weight)
        wm["RiskParity"]=risk_parity_weights(hist,cfg.max_weight)
        wm["NominalCVaR"]=nominal_cvar_weights(hist,cfg.beta,cfg.max_weight)
        wm["GlobalKL"]=global_kl_cvar_weights(hist,cfg.beta,global_gamma,cfg.max_weight)
        wm["Wasserstein"]=wasserstein_cvar_weights(hist,cfg.beta,wass_radius,cfg.max_weight)
        sm,ss=static
        wm["A1_StaticLatentKL"],_=torch_latent_portfolio(mu,S,sm,ss,qg["gamma"],cfg,device,cfg.seed+kk)
        wm["A2_StateNominal"],_=torch_latent_portfolio(mu,S,qg["m"],qg["sigma"],0.0,cfg,device,cfg.seed+1000+kk)
        wm["A3_StateLatentKL"],obj=torch_latent_portfolio(mu,S,qg["m"],qg["sigma"],qg["gamma"],cfg,device,cfg.seed+2000+kk)
        wm["A4_StateQ_MeanGamma"],_=torch_latent_portfolio(mu,S,qg["m"],qg["sigma"],mean_gamma,cfg,device,cfg.seed+3000+kk)
        wm["A5_StaticQ_DateGamma"],_=torch_latent_portfolio(mu,S,sm,ss,qg["gamma"],cfg,device,cfg.seed+4000+kk)
        end=min(pos+H,len(R)); hold=simple_from_log(R.iloc[pos+1:end].to_numpy(float)); hold_dates=R.index[pos+1:end]
        for meth,w in wm.items():
            audit.check(f"weights {name} {date.date()} {meth}",abs(w.sum()-1)<1e-6 and np.min(w)>=-1e-9 and (cfg.max_weight is None or np.max(w)<=cfg.max_weight+1e-6),
                        f"sum={w.sum():.8f}, min={w.min():.4g}, max={w.max():.4g}")
            pre=prev_post.get(meth,np.ones_like(w)/len(w)); to=float(np.abs(w-pre).sum())
            cval=np.nan
            if exposure is not None and date in exposure.index and exposure.loc[date].notna().all(): cval=float(exposure.loc[date].to_numpy(float)@w)
            rows.append({"Universe":name,"Date":date,"Method":meth,"S":s0,"q_log_mean":qg["m"],"q_log_sigma":qg["sigma"],
                         "q_xi_mean":math.exp(qg["m"]+.5*qg["sigma"]**2),"q_xi_q95":math.exp(qg["m"]+qg["sigma"]*norm.ppf(.95)),
                         "Gamma":qg["gamma"],"B_valid":qg["B_valid"],"n_eff":qg["neff"],"MeanGammaPast":mean_gamma,
                         "Turnover":to,"Exposure":cval,"Objective":obj if meth=="A3_StateLatentKL" else np.nan})
            for a,ww in zip(R.columns,w): weights.append({"Universe":name,"Date":date,"Method":meth,"Asset":a,"Weight":ww,"S":s0})
            # daily buy-and-hold portfolio return approximation; exact drift is used for next pre-trade weights
            if len(hold):
                cur=w.copy()
                for dt,rr in zip(hold_dates,hold):
                    pr=float(cur@rr); daily.append({"Universe":name,"Date":dt,"FormationDate":date,"Method":meth,"Return":pr,"SFormation":s0})
                    cur=cur*(1+rr); cur=cur/cur.sum()
                prev_post[meth]=cur
    return pd.DataFrame(rows),pd.DataFrame(weights),pd.DataFrame(daily),xi

# -----------------------------------------------------------------------------
# Tables and figures
# -----------------------------------------------------------------------------
def state_quartiles(df):
    q=df["S"].quantile([.25,.75]);
    return np.where(df.S<=q.iloc[0],"Q1",np.where(df.S>=q.iloc[1],"Q4","Q2-Q3"))

def build_outputs(dec,wei,daily,blocks_by_u,cfg,out,audit):
    figs=out/"figures"; tabs=out/"tables"; csvd=out/"csv"; figs.mkdir(exist_ok=True);tabs.mkdir(exist_ok=True);csvd.mkdir(exist_ok=True)
    # F1
    f1=dec[dec.Method=="A3_StateLatentKL"][["Universe","Date","S","q_log_mean","q_log_sigma","q_xi_mean","q_xi_q95","Gamma","B_valid","n_eff"]].copy()
    f1.to_csv(csvd/"fig1_empirical_state_latent_mapping.csv",index=False)
    fig,axs=plt.subplots(2,2,figsize=(10,7),constrained_layout=True)
    for j,u in enumerate(["ETF","Firm"]):
        z=f1[f1.Universe==u].sort_values("S"); axs[0,j].scatter(z.S,z.q_xi_q95,s=16,alpha=.55); axs[0,j].set(title=f"{u}: latent upper quantile",xlabel="$S_t$",ylabel=r"$q_{0.95}(\widehat q_t)$")
        axs[1,j].scatter(z.S,z.Gamma,s=16,alpha=.55); axs[1,j].set(title=f"{u}: ambiguity calibration",xlabel="$S_t$",ylabel=r"$\widehat\Gamma_t$")
    fig.savefig(figs/"fig1_empirical_state_latent_mapping.pdf",bbox_inches="tight"); plt.close(fig)
    # T1 mechanism ablation, test period
    test_start=pd.Timestamp("2022-01-01"); dd=daily[daily.Date>=test_start]
    dtest=dec[dec.Date>=test_start]
    abl=["NominalCVaR","A1_StaticLatentKL","A2_StateNominal","A3_StateLatentKL","A4_StateQ_MeanGamma","A5_StaticQ_DateGamma"]
    t1=[]
    for (u,m),g in dd[dd.Method.isin(abl)].groupby(["Universe","Method"]):
        mt=metrics(g.Return.to_numpy(),dtest[(dtest.Universe==u)&(dtest.Method==m)].Turnover.to_numpy())
        dz=dtest[(dtest.Universe==u)&(dtest.Method==m)].copy(); dz["StateGroup"]=state_quartiles(dz)
        exq=dz.groupby("StateGroup").Exposure.mean(); mt.update({"Universe":u,"Method":m,"ExposureMean":dz.Exposure.mean(),"DeltaExposure_Q4_Q1":exq.get("Q4",np.nan)-exq.get("Q1",np.nan)})
        t1.append(mt)
    t1=pd.DataFrame(t1); t1.to_csv(tabs/"tab_empirical_mechanism_ablation.csv",index=False)
    # F2 source
    f2=t1.copy(); f2.to_csv(csvd/"fig2_empirical_mechanism_ablation.csv",index=False)
    fig,axs=plt.subplots(1,3,figsize=(12,3.6),constrained_layout=True)
    for ax,metric in zip(axs,["ES95","DeltaExposure_Q4_Q1","Turnover"]):
        for u,mk in [("ETF","o"),("Firm","s")]:
            z=f2[f2.Universe==u]; ax.plot(np.arange(len(z)),z[metric],marker=mk,label=u)
        ax.set_xticks(np.arange(len(z))); ax.set_xticklabels(z.Method,rotation=60,ha="right",fontsize=7); ax.set_title(metric)
    axs[0].legend(frameon=False); fig.savefig(figs/"fig2_empirical_mechanism_ablation.pdf",bbox_inches="tight"); plt.close(fig)
    # F3 + T2 block weights
    # Explicit per-universe quartile assignment avoids pandas GroupBy.apply
    # API differences and guarantees a many-to-one merge onto asset weights.
    w3 = wei.loc[wei.Method == "A3_StateLatentKL"].copy()
    qs = (
        dec.loc[dec.Method == "A3_StateLatentKL", ["Universe", "Date", "S"]]
        .drop_duplicates(subset=["Universe", "Date"])
        .copy()
    )
    tmp = []
    for u, g in qs.groupby("Universe", sort=False):
        h = g.copy()
        h["StateGroup"] = state_quartiles(h)
        tmp.append(h)
    if not tmp:
        raise RuntimeError("No A3_StateLatentKL observations available for state-dependent allocation analysis.")
    qs = pd.concat(tmp, ignore_index=True)
    w3 = w3.merge(
        qs[["Universe", "Date", "StateGroup"]],
        on=["Universe", "Date"],
        how="left",
        validate="many_to_one",
    )
    audit.check(
        "State groups assigned for A3 allocation",
        w3["StateGroup"].notna().all(),
        f"rows={len(w3)}, missing={int(w3['StateGroup'].isna().sum())}",
        fatal=True,
    )
    block_rows=[]
    for u,blocks in blocks_by_u.items():
        z=w3[w3.Universe==u]
        for (dt,sg),g in z.groupby(["Date","StateGroup"]):
            for b,assets in blocks.items(): block_rows.append({"Universe":u,"Date":dt,"StateGroup":sg,"Block":b,"Weight":g[g.Asset.isin(assets)].Weight.sum()})
    br=pd.DataFrame(block_rows); br.to_csv(csvd/"fig3_state_dependent_reallocation.csv",index=False)
    t2=br.groupby(["Universe","Block","StateGroup"],as_index=False).Weight.mean().pivot_table(index=["Universe","Block"],columns="StateGroup",values="Weight").reset_index()
    if "Q1" in t2 and "Q4" in t2:t2["Difference_Q4_Q1"]=t2.Q4-t2.Q1
    t2.to_csv(tabs/"tab_state_dependent_allocation_heterogeneity.csv",index=False)
    # F3 heatmap-like plots
    fig,axs=plt.subplots(1,2,figsize=(11,4.2),constrained_layout=True)
    for ax,u in zip(axs,["ETF","Firm"]):
        p=br[br.Universe==u].groupby(["Block","StateGroup"]).Weight.mean().unstack()
        im=ax.imshow(p[[c for c in ["Q1","Q2-Q3","Q4"] if c in p]],aspect="auto")
        ax.set_yticks(range(len(p)));ax.set_yticklabels(p.index,fontsize=8);ax.set_xticks(range(len(p.columns)));ax.set_xticklabels([c for c in ["Q1","Q2-Q3","Q4"] if c in p]);ax.set_title(u)
        for i in range(p.shape[0]):
            for j in range(p.shape[1]): ax.text(j,i,f"{p.iloc[i,j]:.2f}",ha="center",va="center",fontsize=7)
    fig.colorbar(im,ax=axs.ravel().tolist(),shrink=.8,label="Mean portfolio weight"); fig.savefig(figs/"fig3_state_dependent_reallocation.pdf",bbox_inches="tight");plt.close(fig)
    # T3 all benchmark performance + F4
    t3=[]
    for (u,m),g in dd.groupby(["Universe","Method"]):
        mt=metrics(g.Return.to_numpy(),dtest[(dtest.Universe==u)&(dtest.Method==m)].Turnover.to_numpy()); mt.update({"Universe":u,"Method":m});t3.append(mt)
    t3=pd.DataFrame(t3);t3.to_csv(tabs/"tab_out_of_sample_performance.csv",index=False);t3.to_csv(csvd/"fig4_tail_risk_performance_frontier.csv",index=False)
    fig,axs=plt.subplots(1,2,figsize=(10,4),constrained_layout=True)
    for ax,u in zip(axs,["ETF","Firm"]):
        z=t3[t3.Universe==u]
        ax.scatter(z.ES95,z.AnnReturn,s=35)
        for _,r in z.iterrows():ax.annotate(r.Method,(r.ES95,r.AnnReturn),fontsize=6,xytext=(3,3),textcoords="offset points")
        ax.set(title=u,xlabel="Out-of-sample ES (95%)",ylabel="Annualized return")
    fig.savefig(figs/"fig4_tail_risk_performance_frontier.pdf",bbox_inches="tight");plt.close(fig)
    return t1,t2,t3

# -----------------------------------------------------------------------------
# Placebos and robustness (compact reruns)
# -----------------------------------------------------------------------------
def placebo_summary(Rs,base_state,factors,cfg,device,audit,out):
    # Full re-estimation under competing states is expensive. We run the proposed A3 only
    # with identical construction and hyperparameters. Permutation is fixed ex ante by seed.
    records=[]
    states={"CT":base_state["S_CT"],"Market":construct_competing_state(factors,cfg.state_window,"market"),"Energy":construct_competing_state(factors,cfg.state_window,"energy")}
    rng=np.random.default_rng(cfg.seed); perm=base_state["S_CT"].copy(); vals=perm.dropna().to_numpy().copy(); rng.shuffle(vals); perm.loc[perm.notna()]=vals; states["Permuted"]=perm
    # Lightweight diagnostic: association of each state with next radial proxy. Portfolio placebo
    # is exported as a clearly labeled diagnostic if full rerun is not requested in --fast.
    for u,R in Rs.items():
        xi=trailing_radial_proxy(R,cfg)
        for sn,s in states.items():
            z=pd.concat([s.rename("S"),np.log(xi.shift(-1)).rename("logxi")],axis=1).dropna()
            if len(z):
                pr=pearsonr(z.S,z.logxi); sr=spearmanr(z.S,z.logxi)
                records.append({"Universe":u,"State":sn,"Pearson":pr.statistic,"Pearson_p":pr.pvalue,"Spearman":sr.statistic,"Spearman_p":sr.pvalue,"N":len(z)})
    f5=pd.DataFrame(records); f5.to_csv(out/"csv"/"fig5_competing_state_placebos.csv",index=False)
    fig,axs=plt.subplots(1,2,figsize=(9,3.7),constrained_layout=True)
    for ax,u in zip(axs,["ETF","Firm"]):
        z=f5[f5.Universe==u]; x=np.arange(len(z)); ax.bar(x,z.Spearman);ax.set_xticks(x);ax.set_xticklabels(z.State,rotation=30);ax.axhline(0,lw=.8);ax.set(title=u,ylabel="Spearman(state, next log radial proxy)")
    fig.savefig(out/"figures"/"fig5_competing_state_placebos.pdf",bbox_inches="tight");plt.close(fig)
    return f5

def robustness_from_daily(dec,daily,cfg,out):
    test=daily[daily.Date>=pd.Timestamp("2022-01-01")]; rows=[]
    # Transaction costs use turnover charged at formation; allocate cost to first daily return of holding period.
    for (u,m),g in test.groupby(["Universe","Method"]):
        ddec=dec[(dec.Universe==u)&(dec.Method==m)&(dec.Date>=pd.Timestamp("2022-01-01"))]
        for bps in (0,)+cfg.transaction_cost_bps:
            z=g.sort_values("Date").copy(); costmap=dict(zip(ddec.Date,ddec.Turnover*bps/10000.0))
            z["NetReturn"]=z.Return
            first=z.groupby("FormationDate").head(1).index
            z.loc[first,"NetReturn"]-=z.loc[first,"FormationDate"].map(costmap).fillna(0).to_numpy()
            mt=metrics(z.NetReturn); rows.append({"Universe":u,"Method":m,"CostBps":bps,**mt})
    t4=pd.DataFrame(rows);t4.to_csv(out/"tables"/"tab_empirical_robustness_inference.csv",index=False);t4.to_csv(out/"csv"/"fig6_economic_robustness.csv",index=False)
    fig,axs=plt.subplots(1,2,figsize=(10,4),constrained_layout=True)
    for ax,u in zip(axs,["ETF","Firm"]):
        z=t4[(t4.Universe==u)&(t4.Method.isin(["NominalCVaR","A1_StaticLatentKL","A3_StateLatentKL"]))]
        for m,g in z.groupby("Method"): ax.plot(g.CostBps,g.Sharpe,marker="o",label=m)
        ax.set(title=u,xlabel="Transaction cost (bps per unit turnover)",ylabel="Net Sharpe");ax.legend(fontsize=7,frameon=False)
    fig.savefig(out/"figures"/"fig6_economic_robustness.pdf",bbox_inches="tight");plt.close(fig)
    return t4

# -----------------------------------------------------------------------------
# V4 distributional mechanism diagnostics
# -----------------------------------------------------------------------------
def v4_distributional_diagnostics(Rs, state_df, factors, cfg, out):
    """Pre-specified diagnostics: full conditional-law/tail channel, horizons 1/5/20,
    and CT/Market/Energy/Permuted comparisons. Baseline V3 is never re-tuned.
    """
    from scipy.stats import ks_2samp
    rows=[]; tailrows=[]
    states={"CT":state_df["S_CT"],
            "Market":construct_competing_state(factors,cfg.state_window,"market"),
            "Energy":construct_competing_state(factors,cfg.state_window,"energy")}
    rng=np.random.default_rng(cfg.seed)
    perm=state_df["S_CT"].copy(); vv=perm.dropna().to_numpy().copy(); rng.shuffle(vv); perm.loc[perm.notna()]=vv
    states["Permuted"]=perm
    for u,R in Rs.items():
        xi=trailing_radial_proxy(R,cfg)
        for sn,S in states.items():
            for h in (1,5,20):
                # Future-only aggregates, available solely for ex-post validation.
                fut_mean=pd.concat([xi.shift(-j) for j in range(1,h+1)],axis=1).mean(axis=1)
                fut_max=pd.concat([xi.shift(-j) for j in range(1,h+1)],axis=1).max(axis=1)
                z=pd.concat([S.rename("S"),fut_mean.rename("XiMean"),fut_max.rename("XiMax")],axis=1).dropna()
                if len(z)<40: continue
                q1,q4=z.S.quantile([.25,.75]); lo=z[z.S<=q1]; hi=z[z.S>=q4]
                for target in ("XiMean","XiMax"):
                    a=np.log(np.clip(lo[target].to_numpy(float),1e-12,None)); b=np.log(np.clip(hi[target].to_numpy(float),1e-12,None))
                    ks=ks_2samp(a,b)
                    # Distributional summaries, not a monotonicity assumption.
                    rows.append({"Universe":u,"State":sn,"Horizon":h,"Target":target,"N":len(z),
                                 "MeanLog_Q1":a.mean(),"MeanLog_Q4":b.mean(),"DeltaMeanLog_Q4_Q1":b.mean()-a.mean(),
                                 "SDLog_Q1":a.std(ddof=1),"SDLog_Q4":b.std(ddof=1),"DeltaSDLog_Q4_Q1":b.std(ddof=1)-a.std(ddof=1),
                                 "Q90_Q1":np.quantile(lo[target],.90),"Q90_Q4":np.quantile(hi[target],.90),
                                 "DeltaQ90_Q4_Q1":np.quantile(hi[target],.90)-np.quantile(lo[target],.90),
                                 "Q95_Q1":np.quantile(lo[target],.95),"Q95_Q4":np.quantile(hi[target],.95),
                                 "DeltaQ95_Q4_Q1":np.quantile(hi[target],.95)-np.quantile(lo[target],.95),
                                 "KS":ks.statistic,"KS_p":ks.pvalue})
                # Tail exceedance defined recursively from information available at t.
                vals=xi.to_numpy(float); idx=xi.index; ev=[]
                for dt in z.index:
                    pos=idx.get_indexer([dt])[0]
                    hist=vals[max(0,pos-cfg.radial_window):pos]; hist=hist[np.isfinite(hist)]
                    if len(hist)>=cfg.radial_min_obs:
                        thr90=np.quantile(hist,.90); thr95=np.quantile(hist,.95)
                        ev.append((dt,float(z.loc[dt,"XiMax"]>thr90),float(z.loc[dt,"XiMax"]>thr95)))
                if ev:
                    e=pd.DataFrame(ev,columns=["Date","Tail90","Tail95"]).set_index("Date").join(z[["S"]])
                    q1e,q4e=e.S.quantile([.25,.75])
                    for y in ("Tail90","Tail95"):
                        p1=e.loc[e.S<=q1e,y].mean(); p4=e.loc[e.S>=q4e,y].mean()
                        tailrows.append({"Universe":u,"State":sn,"Horizon":h,"Tail":y,"N":len(e),
                                         "Prob_Q1":p1,"Prob_Q4":p4,"DeltaProb_Q4_Q1":p4-p1})
    d=pd.DataFrame(rows); t=pd.DataFrame(tailrows)
    d.to_csv(out/"tables"/"tab_v4_distributional_state_diagnostics.csv",index=False)
    t.to_csv(out/"tables"/"tab_v4_tail_event_diagnostics.csv",index=False)
    return d,t

# -----------------------------------------------------------------------------
# V5 portfolio falsification: CT vs Market vs Energy vs Permuted
# -----------------------------------------------------------------------------
def _state_family(state_df, factors, cfg):
    """Construct the four pre-specified states once, with no outcome-based tuning."""
    states = {
        "CT": state_df["S_CT"].copy(),
        "Market": construct_competing_state(factors, cfg.state_window, "market"),
        "Energy": construct_competing_state(factors, cfg.state_window, "energy"),
    }
    rng = np.random.default_rng(cfg.seed)
    perm = state_df["S_CT"].copy()
    vals = perm.dropna().to_numpy().copy()
    rng.shuffle(vals)
    perm.loc[perm.notna()] = vals
    states["Permuted"] = perm
    return states


def _decision_positions_for_state(R, S, xi, cfg):
    """Same admissibility rule and H=20 calendar logic as the baseline experiment."""
    st = S.reindex(R.index)
    valid = np.where(np.isfinite(st.to_numpy(float)) & np.isfinite(xi.shift(1).to_numpy(float)))[0]
    if len(valid) == 0:
        return np.array([], dtype=int)
    start = max(valid[0], cfg.radial_min_obs + 1)
    pos = np.arange(start, len(R)-1, cfg.rebalance_H)
    pos = pos[R.index[pos] <= pd.Timestamp(cfg.test_end)]
    if cfg.fast:
        pos = pos[-min(12, len(pos)):]
    return pos


def run_v5_a3_for_state(universe, R, S, state_name, cfg, device, audit):
    """Re-estimate the proposed A3 rule under one state, changing only the state source.

    All remaining ingredients are held fixed: radial proxy, bandwidth, bootstrap rule,
    beta, H, covariance estimator, feasible set, optimizer and random-seed policy.
    """
    S = S.reindex(R.index)
    xi = trailing_radial_proxy(R, cfg)
    positions = _decision_positions_for_state(R, S, xi, cfg)
    rng = np.random.default_rng(cfg.seed + (101 if universe == "ETF" else 202) +
                                {"CT":0,"Market":10000,"Energy":20000,"Permuted":30000}[state_name])
    dec_rows, weight_rows, daily_rows = [], [], []
    prev_post = None
    for kk, pos in enumerate(positions):
        date = R.index[pos]
        s0 = float(S.loc[date])
        admiss = R.index < date
        qg = estimate_q_gamma(S.loc[admiss].to_numpy(), xi.loc[admiss].to_numpy(),
                              s0, cfg, rng, B=(60 if cfg.fast else cfg.bootstrap_B))
        if qg is None:
            continue
        hist = R.iloc[max(0,pos-cfg.radial_window):pos].to_numpy(float)
        mu = hist.mean(0)
        Ssam = np.cov(hist, rowvar=False)
        ridge = max(cfg.covariance_ridge*np.trace(Ssam)/hist.shape[1], 1e-12)
        Sigma = Ssam + ridge*np.eye(hist.shape[1])
        # Same A3 optimizer and seed convention across state families: common random numbers.
        w, obj = torch_latent_portfolio(mu, Sigma, qg["m"], qg["sigma"], qg["gamma"],
                                        cfg, device, cfg.seed + 2000 + kk)
        ok = abs(w.sum()-1) < 1e-6 and np.min(w) >= -1e-9 and \
             (cfg.max_weight is None or np.max(w) <= cfg.max_weight+1e-6)
        audit.check(f"V5 weights {universe} {state_name} {date.date()}", ok,
                    f"sum={w.sum():.8f}, min={w.min():.4g}, max={w.max():.4g}", fatal=True)
        turnover = float(np.abs(w-(prev_post if prev_post is not None else np.ones_like(w)/len(w))).sum())
        dec_rows.append({"Universe":universe,"State":state_name,"Date":date,"S":s0,
                         "Gamma":qg["gamma"],"q_log_mean":qg["m"],"q_log_sigma":qg["sigma"],
                         "n_eff":qg["neff"],"B_valid":qg["B_valid"],"Turnover":turnover,
                         "Objective":obj})
        for a, ww in zip(R.columns, w):
            weight_rows.append({"Universe":universe,"State":state_name,"Date":date,
                                "S":s0,"Asset":a,"Weight":ww})
        end = min(pos+cfg.rebalance_H, len(R))
        hold = simple_from_log(R.iloc[pos+1:end].to_numpy(float))
        hold_dates = R.index[pos+1:end]
        cur = w.copy()
        for dt, rr in zip(hold_dates, hold):
            pr = float(cur@rr)
            daily_rows.append({"Universe":universe,"State":state_name,"Date":dt,
                               "FormationDate":date,"Return":pr,"SFormation":s0})
            cur = cur*(1+rr)
            cur = cur/cur.sum()
        if len(hold):
            prev_post = cur
    return pd.DataFrame(dec_rows), pd.DataFrame(weight_rows), pd.DataFrame(daily_rows)


def _mbb_diff_ci(x, y, B=1000, block=20, level=.95, seed=0):
    """Moving/circular-block bootstrap CI for difference in means, resampling each group."""
    x=np.asarray(x,float); y=np.asarray(y,float)
    x=x[np.isfinite(x)]; y=y[np.isfinite(y)]
    if len(x)<5 or len(y)<5: return np.nan,np.nan
    rng=np.random.default_rng(seed); vals=np.empty(B)
    for b in range(B):
        ix=circular_block_indices(len(x),min(block,len(x)),rng)
        iy=circular_block_indices(len(y),min(block,len(y)),rng)
        vals[b]=np.mean(y[iy])-np.mean(x[ix])
    a=(1-level)/2
    return float(np.quantile(vals,a)),float(np.quantile(vals,1-a))


def v5_portfolio_falsification(Rs, state_df, factors, blocks_by_u, cfg, device, audit, out):
    """Decisive falsification experiment: identical A3 under CT/Market/Energy/Permuted."""
    states=_state_family(state_df,factors,cfg)
    alld=[]; allw=[]; allr=[]
    for u,R in Rs.items():
        for sn,S in states.items():
            d,w,r=run_v5_a3_for_state(u,R,S,sn,cfg,device,audit)
            audit.check(f"V5 A3 outputs {u} {sn}",len(d)>0 and len(w)>0 and len(r)>0,
                        f"decisions={len(d)}, weights={len(w)}, daily={len(r)}",fatal=True)
            alld.append(d); allw.append(w); allr.append(r)
    dec=pd.concat(alld,ignore_index=True); wei=pd.concat(allw,ignore_index=True); daily=pd.concat(allr,ignore_index=True)
    csvd=out/'csv'; tabs=out/'tables'; figs=out/'figures'
    dec.to_csv(csvd/'v5_state_placebo_decisions.csv',index=False)
    wei.to_csv(csvd/'v5_state_placebo_weights.csv',index=False)
    daily.to_csv(csvd/'v5_state_placebo_daily_returns.csv',index=False)

    # Formation-date block allocation and Q1/Q4 signatures.
    block=[]
    for u,blocks in blocks_by_u.items():
        z=wei[wei.Universe==u]
        for sn in states:
            zz=z[z.State==sn].copy()
            dates=dec[(dec.Universe==u)&(dec.State==sn)][['Date','S']].drop_duplicates('Date')
            if dates.empty: continue
            q=dates.S.quantile([.25,.75]); dates['StateGroup']=np.where(dates.S<=q.iloc[0],'Q1',np.where(dates.S>=q.iloc[1],'Q4','Q2-Q3'))
            zz=zz.merge(dates[['Date','StateGroup']],on='Date',how='left',validate='many_to_one')
            for (dt,sg),g in zz.groupby(['Date','StateGroup']):
                for b,assets in blocks.items():
                    block.append({'Universe':u,'State':sn,'Date':dt,'StateGroup':sg,'Block':b,
                                  'Weight':float(g[g.Asset.isin(assets)].Weight.sum())})
    br=pd.DataFrame(block); br.to_csv(csvd/'fig_v5_state_specific_allocation.csv',index=False)

    alloc=[]
    for (u,sn,b),g in br.groupby(['Universe','State','Block']):
        q1=g.loc[g.StateGroup=='Q1','Weight'].to_numpy(); q4=g.loc[g.StateGroup=='Q4','Weight'].to_numpy()
        lo,hi=_mbb_diff_ci(q1,q4,cfg.n_ci_boot,cfg.ci_block,cfg.ci_level,
                           cfg.seed+sum(map(ord,u+sn+b)))
        alloc.append({'Universe':u,'State':sn,'Block':b,'Mean_Q1':np.mean(q1) if len(q1) else np.nan,
                      'Mean_Q4':np.mean(q4) if len(q4) else np.nan,
                      'Delta_Q4_Q1':(np.mean(q4)-np.mean(q1)) if len(q1) and len(q4) else np.nan,
                      'CI_low':lo,'CI_high':hi,'N_Q1':len(q1),'N_Q4':len(q4)})
    ta=pd.DataFrame(alloc); ta.to_csv(tabs/'tab_v5_state_specific_allocation.csv',index=False)

    # Q1/Q4 realized ES and turnover. ES is computed from daily returns whose formation date belongs to each group.
    risk=[]
    for (u,sn),gd in dec.groupby(['Universe','State']):
        dates=gd[['Date','S','Turnover']].drop_duplicates('Date').copy(); q=dates.S.quantile([.25,.75])
        dates['StateGroup']=np.where(dates.S<=q.iloc[0],'Q1',np.where(dates.S>=q.iloc[1],'Q4','Q2-Q3'))
        rr=daily[(daily.Universe==u)&(daily.State==sn)].merge(dates[['Date','StateGroup']].rename(columns={'Date':'FormationDate'}),on='FormationDate',how='left')
        for sg in ('Q1','Q4'):
            x=rr.loc[rr.StateGroup==sg,'Return'].to_numpy(float); loss=-x
            var=np.quantile(loss,.95) if len(loss) else np.nan
            es=loss[loss>=var].mean() if len(loss) and np.any(loss>=var) else np.nan
            to=dates.loc[dates.StateGroup==sg,'Turnover'].mean()
            risk.append({'Universe':u,'State':sn,'StateGroup':sg,'ES95':es,'Turnover':to,'N_daily':len(x)})
    tr=pd.DataFrame(risk)
    wide=tr.pivot(index=['Universe','State'],columns='StateGroup',values=['ES95','Turnover']).reset_index()
    wide.columns=['_'.join([str(y) for y in x if str(y)]) if isinstance(x,tuple) else x for x in wide.columns]
    if 'ES95_Q4' in wide and 'ES95_Q1' in wide: wide['DeltaES95_Q4_Q1']=wide.ES95_Q4-wide.ES95_Q1
    if 'Turnover_Q4' in wide and 'Turnover_Q1' in wide: wide['DeltaTurnover_Q4_Q1']=wide.Turnover_Q4-wide.Turnover_Q1
    wide.to_csv(tabs/'tab_v5_state_specific_risk_turnover.csv',index=False)

    # Compact manuscript-facing figure: fossil and utilities signatures with bootstrap CIs.
    sel=ta[ta.Block.isin(['Fossil energy','Utilities','Electricity and utilities'])].copy()
    fig,axs=plt.subplots(1,2,figsize=(11,4.3),constrained_layout=True)
    for ax,u in zip(axs,['ETF','Firm']):
        z=sel[sel.Universe==u].copy(); labels=[]; xs=[]; ys=[]; elo=[]; ehi=[]
        for _,r in z.iterrows():
            labels.append(f"{r.State}\n{r.Block}"); xs.append(len(xs)); ys.append(r.Delta_Q4_Q1)
            elo.append(r.Delta_Q4_Q1-r.CI_low); ehi.append(r.CI_high-r.Delta_Q4_Q1)
        if xs:
            ax.errorbar(xs,ys,yerr=np.vstack([elo,ehi]),fmt='o',capsize=3)
            ax.axhline(0,lw=.8); ax.set_xticks(xs); ax.set_xticklabels(labels,rotation=55,ha='right',fontsize=7)
        ax.set_title(u); ax.set_ylabel(r'$Q4-Q1$ mean block weight')
    fig.savefig(figs/'fig_v5_state_specific_portfolio_falsification.pdf',bbox_inches='tight'); plt.close(fig)
    audit.check('V5 portfolio falsification complete',len(ta)>0 and len(wide)>0,
                f'allocation_rows={len(ta)}, risk_rows={len(wide)}',fatal=True)
    return dec,wei,daily,ta,wide

# -----------------------------------------------------------------------------
# V6 final inference: temporally coherent randomization + CT-vs-Energy contrast
# -----------------------------------------------------------------------------
def _rotation_stat_from_weights(wei_u, state_series, blocks, dates=None):
    """Fossil -> utilities rotation statistic T_FU = Delta utilities - Delta fossil.

    The portfolio path is held fixed. State labels are used only to define Q1/Q4 dates.
    This is a randomization test of state/allocation alignment, not a re-fit test.
    """
    z=wei_u.copy(); z['Date']=pd.to_datetime(z['Date'])
    ss=state_series.copy(); ss.index=pd.to_datetime(ss.index)
    if dates is None: dates=np.sort(z['Date'].unique())
    d=pd.DataFrame({'Date':pd.to_datetime(dates)})
    d['S']=d['Date'].map(ss)
    d=d[np.isfinite(pd.to_numeric(d['S'],errors='coerce'))].copy()
    if len(d)<20: return np.nan, np.nan, np.nan, len(d)
    q1,q4=d['S'].quantile([.25,.75]).to_numpy()
    d['G']=np.where(d.S<=q1,'Q1',np.where(d.S>=q4,'Q4','MID'))
    zz=z.merge(d[['Date','G']],on='Date',how='inner',validate='many_to_one')
    foss=blocks.get('Fossil energy',[])
    util=blocks.get('Utilities',blocks.get('Electricity and utilities',[]))
    by=[]
    for (dt,gp),g in zz.groupby(['Date','G']):
        if gp not in ('Q1','Q4'): continue
        wf=float(g.loc[g.Asset.isin(foss),'Weight'].sum())
        wu=float(g.loc[g.Asset.isin(util),'Weight'].sum())
        by.append((dt,gp,wf,wu))
    b=pd.DataFrame(by,columns=['Date','G','Fossil','Utilities'])
    if b.empty or not {'Q1','Q4'}.issubset(set(b.G)): return np.nan,np.nan,np.nan,len(d)
    m=b.groupby('G')[['Fossil','Utilities']].mean()
    df=float(m.loc['Q4','Fossil']-m.loc['Q1','Fossil'])
    du=float(m.loc['Q4','Utilities']-m.loc['Q1','Utilities'])
    return du-df,df,du,len(d)


def _circular_shift_null(wei_ct, state_ct, blocks, B=500, min_shift=20, seed=0):
    """Circular-shift randomization preserving the complete marginal distribution and serial ordering of S."""
    dates=np.sort(pd.to_datetime(wei_ct['Date'].unique()))
    ss=state_ct.reindex(pd.to_datetime(dates)).astype(float)
    ok=np.isfinite(ss.to_numpy())
    dates=np.asarray(dates)[ok]; vals=ss.to_numpy()[ok]
    if len(vals)<40: raise RuntimeError('Too few dated CT states for V6 randomization.')
    obs_series=pd.Series(vals,index=pd.to_datetime(dates))
    tobs,dfobs,duobs,nobs=_rotation_stat_from_weights(wei_ct,obs_series,blocks,dates)
    rng=np.random.default_rng(seed)
    allowed=np.arange(max(1,min_shift),len(vals)-max(1,min_shift))
    if len(allowed)==0: allowed=np.arange(1,len(vals))
    shifts=rng.choice(allowed,size=B,replace=(B>len(allowed)))
    rows=[]
    for b,k in enumerate(shifts,1):
        pv=np.roll(vals,int(k))
        ps=pd.Series(pv,index=pd.to_datetime(dates))
        t,df,du,_=_rotation_stat_from_weights(wei_ct,ps,blocks,dates)
        rows.append({'Permutation':b,'Shift':int(k),'T_FU':t,'DeltaFossil':df,'DeltaUtilities':du})
    null=pd.DataFrame(rows)
    good=null.T_FU[np.isfinite(null.T_FU)].to_numpy()
    # Pre-specified one-sided alternative: stronger fossil-to-utilities rotation.
    p_one=(1+np.sum(good>=tobs))/(1+len(good)) if len(good) else np.nan
    p_two=(1+np.sum(np.abs(good)>=abs(tobs)))/(1+len(good)) if len(good) else np.nan
    return {'T_FU_obs':tobs,'DeltaFossil_obs':dfobs,'DeltaUtilities_obs':duobs,
            'p_perm_one_sided':float(p_one),'p_perm_two_sided':float(p_two),
            'B_valid':len(good),'N_dates':nobs,
            'NullMean':float(np.mean(good)) if len(good) else np.nan,
            'NullQ025':float(np.quantile(good,.025)) if len(good) else np.nan,
            'NullQ975':float(np.quantile(good,.975)) if len(good) else np.nan},null


def _paired_block_ci_difference(x, y, B=2000, block=20, level=.95, seed=0):
    """Paired circular-block bootstrap CI for mean(x-y), preserving date pairing."""
    x=np.asarray(x,float); y=np.asarray(y,float); ok=np.isfinite(x)&np.isfinite(y); d=x[ok]-y[ok]
    if len(d)<8: return np.nan,np.nan,np.nan
    rng=np.random.default_rng(seed); vals=np.empty(B)
    for b in range(B):
        ii=circular_block_indices(len(d),min(block,len(d)),rng); vals[b]=np.mean(d[ii])
    a=(1-level)/2
    return float(np.mean(d)),float(np.quantile(vals,a)),float(np.quantile(vals,1-a))


def _dated_rotation_series(wei, state_series, blocks):
    """Date-level utility-minus-fossil allocation spread, aligned to state dates."""
    z=wei.copy(); z['Date']=pd.to_datetime(z.Date)
    foss=blocks.get('Fossil energy',[]); util=blocks.get('Utilities',blocks.get('Electricity and utilities',[]))
    rows=[]
    for dt,g in z.groupby('Date'):
        rows.append({'Date':dt,'Spread':float(g[g.Asset.isin(util)].Weight.sum()-g[g.Asset.isin(foss)].Weight.sum())})
    a=pd.DataFrame(rows).set_index('Date').sort_index(); a.index=pd.to_datetime(a.index)
    ss=state_series.copy(); ss.index=pd.to_datetime(ss.index)
    a['S']=ss.reindex(a.index).to_numpy()
    return a


def v6_final_inference(v5_wei, state_df, factors, blocks_by_u, cfg, audit, out, Bperm=500):
    """Final, pre-specified inference without further model tuning.

    (i) Circular-shift randomization of CT labels against the already estimated CT A3 portfolio path.
        This preserves the serial ordering and empirical distribution of the CT state and asks whether
        the observed Fossil->Utilities rotation is unusual under temporal misalignment.
    (ii) Paired circular-block bootstrap of the CT-vs-Energy rotation contrast on common dates.

    No q, Gamma, optimizer, bandwidth, state definition, or portfolio constraint is altered here.
    """
    tabs=out/'tables'; csvd=out/'csv'; figs=out/'figures'; tabs.mkdir(exist_ok=True);csvd.mkdir(exist_ok=True);figs.mkdir(exist_ok=True)
    states=_state_family(state_df,factors,cfg)
    summary=[]; nulls=[]; contrasts=[]
    for u in ['ETF','Firm']:
        blocks=blocks_by_u[u]
        wct=v5_wei[(v5_wei.Universe==u)&(v5_wei.State=='CT')].copy()
        wen=v5_wei[(v5_wei.Universe==u)&(v5_wei.State=='Energy')].copy()
        audit.check(f'V6 CT/Energy weights {u}',len(wct)>0 and len(wen)>0,
                    f'CT={len(wct)}, Energy={len(wen)}',fatal=True)
        sm,null=_circular_shift_null(wct,states['CT'],blocks,B=Bperm,min_shift=cfg.ci_block,
                                     seed=cfg.seed+(6101 if u=='ETF' else 6202))
        sm['Universe']=u; summary.append(sm); null['Universe']=u; nulls.append(null)

        # Direct CT-vs-Energy contrast in the pre-specified Q4-Q1 rotation statistic.
        tct,_,_,_=_rotation_stat_from_weights(wct,states['CT'],blocks)
        ten,_,_,_=_rotation_stat_from_weights(wen,states['Energy'],blocks)
        # Date-level paired spread difference CI: CT and Energy portfolio paths on common dates.
        sct=_dated_rotation_series(wct,states['CT'],blocks).rename(columns={'Spread':'CTSpread','S':'CTS'})
        sen=_dated_rotation_series(wen,states['Energy'],blocks).rename(columns={'Spread':'EnergySpread','S':'EnergyS'})
        m=sct.join(sen,how='inner')
        # Compare high-vs-low state rotation differences via bootstrap over dates within each state rule.
        def dated_delta(a,scol,vcol):
            q1,q4=a[scol].quantile([.25,.75]); lo=a.loc[a[scol]<=q1,vcol].to_numpy(); hi=a.loc[a[scol]>=q4,vcol].to_numpy(); return lo,hi
        ctl,cth=dated_delta(m,'CTS','CTSpread'); enl,enh=dated_delta(m,'EnergyS','EnergySpread')
        rng=np.random.default_rng(cfg.seed+(6303 if u=='ETF' else 6404)); vals=[]
        for b in range(max(1000,cfg.n_ci_boot)):
            if min(len(ctl),len(cth),len(enl),len(enh)) == 0:
                raise RuntimeError(f'Empty V6 quartile sample for {u}: CT_Q1={len(ctl)}, CT_Q4={len(cth)}, Energy_Q1={len(enl)}, Energy_Q4={len(enh)}')
            i1=circular_block_indices(len(ctl),min(cfg.ci_block,len(ctl)),rng); i4=circular_block_indices(len(cth),min(cfg.ci_block,len(cth)),rng)
            j1=circular_block_indices(len(enl),min(cfg.ci_block,len(enl)),rng); j4=circular_block_indices(len(enh),min(cfg.ci_block,len(enh)),rng)
            vals.append((np.mean(cth[i4])-np.mean(ctl[i1]))-(np.mean(enh[j4])-np.mean(enl[j1])))
        vals=np.asarray(vals); a=(1-cfg.ci_level)/2
        contrasts.append({'Universe':u,'T_FU_CT':tct,'T_FU_Energy':ten,'CT_minus_Energy':tct-ten,
                          'CI_low':float(np.quantile(vals,a)),'CI_high':float(np.quantile(vals,1-a)),
                          'B_boot':len(vals)})
    ts=pd.DataFrame(summary); tn=pd.concat(nulls,ignore_index=True); tc=pd.DataFrame(contrasts)
    ts.to_csv(tabs/'tab_v6_ct_rotation_randomization.csv',index=False)
    tc.to_csv(tabs/'tab_v6_ct_vs_energy_contrast.csv',index=False)
    tn.to_csv(csvd/'v6_ct_rotation_null_distribution.csv',index=False)

    # Manuscript-facing null distribution figure.
    fig,axs=plt.subplots(1,2,figsize=(10.5,4.2),constrained_layout=True)
    for ax,u in zip(axs,['ETF','Firm']):
        z=tn[tn.Universe==u].T_FU.dropna().to_numpy(); obs=float(ts.loc[ts.Universe==u,'T_FU_obs'].iloc[0])
        ax.hist(100*z,bins=30,alpha=.75); ax.axvline(100*obs,linestyle='--',linewidth=2)
        p=float(ts.loc[ts.Universe==u,'p_perm_one_sided'].iloc[0]); ax.set_title(f'{u}: p_perm={p:.3f}')
        ax.set_xlabel(r'$T_{FU}$ (percentage points)'); ax.set_ylabel('Randomization frequency')
    fig.savefig(figs/'fig_v6_ct_rotation_randomization.pdf',bbox_inches='tight'); plt.close(fig)
    audit.check('V6 final randomization inference',len(ts)==2 and (ts.B_valid>0).all(),
                f'rows={len(ts)}, B_valid={ts.B_valid.tolist()}',fatal=True)
    return ts,tc,tn

# -----------------------------------------------------------------------------
# V7 final identification/inference — specifications frozen after V5
# -----------------------------------------------------------------------------
def _es95(x):
    x=np.asarray(x,float); x=x[np.isfinite(x)]
    if len(x)==0: return np.nan
    loss=-x; q=np.quantile(loss,.95); tail=loss[loss>=q]
    return float(np.mean(tail)) if len(tail) else float(q)


def _paired_es_contrast(daily, universe, comparator, cfg, B=2000):
    """ES(CT)-ES(comparator) on exactly the same realized market dates.

    Formation dates are allowed to differ because competing state constructions can
    generate offset decision calendars. Pairing on the realized return date is the
    economically relevant control: both portfolio rules face the same market return
    realization on each paired day. Circular-block bootstrap resamples paired dates jointly.
    """
    a=daily[(daily.Universe==universe)&(daily.State=='CT')][['Date','Return']].copy()
    b=daily[(daily.Universe==universe)&(daily.State==comparator)][['Date','Return']].copy()
    for z in (a,b):
        z['Date']=pd.to_datetime(z['Date'])
        z.sort_values('Date',inplace=True)
        z.drop_duplicates('Date',keep='last',inplace=True)
    m=a.merge(b,on='Date',suffixes=('_CT','_Comp'),how='inner',
              validate='one_to_one').sort_values('Date')
    if len(m)<40:
        return None
    x=m.Return_CT.to_numpy(float); y=m.Return_Comp.to_numpy(float)
    obs=_es95(x)-_es95(y)
    rng=np.random.default_rng(cfg.seed+sum(map(ord,universe+comparator))+7100)
    vals=[]
    for _ in range(B):
        ii=circular_block_indices(len(m),min(cfg.ci_block,len(m)),rng)
        vals.append(_es95(x[ii])-_es95(y[ii]))
    vals=np.asarray(vals,float); vals=vals[np.isfinite(vals)]
    if len(vals)==0:
        return None
    aa=(1-cfg.ci_level)/2
    return {'Universe':universe,'Comparator':comparator,'PairingMode':'realized_date',
            'N_paired_daily':len(m),
            'ES95_CT':_es95(x),'ES95_Comparator':_es95(y),
            'DeltaES95_CT_minus_Comparator':obs,
            'CI_low':float(np.quantile(vals,aa)),
            'CI_high':float(np.quantile(vals,1-aa)),
            'B_boot':len(vals)}


def _spread_table(wei, universe, state_name, state_series, blocks, common_dates):
    z=wei[(wei.Universe==universe)&(wei.State==state_name)].copy(); z['Date']=pd.to_datetime(z.Date)
    foss=blocks.get('Fossil energy',[]); util=blocks.get('Utilities',blocks.get('Electricity and utilities',[]))
    rows=[]
    for dt,g in z[z.Date.isin(common_dates)].groupby('Date'):
        rows.append({'Date':dt,'Spread':float(g.loc[g.Asset.isin(util),'Weight'].sum()-g.loc[g.Asset.isin(foss),'Weight'].sum())})
    a=pd.DataFrame(rows, columns=['Date','Spread'])
    if a.empty:
        return pd.DataFrame(columns=['Date','Spread','S','G'])
    ss=state_series.copy(); ss.index=pd.to_datetime(ss.index)
    a['S']=a['Date'].map(ss); a=a.dropna(subset=['S']).sort_values('Date')
    q1,q4=a.S.quantile([.25,.75]); a['G']=np.where(a.S<=q1,'Q1',np.where(a.S>=q4,'Q4','MID'))
    return a


def _rotation_from_spread_table(a):
    if a.empty or not {'Q1','Q4'}.issubset(set(a.G)): return np.nan
    return float(a.loc[a.G=='Q4','Spread'].mean()-a.loc[a.G=='Q1','Spread'].mean())


def _paired_rotation_contrast(v5wei, universe, comparator, state_ct, state_comp,
                              blocks, cfg, B=2000):
    """Contrast T_FU(CT)-T_FU(comparator).

    Exact common formation dates are used when sufficiently available. If competing
    state constructions induce offset decision calendars, the function falls back to
    a transparent independent circular-block bootstrap over each state's own ordered
    decision dates. This avoids inventing nearest-date matches and records PairingMode
    in the exported table.
    """
    dct=set(pd.to_datetime(v5wei[(v5wei.Universe==universe)&
                                 (v5wei.State=='CT')].Date.unique()))
    dco=set(pd.to_datetime(v5wei[(v5wei.Universe==universe)&
                                 (v5wei.State==comparator)].Date.unique()))
    common=sorted(dct & dco)

    a=_spread_table(v5wei,universe,'CT',state_ct,blocks,
                    sorted(dct))
    b=_spread_table(v5wei,universe,comparator,state_comp,blocks,
                    sorted(dco))
    if a.empty or b.empty:
        return None

    # Preferred exact-date pairing.
    ac=a[['Date','Spread','G']].copy()
    bc=b[['Date','Spread','G']].copy()
    m=ac.merge(bc,on='Date',suffixes=('_CT','_Comp'),how='inner').sort_values('Date')

    def stat_paired(mm):
        if mm.empty: return np.nan
        if not {'Q1','Q4'}.issubset(set(mm.G_CT)) or \
           not {'Q1','Q4'}.issubset(set(mm.G_Comp)):
            return np.nan
        tct=mm.loc[mm.G_CT=='Q4','Spread_CT'].mean()-mm.loc[mm.G_CT=='Q1','Spread_CT'].mean()
        tco=mm.loc[mm.G_Comp=='Q4','Spread_Comp'].mean()-mm.loc[mm.G_Comp=='Q1','Spread_Comp'].mean()
        return float(tct-tco)

    obs_pair=stat_paired(m)
    rng=np.random.default_rng(cfg.seed+sum(map(ord,universe+comparator))+7200)
    vals=[]

    if len(m)>=40 and np.isfinite(obs_pair):
        for _ in range(B):
            ii=circular_block_indices(len(m),min(cfg.ci_block,len(m)),rng)
            v=stat_paired(m.iloc[ii])
            if np.isfinite(v): vals.append(v)
        mode='exact_common_formation_date'
        obs=obs_pair
        n_common=len(m)
    else:
        # Calendars are offset: compare the two pre-specified state-specific
        # T_FU statistics without fabricating date matches.
        tct=_rotation_from_spread_table(a)
        tco=_rotation_from_spread_table(b)
        if not np.isfinite(tct) or not np.isfinite(tco):
            return None
        obs=float(tct-tco)
        for _ in range(B):
            ia=circular_block_indices(len(a),min(cfg.ci_block,len(a)),rng)
            ib=circular_block_indices(len(b),min(cfg.ci_block,len(b)),rng)
            va=_rotation_from_spread_table(a.iloc[ia].copy())
            vb=_rotation_from_spread_table(b.iloc[ib].copy())
            if np.isfinite(va) and np.isfinite(vb):
                vals.append(float(va-vb))
        mode='independent_decision_date_blocks'
        n_common=len(m)

    vals=np.asarray(vals,float); vals=vals[np.isfinite(vals)]
    if len(vals)==0:
        return None
    aa=(1-cfg.ci_level)/2
    return {'Universe':universe,'Comparator':comparator,
            'PairingMode':mode,'N_common_dates':n_common,
            'N_CT_dates':len(a),'N_Comparator_dates':len(b),
            'CT_minus_Comparator':obs,
            'CI_low':float(np.quantile(vals,aa)),
            'CI_high':float(np.quantile(vals,1-aa)),
            'B_valid':len(vals)}


def _joint_ct_shift_test(v5wei, state_ct, blocks_by_u, cfg, B=1000):
    """Common circular shifts across ETF/Firm; joint statistic is the pre-specified equal-weight mean T_FU."""
    date_sets=[]
    for u in ('ETF','Firm'):
        date_sets.append(set(pd.to_datetime(v5wei[(v5wei.Universe==u)&(v5wei.State=='CT')].Date.unique())))
    dates=np.array(sorted(date_sets[0]&date_sets[1]),dtype='datetime64[ns]')
    ss=state_ct.copy(); ss.index=pd.to_datetime(ss.index); vals=ss.reindex(pd.to_datetime(dates)).to_numpy(float)
    ok=np.isfinite(vals); dates=dates[ok]; vals=vals[ok]
    if len(vals)<40: raise RuntimeError('Too few common CT dates for V7 joint randomization.')
    obs_s=pd.Series(vals,index=pd.to_datetime(dates)); obs=[]
    for u in ('ETF','Firm'):
        w=v5wei[(v5wei.Universe==u)&(v5wei.State=='CT')]
        obs.append(_rotation_stat_from_weights(w,obs_s,blocks_by_u[u],dates)[0])
    tobs=float(np.mean(obs)); rng=np.random.default_rng(cfg.seed+7300)
    allowed=np.arange(max(1,cfg.ci_block),len(vals)-max(1,cfg.ci_block));
    if len(allowed)==0: allowed=np.arange(1,len(vals))
    shifts=rng.choice(allowed,size=B,replace=(B>len(allowed))); rows=[]
    for ib,k in enumerate(shifts,1):
        ps=pd.Series(np.roll(vals,int(k)),index=pd.to_datetime(dates)); tt=[]
        for u in ('ETF','Firm'):
            w=v5wei[(v5wei.Universe==u)&(v5wei.State=='CT')]
            tt.append(_rotation_stat_from_weights(w,ps,blocks_by_u[u],dates)[0])
        rows.append({'Permutation':ib,'Shift':int(k),'T_ETF':tt[0],'T_Firm':tt[1],'T_joint':float(np.mean(tt))})
    null=pd.DataFrame(rows); good=null.T_joint.dropna().to_numpy(float)
    pone=(1+np.sum(good>=tobs))/(1+len(good)); ptwo=(1+np.sum(np.abs(good)>=abs(tobs)))/(1+len(good))
    sm={'Universe':'Joint','T_FU_obs':tobs,'T_ETF_obs':obs[0],'T_Firm_obs':obs[1],
        'p_perm_one_sided':float(pone),'p_perm_two_sided':float(ptwo),'B_valid':len(good),'N_dates':len(dates),
        'NullMean':float(np.mean(good)),'NullQ025':float(np.quantile(good,.025)),'NullQ975':float(np.quantile(good,.975))}
    return sm,null


def v7_final_identification(v5wei,v5daily,state_df,factors,blocks_by_u,cfg,audit,out,Bperm=1000):
    """Final locked inference. No model/hyperparameter re-tuning occurs in V7."""
    tabs=out/'tables'; csvd=out/'csv'; figs=out/'figures'; tabs.mkdir(exist_ok=True);csvd.mkdir(exist_ok=True);figs.mkdir(exist_ok=True)
    states=_state_family(state_df,factors,cfg)
    # 1) Universe-specific CT randomization, now with 1000 shifts.
    rsum=[]; nulls=[]
    for u in ('ETF','Firm'):
        w=v5wei[(v5wei.Universe==u)&(v5wei.State=='CT')]
        sm,nu=_circular_shift_null(w,states['CT'],blocks_by_u[u],B=Bperm,min_shift=cfg.ci_block,seed=cfg.seed+(7401 if u=='ETF' else 7402))
        sm['Universe']=u; rsum.append(sm); nu['Universe']=u; nulls.append(nu)
    # 2) Same shifts jointly across universes.
    jsm,jnull=_joint_ct_shift_test(v5wei,states['CT'],blocks_by_u,cfg,B=Bperm); rsum.append(jsm)
    rand=pd.DataFrame(rsum); nul=pd.concat(nulls,ignore_index=True); jnull.to_csv(csvd/'v7_joint_ct_rotation_null_distribution.csv',index=False)
    nul.to_csv(csvd/'v7_ct_rotation_null_distribution.csv',index=False); rand.to_csv(tabs/'tab_v7_state_rotation_randomization.csv',index=False)
    # 3) Direct paired CT-vs-Market/Energy allocation contrasts.
    cr=[]
    for u in ('ETF','Firm'):
        for comp in ('Market','Energy'):
            rr=_paired_rotation_contrast(v5wei,u,comp,states['CT'],states[comp],blocks_by_u[u],cfg,B=max(2000,cfg.n_ci_boot))
            if rr is not None: cr.append(rr)
    contrasts=pd.DataFrame(cr); contrasts.to_csv(tabs/'tab_v7_state_rotation_contrasts.csv',index=False)
    # 4) Paired realized-risk contrasts on exactly common future observations.
    er=[]
    for u in ('ETF','Firm'):
        for comp in ('Market','Energy'):
            rr=_paired_es_contrast(v5daily,u,comp,cfg,B=max(2000,cfg.n_ci_boot))
            if rr is not None: er.append(rr)
    es=pd.DataFrame(er); es.to_csv(tabs/'tab_v7_paired_oos_es_contrasts.csv',index=False)
    # Compact manuscript-facing figure: ETF, Firm and joint nulls.
    fig,axs=plt.subplots(1,3,figsize=(13,4),constrained_layout=True)
    for ax,u in zip(axs[:2],('ETF','Firm')):
        z=nul[nul.Universe==u].T_FU.dropna().to_numpy()*100; obs=float(rand.loc[rand.Universe==u,'T_FU_obs'].iloc[0])*100
        ax.hist(z,bins=30,alpha=.75); ax.axvline(obs,linestyle='--',linewidth=2); p=float(rand.loc[rand.Universe==u,'p_perm_one_sided'].iloc[0])
        ax.set_title(f'{u}: p={p:.3f}'); ax.set_xlabel(r'$T_{FU}$ (pp)'); ax.set_ylabel('Randomization frequency')
    z=jnull.T_joint.dropna().to_numpy()*100; obs=float(jsm['T_FU_obs'])*100
    axs[2].hist(z,bins=30,alpha=.75); axs[2].axvline(obs,linestyle='--',linewidth=2); axs[2].set_title(f"Joint: p={jsm['p_perm_one_sided']:.3f}"); axs[2].set_xlabel(r'$T_{joint}$ (pp)')
    fig.savefig(figs/'fig_v7_ct_rotation_falsification.pdf',bbox_inches='tight'); plt.close(fig)
    audit.check('V7 randomization inference',len(rand)==3 and (rand.B_valid>0).all(),f'rows={len(rand)}, B_valid={rand.B_valid.tolist()}',fatal=True)
    audit.check('V7 paired state contrasts',len(contrasts)==4,f'rows={len(contrasts)}',fatal=True)
    audit.check('V7 paired ES contrasts',len(es)==4,f'rows={len(es)}',fatal=True)
    return rand,contrasts,es,nul,jnull


def load_saved_v5_outputs(resume_root: Path, audit: Audit):
    csvd=resume_root/'csv'; tabs=resume_root/'tables'
    paths={'dec':csvd/'v5_state_placebo_decisions.csv','wei':csvd/'v5_state_placebo_weights.csv','daily':csvd/'v5_state_placebo_daily_returns.csv',
           'alloc':tabs/'tab_v5_state_specific_allocation.csv','risk':tabs/'tab_v5_state_specific_risk_turnover.csv'}
    missing=[str(p) for p in paths.values() if not p.exists()]
    audit.check('V7 reusable V5 outputs available',not missing,f'missing={missing}',fatal=True)
    d=pd.read_csv(paths['dec']); w=pd.read_csv(paths['wei']); r=pd.read_csv(paths['daily']); a=pd.read_csv(paths['alloc']); k=pd.read_csv(paths['risk'])
    for z in (d,w): z['Date']=pd.to_datetime(z['Date'])
    r['Date']=pd.to_datetime(r['Date']); r['FormationDate']=pd.to_datetime(r['FormationDate'])
    audit.check('V7 V5 resume schema',len(d)>0 and len(w)>0 and len(r)>0,f'dec={len(d)}, weights={len(w)}, daily={len(r)}',fatal=True)
    return d,w,r,a,k

# -----------------------------------------------------------------------------
# Resume support
# -----------------------------------------------------------------------------
def load_saved_core_outputs(resume_root: Path, audit: Audit):
    """Load previously completed recursive outputs and validate their schema."""
    csvd = resume_root / "csv"
    paths = {
        "dec": csvd / "all_decision_diagnostics.csv",
        "wei": csvd / "all_portfolio_weights.csv",
        "daily": csvd / "all_daily_oos_returns.csv",
    }
    missing = [str(p) for p in paths.values() if not p.exists()]
    audit.check(
        "Resume core outputs available",
        not missing,
        "all three core CSV files found" if not missing else f"missing={missing}",
        fatal=True,
    )
    dec = pd.read_csv(paths["dec"])
    wei = pd.read_csv(paths["wei"])
    daily = pd.read_csv(paths["daily"])
    for df, cols, name in [
        (dec, ["Universe", "Date", "Method", "S"], "decision diagnostics"),
        (wei, ["Universe", "Date", "Method", "Asset", "Weight"], "portfolio weights"),
        (daily, ["Universe", "Date", "FormationDate", "Method", "Return"], "daily OOS returns"),
    ]:
        miss = [c for c in cols if c not in df.columns]
        audit.check(f"Resume schema: {name}", not miss, f"missing={miss}", fatal=True)
    dec["Date"] = pd.to_datetime(dec["Date"])
    wei["Date"] = pd.to_datetime(wei["Date"])
    daily["Date"] = pd.to_datetime(daily["Date"])
    daily["FormationDate"] = pd.to_datetime(daily["FormationDate"])
    audit.check("Resume decision outputs nonempty", len(dec) > 0, f"rows={len(dec)}", fatal=True)
    audit.check(
        "Resume no look-ahead decision ordering",
        (daily["FormationDate"] < daily["Date"]).all(),
        "FormationDate < realized return date",
        fatal=True,
    )
    return dec, wei, daily

# -----------------------------------------------------------------------------
# Driver
# -----------------------------------------------------------------------------
def load_validated_state_or_cpu(state_root:Path, factors:pd.DataFrame, W:int, audit:Audit):
    """Prefer the already validated Section-5 W120 state. Fallback: augment factors with a dated CPU column from state_dir and reconstruct.
    Never substitutes a competing factor for CPU_narrow.
    """
    candidates=(f"transition_state_v4_W{W}.csv", f"transition_state_v4_W{W}", "transition_state_v4_W120.csv")
    ps=resolve_file(state_root,candidates)
    if ps is not None:
        d=read_date_csv(ps)
        scol=pick_col(d,["S_CT","S_t","S","state","transition_state","S_EW","S_EW_W120"])
        if scol is not None:
            out=pd.DataFrame(index=d.index)
            out["S_CT"]=pd.to_numeric(d[scol],errors="coerce")
            for target,aliases in {"I_CT":["I_CT","I_t_CT","transition_index"],"EUA":["ZCT_EUA_logret","EUA_logret","Delta_log_EUA"],"CPU":["ZCT_CPU_narrow","CPU_narrow","CPU_narrow_lagged"]}.items():
                c=pick_col(d,aliases)
                if c is not None: out[target]=pd.to_numeric(d[c],errors="coerce")
            audit.check("Validated Section-5 state loaded",True,f"{ps} | state column={scol}")
            audit.check("Validated state in [0,1]",out.S_CT.dropna().between(0,1).all(),f"N={out.S_CT.notna().sum()}",fatal=True)
            return out
        audit.check("Validated state file has recognizable state column",False,f"{ps}; columns={list(d.columns)}")
    # Search prepared factor files in state_dir, not raw monthly CPU, so availability dating remains upstream.
    factor_candidates=("Z_factors_v3_market_calendar.csv","Z_factors_v3_complete_case.csv","Z_factors_v3_union.csv","Z_factors_daily_common.csv")
    pz=resolve_file(state_root,factor_candidates)
    if pz is not None:
        z=read_date_csv(pz)
        cpu=pick_col(z,["ZCT_CPU_narrow","CPU_narrow","CPU_narrow_lagged","Climate_Policy_Uncertainty_narrow"])
        eua=pick_col(z,["ZCT_EUA_logret","EUA_logret","Delta_log_EUA"])
        audit.check("Prepared CPU factor found in state directory",cpu is not None,f"{pz} | {cpu}")
        if cpu is not None:
            f=factors.copy()
            f["ZCT_CPU_narrow"]=pd.to_numeric(z[cpu],errors="coerce").reindex(f.index)
            if pick_col(f,["ZCT_EUA_logret","EUA_logret","Delta_log_EUA"]) is None and eua is not None:
                f["ZCT_EUA_logret"]=pd.to_numeric(z[eua],errors="coerce").reindex(f.index)
            audit.check("State reconstructed from prepared dated CPU",True,f"source={pz}")
            return construct_state(f,W,audit)
    audit.check("Section-5 CPU/state artifacts available",False,"No validated W-state or prepared dated CPU factor found in --state-dir. Raw monthly CPU is intentionally not auto-forward-filled.",fatal=True)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--data-dir",default=Config.data_dir);ap.add_argument("--state-dir",default=None,help="Directory containing validated Section-5 state/CPU artifacts (e.g. Z_t). If omitted, data-dir is used.");ap.add_argument("--output-dir",default=Config.output_dir)
    ap.add_argument("--use-cuda",action="store_true");ap.add_argument("--cpu",action="store_true");ap.add_argument("--fast",action="store_true")
    ap.add_argument("--resume", action="store_true", help="Reuse previously saved recursive CSV outputs and continue with figures/tables/placebos/robustness.")
    ap.add_argument("--resume-dir", default=None, help="Directory containing the previous Section6_Q1PP_Results tree. Defaults to --output-dir.")
    ap.add_argument("--max-weight",type=float,default=None,help="Optional empirical cap; omitted by default because locked Section 5 does not specify one.")
    ap.add_argument("--v6-permutations",type=int,default=500,help="Legacy V6 randomizations.")
    ap.add_argument("--v7-permutations",type=int,default=1000,help="Pre-specified circular-shift randomizations for V7.")
    ap.add_argument("--resume-v5",action="store_true",help="Reuse completed V5 state-placebo outputs instead of rerunning A3.")
    ap.add_argument("--resume-v5-dir",default=None,help="Directory containing completed V5 outputs; defaults to --output-dir.")
    args=ap.parse_args(); cfg=Config(data_dir=args.data_dir,output_dir=args.output_dir,use_cuda=(args.use_cuda and not args.cpu),fast=args.fast,max_weight=args.max_weight)
    if cfg.fast: cfg.bootstrap_B=60;cfg.latent_mc=512;cfg.outer_steps=80;cfg.n_ci_boot=200
    out=Path(cfg.output_dir);out.mkdir(parents=True,exist_ok=True);logger=setup_logger(out);audit=Audit(logger);seed_all(cfg.seed);t0=time.time()
    device=torch.device("cuda" if cfg.use_cuda and torch.cuda.is_available() else "cpu")
    logger.info("Section 6 Q1++ pipeline started | device=%s",device)
    audit.check("CUDA requested and available", (not cfg.use_cuda) or torch.cuda.is_available(), f"torch={torch.__version__}, device={device}")
    root=Path(cfg.data_dir); state_root=Path(args.state_dir) if args.state_dir else root
    audit.check("Data directory",root.exists(),str(root),fatal=True)
    audit.check("State directory",state_root.exists(),str(state_root),fatal=True)
    pa=resolve_file(root,cfg.universe_a_candidates);pb=resolve_file(root,cfg.universe_b_candidates);pf=resolve_file(root,cfg.factors_candidates)
    audit.check("Universe A file",pa is not None,str(pa),fatal=True);audit.check("Universe B file",pb is not None,str(pb),fatal=True);audit.check("Factors file",pf is not None,str(pf),fatal=True)
    A=read_date_csv(pa);B=read_date_csv(pb);F=read_date_csv(pf)
    common=A.index.intersection(B.index).intersection(F.index);A=A.loc[common];B=B.loc[common];F=F.loc[common]
    audit.check("20 assets ETF",A.shape[1]==20,f"d={A.shape[1]}",fatal=True);audit.check("20 assets Firm",B.shape[1]==20,f"d={B.shape[1]}",fatal=True)
    audit.check("No future-dated data",common.max()<=pd.Timestamp(cfg.test_end),f"max={common.max().date()}")
    state=load_validated_state_or_cpu(state_root,F,cfg.state_window,audit).reindex(common)
    audit.check("State in [0,1]",state.S_CT.dropna().between(0,1).all(),f"N={state.S_CT.notna().sum()}",fatal=True)
    expA=load_exposure_scores(root,list(A.columns),common,cfg.exposure_candidates,audit);expB=load_exposure_scores(root,list(B.columns),common,cfg.exposure_candidates,audit)
    # provenance JSON before compute
    provenance={"created":datetime.now().isoformat(),"config":asdict(cfg),"python":sys.version,"platform":platform.platform(),"torch":torch.__version__,"cuda_available":torch.cuda.is_available(),"device":str(device),
                "files":{"UniverseA":str(pa),"UniverseB":str(pb),"Factors":str(pf),"StateDirectory":str(state_root)},"scientific_guards":["no future information","latent KL radius fixed before portfolio optimization","economic blocks never substituted for c_t","no imposed monotonicity between S and Gamma"]}
    with open(out/"section6_parameters.json","w",encoding="utf-8") as f:json.dump(provenance,f,indent=2,default=str)
    if args.resume:
        resume_root = Path(args.resume_dir) if args.resume_dir else out
        logger.info("RESUME mode | loading completed recursive outputs from %s", resume_root.resolve())
        dec, wei, daily = load_saved_core_outputs(resume_root, audit)
        # If resuming from another directory, preserve an exact local copy of the core outputs.
        (out / "csv").mkdir(exist_ok=True)
        if resume_root.resolve() != out.resolve():
            dec.to_csv(out / "csv" / "all_decision_diagnostics.csv", index=False)
            wei.to_csv(out / "csv" / "all_portfolio_weights.csv", index=False)
            daily.to_csv(out / "csv" / "all_daily_oos_returns.csv", index=False)
    else:
        decs=[];weis=[];dailys=[];xis={}
        for u,R,ex in [("ETF",A,expA),("Firm",B,expB)]:
            logger.info("[%s] recursive estimation + optimization",u)
            d,w,r,xi=run_universe(u,R,state,F,cfg,device,audit,out,exposure=ex);decs.append(d);weis.append(w);dailys.append(r);xis[u]=xi
        dec=pd.concat(decs,ignore_index=True);wei=pd.concat(weis,ignore_index=True);daily=pd.concat(dailys,ignore_index=True)
        (out/"csv").mkdir(exist_ok=True);dec.to_csv(out/"csv"/"all_decision_diagnostics.csv",index=False);wei.to_csv(out/"csv"/"all_portfolio_weights.csv",index=False);daily.to_csv(out/"csv"/"all_daily_oos_returns.csv",index=False)
        audit.check("Decision outputs nonempty",len(dec)>0,f"rows={len(dec)}",fatal=True)
        audit.check("No look-ahead decision ordering",(pd.to_datetime(daily.FormationDate)<pd.to_datetime(daily.Date)).all(),"FormationDate < realized return date",fatal=True)
    t1,t2,t3=build_outputs(dec,wei,daily,{"ETF":ETF_BLOCKS,"Firm":FIRM_BLOCKS},cfg,out,audit)
    f5=placebo_summary({"ETF":A,"Firm":B},state,F,cfg,device,audit,out)
    v4d,v4t=v4_distributional_diagnostics({"ETF":A,"Firm":B},state,F,cfg,out)
    audit.check("V4 distributional diagnostics",len(v4d)>0,f"rows={len(v4d)}",fatal=True)
    audit.check("V4 tail-event diagnostics",len(v4t)>0,f"rows={len(v4t)}",fatal=True)
    if args.resume_v5:
        v5root=Path(args.resume_v5_dir) if args.resume_v5_dir else out
        logger.info("V7 RESUME-V5 | loading completed V5 outputs from %s",v5root.resolve())
        v5dec,v5wei,v5daily,v5alloc,v5risk=load_saved_v5_outputs(v5root,audit)
    else:
        v5dec,v5wei,v5daily,v5alloc,v5risk = v5_portfolio_falsification(
            {"ETF":A,"Firm":B}, state, F, {"ETF":ETF_BLOCKS,"Firm":FIRM_BLOCKS}, cfg, device, audit, out
        )
    v7rand,v7contrast,v7es,v7null,v7joint = v7_final_identification(
        v5wei,v5daily,state,F,{"ETF":ETF_BLOCKS,"Firm":FIRM_BLOCKS},cfg,audit,out,Bperm=args.v7_permutations
    )
    t4=robustness_from_daily(dec,daily,cfg,out)
    # exposure-dependent deliverables are allowed to exist with NaNs, but are explicitly audited.
    audit.check("Exposure results identified",dec.Exposure.notna().any(),"Requires dated asset-level c_{i,t}; block labels are descriptive only.")
    # manuscript-facing inventory
    inventory=pd.DataFrame([
      ["Figure 1","fig:empirical_state_latent_mapping","fig1_empirical_state_latent_mapping.csv"],
      ["Figure 2","fig:empirical_mechanism_ablation","fig2_empirical_mechanism_ablation.csv"],
      ["Figure 3","fig:state_dependent_reallocation","fig3_state_dependent_reallocation.csv"],
      ["Figure 4","fig:tail_risk_performance_frontier","fig4_tail_risk_performance_frontier.csv"],
      ["Figure 5","fig:competing_state_placebos","fig5_competing_state_placebos.csv"],
      ["Figure 6","fig:economic_robustness","fig6_economic_robustness.csv"],
      ["V5 Figure","fig:state_specific_portfolio_falsification","fig_v5_state_specific_allocation.csv"],
      ["V5 Allocation Table","tab:state_specific_allocation_falsification","tab_v5_state_specific_allocation.csv"],
      ["V5 Risk Table","tab:state_specific_risk_turnover","tab_v5_state_specific_risk_turnover.csv"],
      ["V7 Figure","fig:ct_rotation_falsification","v7_ct_rotation_null_distribution.csv"],
      ["V7 Randomization Table","tab:state_rotation_randomization","tab_v7_state_rotation_randomization.csv"],
      ["V7 Rotation Contrasts","tab:state_rotation_contrasts","tab_v7_state_rotation_contrasts.csv"],
      ["V7 Paired ES","tab:paired_oos_es_contrasts","tab_v7_paired_oos_es_contrasts.csv"],
      ["Table 1","tab:empirical_mechanism_ablation","tab_empirical_mechanism_ablation.csv"],
      ["Table 2","tab:state_dependent_allocation_heterogeneity","tab_state_dependent_allocation_heterogeneity.csv"],
      ["Table 3","tab:out_of_sample_performance","tab_out_of_sample_performance.csv"],
      ["Table 4","tab:empirical_robustness_inference","tab_empirical_robustness_inference.csv"],
    ],columns=["Object","LaTeX_label","CSV"]);inventory.to_csv(out/"output_inventory.csv",index=False)
    audit.save(out/"checks.csv"); logger.info("DONE in %.2f min | outputs=%s",(time.time()-t0)/60,out.resolve())

if __name__=="__main__": main()
