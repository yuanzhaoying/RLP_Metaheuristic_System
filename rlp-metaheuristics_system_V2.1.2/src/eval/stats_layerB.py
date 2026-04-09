# src/prlp/eval/stats_layerB.py
from __future__ import annotations
import os
import numpy as np
import pandas as pd
from scipy.stats import chi2, rankdata

import statsmodels.formula.api as smf
from statsmodels.stats.anova import anova_lm

# -----------------------------
# helpers: response construction
# -----------------------------
def build_layerB_df(perf: pd.DataFrame, response: str = "log1p_gap", eps: float = 1e-12) -> pd.DataFrame:
    """
    perf: results/summary/scenario_algo_perf.csv
    must include: scenario_id, instance_id, set, delta, encoding, operator, search_strategy,
                  median_obj, best_obj, RPD_median

    response:
      - log1p_gap: y = log(1 + (median-best)/(best+eps))
      - log1p_rpd: y = log(1 + RPD_median)
    """
    df = perf.copy()
    df["gap"] = (df["median_obj"] - df["best_obj"]) / (df["best_obj"] + eps)
    if response == "log1p_gap":
        df["y"] = np.log1p(df["gap"].clip(lower=0))
    elif response == "log1p_rpd":
        df["y"] = np.log1p(df["RPD_median"].clip(lower=0))
    else:
        raise ValueError(response)

    for c in ["encoding", "operator", "search_strategy", "set", "instance_id"]:
        if c in df.columns:
            df[c] = df[c].astype("category")
    return df

# -----------------------------
# MixedLM fitting
# -----------------------------
def _vc_formula(df: pd.DataFrame, use_set_vc: bool = True) -> dict | None:
    vc = {}
    if use_set_vc and "set" in df.columns:
        vc["set_vc"] = "0 + C(set)"
    return vc if len(vc) else None

def _fit_mixedlm(df: pd.DataFrame, formula: str, vc_formula: dict | None,
                 reml: bool = False, method: str = "lbfgs", maxiter: int = 200):
    model = smf.mixedlm(formula, df, groups=df["instance_id"], vc_formula=vc_formula)
    try:
        res = model.fit(reml=reml, method=method, maxiter=maxiter, disp=False)
    except Exception:
        # fallback
        res = model.fit(reml=reml, method="powell", maxiter=maxiter, disp=False)
    return res

def mixedlm_full(df: pd.DataFrame, include_delta: bool = True,
                 include_interaction: bool = True,
                 use_set_vc: bool = True,
                 reml: bool = False, method: str = "lbfgs", maxiter: int = 200):
    terms = ["C(encoding)", "C(operator)", "C(search_strategy)"]
    if include_delta:
        terms.append("C(delta)")
    if include_interaction:
        terms.append("C(encoding):C(operator)")
    formula = "y ~ " + " + ".join(terms)
    vc = _vc_formula(df, use_set_vc=use_set_vc)
    res = _fit_mixedlm(df, formula, vc, reml=reml, method=method, maxiter=maxiter)
    return res, formula, vc

# -----------------------------
# Factor-level LRT (full vs reduced)
# -----------------------------
def mixedlm_factor_lrt(df: pd.DataFrame,
                       include_delta: bool = True,
                       use_set_vc: bool = True,
                       reml: bool = False, method: str = "lbfgs", maxiter: int = 200):
    """
    Return a dataframe with factor-level p-values using likelihood ratio tests.
    """
    full_res, full_formula, vc = mixedlm_full(
        df,
        include_delta=include_delta,
        include_interaction=True,
        use_set_vc=use_set_vc,
        reml=reml, method=method, maxiter=maxiter
    )
    llf_full = float(full_res.llf)
    k_full = len(full_res.params)

    terms = ["C(encoding)", "C(operator)", "C(search_strategy)"]
    if include_delta:
        terms.append("C(delta)")
    terms.append("C(encoding):C(operator)")

    rows = []
    for drop_term in terms:
        # build reduced formula
        keep = [t for t in terms if t != drop_term]
        red_formula = "y ~ " + " + ".join(keep)
        red_res = _fit_mixedlm(df, red_formula, vc, reml=reml, method=method, maxiter=maxiter)
        llf_red = float(red_res.llf)
        k_red = len(red_res.params)
        lr = 2.0 * (llf_full - llf_red)
        df_diff = max(1, k_full - k_red)
        p = float(chi2.sf(lr, df_diff))
        rows.append({"term": drop_term, "lr_stat": lr, "df_diff": df_diff, "p_value": p})

    out = pd.DataFrame(rows).sort_values("p_value").reset_index(drop=True)
    return out, full_res, full_formula

# -----------------------------
# Cluster bootstrap CI (resample instances)
# -----------------------------
def bootstrap_ci_mixedlm(df: pd.DataFrame,
                         include_delta: bool = True,
                         use_set_vc: bool = True,
                         reml: bool = False, method: str = "lbfgs", maxiter: int = 200,
                         n_resamples: int = 200, seed: int = 0):
    rng = np.random.default_rng(seed)
    inst_ids = df["instance_id"].astype(str).unique().tolist()

    samples = []
    for b in range(n_resamples):
        pick = rng.choice(inst_ids, size=len(inst_ids), replace=True)
        db = pd.concat([df[df["instance_id"].astype(str) == sid] for sid in pick], ignore_index=True)
        try:
            res, _, _ = mixedlm_full(
                db,
                include_delta=include_delta,
                include_interaction=True,
                use_set_vc=use_set_vc,
                reml=reml, method=method, maxiter=maxiter
            )
            # collect params
            for term, val in res.params.items():
                samples.append({"boot_id": b, "term": term, "coef": float(val)})
        except Exception:
            # skip failed fits
            continue

    if not samples:
        return pd.DataFrame(columns=["term","boot_mean","boot_std","ci025","ci975","n"])

    s = pd.DataFrame(samples)
    ci = (s.groupby("term")["coef"]
            .agg(
                boot_mean="mean",
                boot_std="std",
                ci025=lambda x: np.quantile(x, 0.025),
                ci975=lambda x: np.quantile(x, 0.975),
                n="count"
            )
            .reset_index())
    return ci

# -----------------------------
# ART: aligned rank transform tests (factor-level) + partial eta^2
# -----------------------------
def art_anova(df: pd.DataFrame,
              include_delta: bool = True,
              include_interaction: bool = True,
              anova_type: int = 2):
    """
    ART for factorial terms.
    We treat instance_id as a blocking fixed effect: C(instance_id).
    """
    terms = ["C(encoding)", "C(operator)", "C(search_strategy)"]
    if include_delta:
        terms.append("C(delta)")
    if include_interaction:
        terms.append("C(encoding):C(operator)")

    rows = []
    for t in terms:
        # reduced model excludes term t
        red_terms = [x for x in terms if x != t]
        red_formula = "y ~ C(instance_id) + " + " + ".join(red_terms)
        red = smf.ols(red_formula, df).fit()

        aligned = df["y"].values - red.fittedvalues.values
        y_rank = rankdata(aligned)

        d2 = df.copy()
        d2["y_rank"] = y_rank

        full_formula = "y_rank ~ C(instance_id) + " + " + ".join(terms)
        full = smf.ols(full_formula, d2).fit()
        aov = anova_lm(full, typ=anova_type)

        # locate row for term (sometimes spacing differs)
        key = None
        if t in aov.index:
            key = t
        else:
            # try match ignoring spaces
            for ix in aov.index:
                if ix.replace(" ", "") == t.replace(" ", ""):
                    key = ix
                    break
        if key is None:
            continue

        ss_term = float(aov.loc[key, "sum_sq"])
        ss_res = float(aov.loc["Residual", "sum_sq"])
        eta2 = ss_term / (ss_term + ss_res + 1e-12)

        rows.append({
            "term": t,
            "F": float(aov.loc[key, "F"]),
            "p_value": float(aov.loc[key, "PR(>F)"]),
            "partial_eta2": float(eta2),
            "anova_type": int(anova_type),
        })

    return pd.DataFrame(rows).sort_values("p_value").reset_index(drop=True)