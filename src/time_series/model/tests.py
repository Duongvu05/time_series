import numpy as np
import pandas as pd
import statsmodels.api as sm
from loguru import logger
from scipy.stats import jarque_bera
from statsmodels.stats.diagnostic import het_arch
from statsmodels.tsa.stattools import adfuller


def run_jarque_bera(series: pd.Series, alpha: float = 0.05) -> dict:
    clean = series.dropna()
    stat, pval = jarque_bera(clean)
    reject = bool(pval < alpha)
    logger.info("Jarque-Bera [{}]: stat={:.4f}, p={:.6g}, reject_normality={}", series.name, stat, pval, reject)
    return {"jb_stat": float(stat), "p_value": float(pval), "reject_normality": reject}


def run_adf(series: pd.Series, alpha: float = 0.05) -> dict:
    clean = series.dropna()
    stat, pval, used_lag, nobs, _, _ = adfuller(clean, autolag="AIC")
    is_stationary = bool(pval < alpha)
    logger.info("ADF [{}]: stat={:.4f}, p={:.6g}, lag={}, stationary={}", series.name, stat, pval, used_lag, is_stationary)
    return {
        "variable": series.name,
        "adf_stat": float(stat),
        "p_value": float(pval),
        "used_lag": int(used_lag),
        "nobs": int(nobs),
        "is_stationary": is_stationary,
    }


def run_arch_lm(series: pd.Series, nlags: int = 10, alpha: float = 0.05) -> dict:
    clean = series.dropna()
    ols_X = np.ones(len(clean))
    ols_res = sm.OLS(clean, ols_X).fit()
    stat, pval, _, _ = het_arch(ols_res.resid, nlags=nlags)
    has_arch = bool(pval < alpha)
    logger.info("ARCH-LM [{}] nlags={}: stat={:.4f}, p={:.6g}, arch_effects={}", series.name, nlags, stat, pval, has_arch)
    return {"lm_stat": float(stat), "p_value": float(pval), "has_arch_effects": has_arch}


def build_model_features(
    full_df: pd.DataFrame,
    train_index: pd.Index,
    test_index: pd.Index,
    exog_vars: list[str] = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if exog_vars is None:
        exog_vars = ["SVI", "SVI_pos", "SVI_neg"]

    adf_results = {var: run_adf(full_df[var].loc[train_index]) for var in exog_vars}

    model_df = full_df[["return", "target"]].copy()
    for var in exog_vars:
        is_stationary = adf_results[var]["is_stationary"]
        full_series = full_df[var].copy()
        transformed = full_series if is_stationary else full_series.diff()
        transform_label = "LEVEL" if is_stationary else "DIFF"
        logger.info("Feature [{}]: using {} (stationary={})", var, transform_label, is_stationary)
        model_df[f"lag_{var}"] = transformed.shift(1)

    train = model_df.loc[train_index].copy().dropna()
    test = model_df.loc[test_index].copy().dropna()

    logger.info("Model train: {} rows [{} -> {}]", len(train), train.index.min().date(), train.index.max().date())
    logger.info("Model test:  {} rows [{} -> {}]", len(test), test.index.min().date(), test.index.max().date())
    return model_df, train, test
