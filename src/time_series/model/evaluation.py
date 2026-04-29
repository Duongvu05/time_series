import numpy as np
import pandas as pd
from loguru import logger
from scipy import stats

EPS = 1e-10


def evaluate_volatility_forecasts(actual: pd.Series, forecast: pd.Series) -> dict:
    aligned_actual, aligned_forecast = actual.align(forecast, join="inner")
    aligned_actual = aligned_actual.clip(lower=EPS)
    aligned_forecast = aligned_forecast.clip(lower=EPS)

    rmse = float(np.sqrt(np.mean((aligned_actual - aligned_forecast) ** 2)))
    qlike = float(np.mean(aligned_actual / aligned_forecast - np.log(aligned_actual / aligned_forecast) - 1.0))
    return {"RMSE": rmse, "QLIKE": qlike}


def qlike_loss_series(actual: pd.Series, forecast: pd.Series, eps: float = EPS) -> pd.Series:
    aligned_actual, aligned_forecast = actual.align(forecast, join="inner")
    aligned_actual = aligned_actual.clip(lower=eps)
    aligned_forecast = aligned_forecast.clip(lower=eps)
    return aligned_actual / aligned_forecast - np.log(aligned_actual / aligned_forecast) - 1.0


def se_loss_series(actual: pd.Series, forecast: pd.Series) -> pd.Series:
    aligned_actual, aligned_forecast = actual.align(forecast, join="inner")
    return (aligned_actual - aligned_forecast) ** 2


def dm_test(
    loss_model_a: pd.Series,
    loss_model_b: pd.Series,
    h: int = 1,
    hac_lags: int | None = None,
) -> dict:
    """Diebold-Mariano test with Newey-West HAC variance + Harvey-Leybourne-Newbold correction.

    d_t = loss_benchmark - loss_model_a
    mean_loss_diff > 0 => model_a has lower loss than benchmark.
    """
    aligned_a, aligned_b = loss_model_a.align(loss_model_b, join="inner")
    d = (aligned_b - aligned_a).dropna()
    n = len(d)

    if n <= max(2, h):
        return {"n": n, "dm_stat": np.nan, "p_value": np.nan, "mean_loss_diff": np.nan, "hac_lags": np.nan, "h": h}

    d_values = d.to_numpy(dtype=float)
    d_mean = float(np.mean(d_values))
    d_centered = d_values - d_mean

    if hac_lags is None:
        hac_lags = int(np.floor(1.5 * n ** (1.0 / 3.0)))

    lag_max = max(h - 1, min(int(hac_lags), n - 1))

    gamma0 = float(np.mean(d_centered * d_centered))
    long_run_var = gamma0
    for lag in range(1, lag_max + 1):
        gamma_lag = float(np.mean(d_centered[lag:] * d_centered[:-lag]))
        weight = 1.0 - lag / (lag_max + 1.0)
        long_run_var += 2.0 * weight * gamma_lag

    long_run_var = max(long_run_var, np.finfo(float).eps)
    dm_raw = d_mean / np.sqrt(long_run_var / n)

    h = int(max(1, h))
    hln_factor_sq = (n + 1.0 - 2.0 * h + (h * (h - 1.0)) / n) / n
    hln_factor_sq = max(hln_factor_sq, np.finfo(float).eps)
    dm_hln = dm_raw * np.sqrt(hln_factor_sq)
    p_value = 2.0 * (1.0 - stats.norm.cdf(abs(dm_hln)))

    return {
        "n": n,
        "dm_stat": float(dm_hln),
        "p_value": float(p_value),
        "mean_loss_diff": float(d_mean),
        "hac_lags": int(lag_max),
        "h": h,
    }


def build_comparison_table(
    actual: pd.Series,
    forecast_dict: dict[str, pd.Series],
    sort_by: list[str] = None,
) -> pd.DataFrame:
    if sort_by is None:
        sort_by = ["RMSE", "QLIKE"]

    rows = []
    for name, pred in forecast_dict.items():
        metrics = evaluate_volatility_forecasts(actual, pred)
        rows.append({"Model": name, **metrics})

    cmp = pd.DataFrame(rows).sort_values(sort_by).reset_index(drop=True)
    cmp["Best_RMSE"] = cmp["RMSE"].eq(cmp["RMSE"].min())
    cmp["Best_QLIKE"] = cmp["QLIKE"].eq(cmp["QLIKE"].min())

    best_rmse = cmp.loc[cmp["RMSE"].idxmin(), "Model"]
    best_qlike = cmp.loc[cmp["QLIKE"].idxmin(), "Model"]
    logger.info("Best by RMSE:  {}", best_rmse)
    logger.info("Best by QLIKE: {}", best_qlike)
    return cmp


def build_dm_table(
    actual: pd.Series,
    extended_forecasts: dict[str, pd.Series],
    baseline_name: str,
    baseline_forecast: pd.Series,
    h: int = 1,
    hac_lags: int = 1,
) -> pd.DataFrame:
    rows = []
    for ext_name, ext_fcst in extended_forecasts.items():
        qlike_dm = dm_test(
            loss_model_a=qlike_loss_series(actual, ext_fcst),
            loss_model_b=qlike_loss_series(actual, baseline_forecast),
            h=h,
            hac_lags=hac_lags,
        )
        se_dm = dm_test(
            loss_model_a=se_loss_series(actual, ext_fcst),
            loss_model_b=se_loss_series(actual, baseline_forecast),
            h=h,
            hac_lags=hac_lags,
        )
        rows.append({
            "Pair": f"{ext_name} vs {baseline_name}",
            "MeanDiff_QLIKE(Base-Ext)": qlike_dm["mean_loss_diff"],
            "DM_QLIKE": qlike_dm["dm_stat"],
            "PValue_QLIKE": qlike_dm["p_value"],
            "Decision_QLIKE_5pct": "Reject H0" if qlike_dm["p_value"] < 0.05 else "Fail to reject H0",
            "MeanDiff_SE(Base-Ext)": se_dm["mean_loss_diff"],
            "DM_SE": se_dm["dm_stat"],
            "PValue_SE": se_dm["p_value"],
            "Decision_SE_5pct": "Reject H0" if se_dm["p_value"] < 0.05 else "Fail to reject H0",
            "N": qlike_dm["n"],
        })
    return pd.DataFrame(rows)
