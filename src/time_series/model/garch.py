import numpy as np
import pandas as pd
from arch import arch_model
from loguru import logger

EPS = 1e-10


def fit_arch_model(
    name: str,
    y: pd.Series,
    x: pd.DataFrame | None,
    mean: str,
    vol: str,
    p: int = 1,
    o: int = 0,
    q: int = 1,
):
    am = arch_model(
        y=y,
        x=x,
        mean=mean,
        lags=0,
        vol=vol,
        p=p,
        o=o,
        q=q,
        dist="studentst",
        rescale=False,
    )
    res = am.fit(disp="off")
    logger.info("Fitted [{}]: AIC={:.2f}, BIC={:.2f}, LogLik={:.2f}", name, res.aic, res.bic, res.loglikelihood)
    return am, res


def _intercept_name(params_index: pd.Index) -> str:
    for candidate in ("mu", "Const"):
        if candidate in params_index:
            return candidate
    return params_index[0]


def _mean_path_from_result(res, data_frame: pd.DataFrame) -> pd.Series:
    params = res.params
    intercept = float(params[_intercept_name(params.index)])
    mean_values = np.full(len(data_frame), intercept, dtype=float)

    for column in data_frame.columns:
        if column in params.index:
            mean_values += float(params[column]) * data_frame[column].to_numpy(dtype=float)

    return pd.Series(mean_values, index=data_frame.index, name="mean_forecast")


def recursive_garch_forecast(res, test_returns: pd.Series, mean_series: pd.Series | None = None) -> pd.Series:
    params = res.params
    omega = float(params["omega"])
    alpha = float(params["alpha[1]"])
    beta = float(params["beta[1]"])

    sigma2_prev = float(res.conditional_volatility.iloc[-1] ** 2)
    eps_prev = float(res.resid.iloc[-1])

    if mean_series is None:
        mean_series = pd.Series(float(params[_intercept_name(params.index)]), index=test_returns.index)

    forecasts = []
    for date, ret_t in test_returns.items():
        sigma2_t = omega + alpha * (eps_prev**2) + beta * sigma2_prev
        forecasts.append(sigma2_t)
        eps_prev = float(ret_t - mean_series.loc[date])
        sigma2_prev = sigma2_t

    return pd.Series(forecasts, index=test_returns.index, name="forecast_vol")


def recursive_egarch_forecast(res, test_returns: pd.Series, mean_series: pd.Series | None = None) -> pd.Series:
    params = res.params
    omega = float(params["omega"])
    alpha = float(params["alpha[1]"])
    gamma = float(params["gamma[1]"])
    beta = float(params["beta[1]"])

    sigma2_prev = float(res.conditional_volatility.iloc[-1] ** 2)
    eps_prev = float(res.resid.iloc[-1])
    z_prev = eps_prev / np.sqrt(max(sigma2_prev, EPS))
    log_sigma2_prev = np.log(max(sigma2_prev, EPS))
    c = np.sqrt(2.0 / np.pi)

    if mean_series is None:
        mean_series = pd.Series(float(params[_intercept_name(params.index)]), index=test_returns.index)

    forecasts = []
    for date, ret_t in test_returns.items():
        log_sigma2_t = omega + alpha * (abs(z_prev) - c) + gamma * z_prev + beta * log_sigma2_prev
        sigma2_t = float(np.exp(log_sigma2_t))
        forecasts.append(sigma2_t)

        eps_t = float(ret_t - mean_series.loc[date])
        sigma2_prev = sigma2_t
        log_sigma2_prev = np.log(max(sigma2_t, EPS))
        z_prev = eps_t / np.sqrt(max(sigma2_t, EPS))

    return pd.Series(forecasts, index=test_returns.index, name="forecast_vol")


def rolling_window_forecast(
    data: pd.DataFrame,
    eval_index: pd.Index,
    window_size: int,
    spec: dict,
    verbose_every: int = 50,
) -> pd.Series:
    preds: list[float] = []
    idxs: list = []

    for i, date in enumerate(eval_index, start=1):
        end_loc = data.index.get_loc(date)
        start_loc = max(0, end_loc - window_size)
        train_win = data.iloc[start_loc:end_loc]

        y_win = train_win["return"]
        x_cols = spec.get("x_cols")
        x_win = train_win[x_cols] if x_cols else None

        am = arch_model(
            y=y_win,
            x=x_win,
            mean=spec["mean"],
            lags=0,
            vol=spec["vol"],
            p=spec["p"],
            o=spec.get("o", 0),
            q=spec["q"],
            dist="studentst",
            rescale=False,
        )
        res = am.fit(disp="off")

        if x_cols:
            x_next = {col: np.array([float(data.loc[date, col])]) for col in x_cols}
            fcast = res.forecast(horizon=1, reindex=False, x=x_next)
        else:
            fcast = res.forecast(horizon=1, reindex=False)

        preds.append(float(fcast.variance.iloc[-1, 0]))
        idxs.append(date)

        if verbose_every and i % verbose_every == 0:
            logger.info("Rolling forecast: {}/{} steps", i, len(eval_index))

    logger.info("Rolling forecast completed: {} predictions", len(preds))
    return pd.Series(preds, index=idxs, name="rolling_forecast_vol")
