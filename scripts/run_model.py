"""Model pipeline: GARCH-family fitting, out-of-sample forecasting, evaluation."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd
from loguru import logger

from time_series.model import (
    build_model_features,
    fit_arch_model,
    recursive_egarch_forecast,
    recursive_garch_forecast,
    rolling_window_forecast,
    run_arch_lm,
    run_adf,
    run_jarque_bera,
)
from time_series.model.evaluation import build_comparison_table, build_dm_table

TRAIN_DATA_PATH = Path("data/insample_data.csv")
TEST_DATA_PATH = Path("data/outofsample_data.csv")
REQUIRED_COLS = ["return", "target", "SVI", "SVI_pos", "SVI_neg"]

ROLLING_WINDOW_SIZE = None  # None = use full train set size
ROLLING_MAX_STEPS = None    # None = evaluate entire test set
ROLLING_VERBOSE_EVERY = 30

EPS = 1e-10


def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_raw = pd.read_csv(TRAIN_DATA_PATH, parse_dates=["date"]).set_index("date").sort_index()
    test_raw = pd.read_csv(TEST_DATA_PATH, parse_dates=["date"]).set_index("date").sort_index()

    for name, df in [("insample", train_raw), ("outsample", test_raw)]:
        missing = [c for c in REQUIRED_COLS if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns in {name} file: {missing}")
        if df.index.duplicated().any():
            raise ValueError(f"Duplicated index values found in {name} file")

    full_df = pd.concat([train_raw, test_raw]).sort_index()
    if full_df.index.duplicated().any():
        raise ValueError("Duplicated dates found after concatenating in-sample and out-of-sample data")

    logger.info("In-sample:     {:,} rows [{} -> {}]", len(train_raw), train_raw.index.min().date(), train_raw.index.max().date())
    logger.info("Out-of-sample: {:,} rows [{} -> {}]", len(test_raw), test_raw.index.min().date(), test_raw.index.max().date())
    return train_raw, test_raw, full_df


def run_preliminary_tests(train_raw: pd.DataFrame) -> None:
    logger.info("=== Phase 1: Preliminary tests (train only) ===")
    run_jarque_bera(train_raw["return"])
    for var in ["return", "SVI", "SVI_pos", "SVI_neg"]:
        run_adf(train_raw[var].rename(var))
    run_arch_lm(train_raw["return"].rename("return"))


def fit_models(train: pd.DataFrame) -> dict:
    logger.info("=== Phase 2: In-sample estimation ===")
    results = {}

    _, res_garch = fit_arch_model("Baseline GARCH(1,1)", train["return"], None, "Constant", "GARCH", p=1, q=1)
    results["Baseline GARCH(1,1)"] = res_garch

    _, res_egarch = fit_arch_model("Baseline EGARCH(1,1)", train["return"], None, "Constant", "EGARCH", p=1, o=1, q=1)
    results["Baseline EGARCH(1,1)"] = res_egarch

    _, res_egarch_svi = fit_arch_model(
        "Extended EGARCH-SVI", train["return"], train[["lag_SVI"]], "ARX", "EGARCH", p=1, o=1, q=1
    )
    results["Extended EGARCH-SVI"] = res_egarch_svi

    _, res_egarch_asym = fit_arch_model(
        "Extended EGARCH-Asym", train["return"], train[["lag_SVI_pos", "lag_SVI_neg"]], "ARX", "EGARCH", p=1, o=1, q=1
    )
    results["Extended EGARCH-Asym"] = res_egarch_asym

    cmp = build_comparison_table(
        train["target"],
        {name: res.conditional_volatility**2 for name, res in results.items()},
    )
    logger.info("In-sample comparison:\n{}", cmp.to_string(index=False))
    return results


def run_recursive_forecasts(results: dict, test: pd.DataFrame) -> dict:
    logger.info("=== Phase 3: Recursive one-step-ahead forecasts ===")
    actual_ret = test["return"]

    from time_series.model.garch import _mean_path_from_result

    forecast_dict = {}

    forecast_dict["Baseline GARCH(1,1)"] = recursive_garch_forecast(results["Baseline GARCH(1,1)"], actual_ret)

    forecast_dict["Baseline EGARCH(1,1)"] = recursive_egarch_forecast(results["Baseline EGARCH(1,1)"], actual_ret)

    mean_test_svi = _mean_path_from_result(results["Extended EGARCH-SVI"], test[["lag_SVI"]])
    forecast_dict["Extended EGARCH-SVI"] = recursive_egarch_forecast(
        results["Extended EGARCH-SVI"], actual_ret, mean_series=mean_test_svi
    )

    mean_test_asym = _mean_path_from_result(results["Extended EGARCH-Asym"], test[["lag_SVI_pos", "lag_SVI_neg"]])
    forecast_dict["Extended EGARCH-Asym"] = recursive_egarch_forecast(
        results["Extended EGARCH-Asym"], actual_ret, mean_series=mean_test_asym
    )

    actual_vol = test["target"]
    oos_cmp = build_comparison_table(actual_vol, forecast_dict)
    logger.info("Out-of-sample comparison (recursive):\n{}", oos_cmp.to_string(index=False))

    dm_table = build_dm_table(
        actual=actual_vol,
        extended_forecasts={
            "Extended EGARCH-SVI": forecast_dict["Extended EGARCH-SVI"],
            "Extended EGARCH-Asym": forecast_dict["Extended EGARCH-Asym"],
        },
        baseline_name="Baseline EGARCH(1,1)",
        baseline_forecast=forecast_dict["Baseline EGARCH(1,1)"],
        h=1,
        hac_lags=1,
    )
    logger.info("Diebold-Mariano results:\n{}", dm_table.to_string(index=False))
    return forecast_dict


def run_rolling_forecasts(model_df: pd.DataFrame, test_index: pd.Index, train_size: int) -> dict:
    logger.info("=== Phase 4: Rolling-window forecasts ===")
    rolling_data = model_df[["return", "target", "lag_SVI", "lag_SVI_pos", "lag_SVI_neg"]].dropna()
    rolling_eval_index = rolling_data.index.intersection(test_index)

    if ROLLING_MAX_STEPS is not None:
        rolling_eval_index = rolling_eval_index[:ROLLING_MAX_STEPS]

    window_size = train_size if ROLLING_WINDOW_SIZE is None else ROLLING_WINDOW_SIZE
    logger.info("Rolling window size: {}, forecast steps: {}", window_size, len(rolling_eval_index))

    rolling_specs = {
        "Baseline GARCH(1,1)": {"mean": "Constant", "vol": "GARCH", "p": 1, "q": 1},
        "Baseline EGARCH(1,1)": {"mean": "Constant", "vol": "EGARCH", "p": 1, "o": 1, "q": 1},
        "Extended EGARCH-SVI": {"mean": "ARX", "vol": "EGARCH", "p": 1, "o": 1, "q": 1, "x_cols": ["lag_SVI"]},
        "Extended EGARCH-Asym": {
            "mean": "ARX", "vol": "EGARCH", "p": 1, "o": 1, "q": 1,
            "x_cols": ["lag_SVI_pos", "lag_SVI_neg"],
        },
    }

    rolling_forecast_dict = {}
    for model_name, spec in rolling_specs.items():
        logger.info("Rolling forecast: {}", model_name)
        rolling_forecast_dict[model_name] = rolling_window_forecast(
            data=rolling_data,
            eval_index=rolling_eval_index,
            window_size=window_size,
            spec=spec,
            verbose_every=ROLLING_VERBOSE_EVERY,
        )

    rolling_actual = rolling_data.loc[rolling_eval_index, "target"].clip(lower=EPS)
    rolling_cmp = build_comparison_table(rolling_actual, rolling_forecast_dict)
    logger.info("Rolling-window comparison:\n{}", rolling_cmp.to_string(index=False))
    return rolling_forecast_dict


def main() -> None:
    logger.info("Starting model pipeline")
    train_raw, test_raw, full_df = load_data()

    run_preliminary_tests(train_raw)

    model_df, train, test = build_model_features(
        full_df=full_df,
        train_index=train_raw.index,
        test_index=test_raw.index,
    )

    results = fit_models(train)
    run_recursive_forecasts(results, test)
    run_rolling_forecasts(model_df, test_raw.index, train_size=len(train))

    logger.info("Model pipeline complete.")


if __name__ == "__main__":
    main()
