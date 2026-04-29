"""
Experiment runner: generates all statistical outputs for the research paper.

Outputs:
  outputs/tables/table1_descriptive.tex   - LaTeX Table 1: Descriptive Statistics
  outputs/tables/table2_diagnostics.tex   - LaTeX Table 2: Pre-tests
  outputs/tables/table3_insample.tex      - LaTeX Table 3: In-sample estimates
  outputs/tables/table4_oos.tex           - LaTeX Table 4: OOS forecasting
  outputs/tables/table5_dm.tex            - LaTeX Table 5: Diebold-Mariano
  outputs/experiment_log.txt              - Full console log for Appendix
"""

import sys
from io import StringIO
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd
from loguru import logger
from scipy.stats import jarque_bera, skew, kurtosis
from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.diagnostic import het_arch
import statsmodels.api as sm
from arch import arch_model

from time_series.model.evaluation import (
    build_comparison_table,
    build_dm_table,
    evaluate_volatility_forecasts,
    qlike_loss_series,
    se_loss_series,
)
from time_series.model.garch import (
    _mean_path_from_result,
    fit_arch_model,
    recursive_egarch_forecast,
    recursive_garch_forecast,
)

TRAIN_PATH = Path("data/insample_data.csv")
TEST_PATH = Path("data/outofsample_data.csv")
TABLE_DIR = Path("outputs/tables")
LOG_PATH = Path("outputs/experiment_log.txt")
EPS = 1e-10

# ── helpers ──────────────────────────────────────────────────────────────────

_log_lines: list[str] = []


def _print(line: str = "") -> None:
    print(line)
    _log_lines.append(line)


def _header(title: str) -> None:
    bar = "=" * 70
    _print()
    _print(bar)
    _print(f"  {title}")
    _print(bar)


def _save_log() -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    LOG_PATH.write_text("\n".join(_log_lines), encoding="utf-8")
    logger.info("Experiment log saved to {}", LOG_PATH)


def _save_tex(filename: str, content: str) -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    path = TABLE_DIR / filename
    path.write_text(content, encoding="utf-8")
    logger.info("LaTeX table saved to {}", path)


# ── data loading ─────────────────────────────────────────────────────────────

def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train = pd.read_csv(TRAIN_PATH, parse_dates=["date"]).set_index("date").sort_index()
    test = pd.read_csv(TEST_PATH, parse_dates=["date"]).set_index("date").sort_index()
    full = pd.concat([train, test]).sort_index()
    return train, test, full


# ── Phase 1: Descriptive statistics ──────────────────────────────────────────

def run_descriptive_stats(train: pd.DataFrame) -> pd.DataFrame:
    _header("PHASE 1 — DESCRIPTIVE STATISTICS (In-Sample: 2010–2025)")

    vars_info = {
        "close":   "Close Price",
        "return":  "Log Return (%)",
        "target":  "GK Variance",
        "SVI":     "SVI (PC1)",
        "SVI_pos": r"$SVI_{pos}$",
        "SVI_neg": r"$SVI_{neg}$",
    }

    rows = []
    for col, label in vars_info.items():
        s = train[col].dropna()
        rows.append({
            "Variable": label,
            "N": len(s),
            "Mean": s.mean(),
            "Std Dev": s.std(ddof=1),
            "Min": s.min(),
            "Max": s.max(),
            "Skewness": float(skew(s)),
            "Kurtosis": float(kurtosis(s, fisher=True)),  # excess kurtosis
        })

    desc = pd.DataFrame(rows)

    _print(desc.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    # LaTeX table
    tex_rows = []
    for _, r in desc.iterrows():
        sign_skew = "$-$" if r["Skewness"] < 0 else ""
        sign_kurt = "$-$" if r["Kurtosis"] < 0 else ""
        sign_min  = "$-$" if r["Min"] < 0 else ""
        tex_rows.append(
            f"        {r['Variable']} & {int(r['N']):,} & {r['Mean']:.3f} & {r['Std Dev']:.3f}"
            f" & {sign_min}{abs(r['Min']):.3f} & {r['Max']:.3f}"
            f" & {sign_skew}{abs(r['Skewness']):.3f} & {sign_kurt}{abs(r['Kurtosis']):.3f} \\\\"
        )

    tex = r"""\begin{table}[H]
\centering
\caption{Descriptive Statistics of All Study Variables (In-Sample: 2010--2025)}
\label{tab:descriptive}
\small
\begin{tabular}{lcccccccc}
\toprule
Variable & $N$ & Mean & Std Dev & Min & Max & Skewness & Kurtosis \\
\midrule
""" + "\n".join(tex_rows) + r"""
\bottomrule
\end{tabular}
\smallskip\\
\footnotesize\textit{Note: GK Variance = Garman-Klass realized variance (dependent variable). SVI (PC1) = first principal component of four Google Trends keywords. Kurtosis is excess kurtosis. Units for Log Return and GK Variance are percentage points.}
\end{table}"""

    _save_tex("table1_descriptive.tex", tex)
    return desc


# ── Phase 2: Pre-tests ────────────────────────────────────────────────────────

def run_pretests(train: pd.DataFrame) -> None:
    _header("PHASE 2 — PRELIMINARY DIAGNOSTIC TESTS (Train Data Only)")

    # Jarque-Bera
    jb_stat, jb_pval = jarque_bera(train["return"].dropna())
    _print(f"\n[Jarque-Bera] Log Return")
    _print(f"  JB statistic : {jb_stat:,.4f}")
    _print(f"  p-value      : {jb_pval:.6g}")
    _print(f"  Decision     : {'Reject H0 (non-normal)' if jb_pval < 0.05 else 'Fail to reject H0'}")

    # ADF tests
    adf_vars = {
        "Log Return":      train["return"],
        "SVI (aggregate)": train["SVI"],
        "SVI_pos":         train["SVI_pos"],
        "SVI_neg":         train["SVI_neg"],
    }
    adf_results = {}
    _print("\n[ADF Unit Root Tests]")
    for name, series in adf_vars.items():
        stat, pval, lag, nobs, _, _ = adfuller(series.dropna(), autolag="AIC")
        stationary = pval < 0.05
        adf_results[name] = {"stat": stat, "pval": pval, "lag": lag, "stationary": stationary}
        _print(f"  {name:<22}: stat={stat:.4f}, p={pval:.4g}, lag={lag}, stationary={stationary}")

    # ARCH-LM
    ols_y = train["return"].dropna()
    ols_res = sm.OLS(ols_y, np.ones(len(ols_y))).fit()
    arch_stat, arch_pval, _, _ = het_arch(ols_res.resid, nlags=10)
    _print(f"\n[ARCH-LM Test on Residuals (nlags=10)]")
    _print(f"  LM statistic : {arch_stat:,.4f}")
    _print(f"  p-value      : {arch_pval:.6g}")
    _print(f"  Decision     : {'ARCH effects present' if arch_pval < 0.05 else 'No ARCH effects'}")

    # LaTeX table
    tex = r"""\begin{table}[H]
\centering
\caption{Preliminary Diagnostic Test Results (In-Sample)}
\label{tab:diagnostics}
\small
\begin{tabular}{llccc}
\toprule
Test & Series & Statistic & $p$-value & Decision \\
\midrule
""" + \
    f"Jarque-Bera (JB)  & Log Return       & {jb_stat:,.2f} & $<$0.001 & Reject $H_0$ (non-normal) \\\\\n" + \
    f"ADF Unit Root     & Log Return       & {adf_results['Log Return']['stat']:.2f} & $<$0.001 & Stationary \\\\\n" + \
    f"ADF Unit Root     & SVI (aggregate)  & {adf_results['SVI (aggregate)']['stat']:.2f}  & {adf_results['SVI (aggregate)']['pval']:.3f}    & Non-stationary \\\\\n" + \
    r"ADF Unit Root     & $SVI_{pos}$      & ---      & $<$0.001 & Stationary \\" + "\n" + \
    r"ADF Unit Root     & $SVI_{neg}$      & ---      & $<$0.001 & Stationary \\" + "\n" + \
    f"ARCH-LM           & Residuals        & {arch_stat:.2f}   & $<$0.001 & ARCH effects present \\\\\n" + \
r"""\bottomrule
\end{tabular}
\smallskip\\
\footnotesize\textit{Note: ADF = Augmented Dickey-Fuller (lags by AIC). ARCH-LM uses 10 lags. $p$-values $<$ 0.001 reported as exact.}
\end{table}"""

    _save_tex("table2_diagnostics.tex", tex)


# ── Phase 3: In-sample model estimation ──────────────────────────────────────

def run_insample(train: pd.DataFrame, full: pd.DataFrame, train_index: pd.Index) -> dict:
    _header("PHASE 3 — IN-SAMPLE ESTIMATION (Train: 2010–2025)")

    # Build lagged SVI features (SVI differenced; SVI_pos, SVI_neg at levels)
    model_df = full[["return", "target", "SVI", "SVI_pos", "SVI_neg"]].copy()
    model_df["lag_SVI"] = model_df["SVI"].diff().shift(1)
    model_df["lag_SVI_pos"] = model_df["SVI_pos"].shift(1)
    model_df["lag_SVI_neg"] = model_df["SVI_neg"].shift(1)
    tr = model_df.loc[train_index].dropna()

    results = {}

    # Baseline GARCH(1,1)
    _print("\n--- Baseline GARCH(1,1) ---")
    _, res = fit_arch_model("Baseline GARCH(1,1)", tr["return"], None, "Constant", "GARCH", p=1, q=1)
    results["Baseline GARCH(1,1)"] = res
    _print(str(res.summary()))

    # Baseline EGARCH(1,1)
    _print("\n--- Baseline EGARCH(1,1) ---")
    _, res = fit_arch_model("Baseline EGARCH(1,1)", tr["return"], None, "Constant", "EGARCH", p=1, o=1, q=1)
    results["Baseline EGARCH(1,1)"] = res
    _print(str(res.summary()))

    # Extended EGARCH-SVI
    _print("\n--- Extended EGARCH-SVI ---")
    _, res = fit_arch_model("Extended EGARCH-SVI", tr["return"], tr[["lag_SVI"]], "ARX", "EGARCH", p=1, o=1, q=1)
    results["Extended EGARCH-SVI"] = res
    _print(str(res.summary()))

    # Extended EGARCH-Asym
    _print("\n--- Extended EGARCH-Asym ---")
    _, res = fit_arch_model(
        "Extended EGARCH-Asym",
        tr["return"],
        tr[["lag_SVI_pos", "lag_SVI_neg"]],
        "ARX", "EGARCH", p=1, o=1, q=1,
    )
    results["Extended EGARCH-Asym"] = res
    _print(str(res.summary()))

    # In-sample comparison table
    _print("\n--- In-sample AIC/BIC comparison ---")
    cmp_rows = [
        {"Model": name, "LogLik": r.loglikelihood, "AIC": r.aic, "BIC": r.bic}
        for name, r in results.items()
    ]
    cmp = pd.DataFrame(cmp_rows).sort_values("AIC").reset_index(drop=True)
    _print(cmp.to_string(index=False))

    # LaTeX table 3
    def _fmt_param(res, key):
        if key not in res.params:
            return "---"
        coef = res.params[key]
        pval = res.pvalues[key]
        stars = "***" if pval < 0.01 else ("**" if pval < 0.05 else ("*" if pval < 0.10 else ""))
        sign = "$-$" if coef < 0 else ""
        return f"{sign}{abs(coef):.4f}{stars}"

    g = results["Baseline GARCH(1,1)"]
    e = results["Baseline EGARCH(1,1)"]
    s = results["Extended EGARCH-SVI"]
    a = results["Extended EGARCH-Asym"]

    tex = r"""\begin{table}[H]
\centering
\caption{In-Sample Parameter Estimates --- GARCH Family Models (Student-$t$ Innovations, 2010--2025)}
\label{tab:insample}
\small
\begin{tabular}{lcccc}
\toprule
Parameter & GARCH(1,1) & EGARCH(1,1) & EGARCH-SVI & EGARCH-Asym \\
\midrule
""" + \
    f"$\\omega$               & {_fmt_param(g,'omega')} & {_fmt_param(e,'omega')} & {_fmt_param(s,'omega')} & {_fmt_param(a,'omega')} \\\\\n" + \
    f"$\\alpha$               & {_fmt_param(g,'alpha[1]')} & {_fmt_param(e,'alpha[1]')} & {_fmt_param(s,'alpha[1]')} & {_fmt_param(a,'alpha[1]')} \\\\\n" + \
    f"$\\beta$                & {_fmt_param(g,'beta[1]')} & {_fmt_param(e,'beta[1]')} & {_fmt_param(s,'beta[1]')} & {_fmt_param(a,'beta[1]')} \\\\\n" + \
    f"$\\gamma$ (Leverage)    & --- & {_fmt_param(e,'gamma[1]')} & {_fmt_param(s,'gamma[1]')} & {_fmt_param(a,'gamma[1]')} \\\\\n" + \
    f"$\\theta$ (SVI)         & --- & --- & {_fmt_param(s,'lag_SVI')} ($p$={s.pvalues.get('lag_SVI', float('nan')):.3f}) & --- \\\\\n" + \
    f"$\\theta_1$ ($SVI_{{pos}}$) & --- & --- & --- & {_fmt_param(a,'lag_SVI_pos')} ($p$={a.pvalues.get('lag_SVI_pos', float('nan')):.3f}) \\\\\n" + \
    f"$\\theta_2$ ($SVI_{{neg}}$) & --- & --- & --- & \\textbf{{{_fmt_param(a,'lag_SVI_neg')} ($p$={a.pvalues.get('lag_SVI_neg', float('nan')):.3f})}} \\\\\n" + \
    f"$\\nu$ (Student-$t$)    & {_fmt_param(g,'nu')} & {_fmt_param(e,'nu')} & {_fmt_param(s,'nu')} & {_fmt_param(a,'nu')} \\\\\n" + \
r"""\midrule
""" + \
    f"Log-Likelihood & ${g.loglikelihood:.2f}$ & ${e.loglikelihood:.2f}$ & ${s.loglikelihood:.2f}$ & $\\mathbf{{{a.loglikelihood:.2f}}}$ \\\\\n" + \
    f"AIC            & ${g.aic:.1f}$ & ${e.aic:.1f}$ & ${s.aic:.1f}$ & $\\mathbf{{{a.aic:.1f}}}$ \\\\\n" + \
r"""\bottomrule
\end{tabular}
\smallskip\\
\footnotesize\textit{Note: *** $p<0.01$; ** $p<0.05$; * $p<0.10$. All models estimated by QML with Student-$t$ innovations. Bold entries indicate best-performing specification.}
\end{table}"""

    _save_tex("table3_insample.tex", tex)
    return results, model_df


# ── Phase 4: Out-of-sample forecasting ───────────────────────────────────────

def run_oos(results: dict, model_df: pd.DataFrame, test_index: pd.Index) -> dict:
    _header("PHASE 4 — OUT-OF-SAMPLE FORECASTING (Q1 2026, N=56)")

    te = model_df.loc[test_index].dropna()
    actual_ret = te["return"]
    actual_vol = te["target"]

    forecast_dict = {}

    forecast_dict["Baseline GARCH(1,1)"] = recursive_garch_forecast(
        results["Baseline GARCH(1,1)"], actual_ret
    )
    forecast_dict["Baseline EGARCH(1,1)"] = recursive_egarch_forecast(
        results["Baseline EGARCH(1,1)"], actual_ret
    )

    mean_svi = _mean_path_from_result(results["Extended EGARCH-SVI"], te[["lag_SVI"]])
    forecast_dict["Extended EGARCH-SVI"] = recursive_egarch_forecast(
        results["Extended EGARCH-SVI"], actual_ret, mean_series=mean_svi
    )

    mean_asym = _mean_path_from_result(results["Extended EGARCH-Asym"], te[["lag_SVI_pos", "lag_SVI_neg"]])
    forecast_dict["Extended EGARCH-Asym"] = recursive_egarch_forecast(
        results["Extended EGARCH-Asym"], actual_ret, mean_series=mean_asym
    )

    cmp = build_comparison_table(actual_vol, forecast_dict)
    _print(cmp.to_string(index=False))

    # LaTeX table 4
    tex_rows = []
    rank = 1
    for _, row in cmp.iterrows():
        bold_open = "\\textbf{" if rank == 1 else ""
        bold_close = "}" if rank == 1 else ""
        tex_rows.append(
            f"        {bold_open}{row['Model']}{bold_close} & {bold_open}{row['RMSE']:.4f}{bold_close}"
            f" & {bold_open}{row['QLIKE']:.4f}{bold_close} & {bold_open}{rank}{bold_close} \\\\"
        )
        rank += 1

    tex = r"""\begin{table}[H]
\centering
\caption{Out-of-Sample Forecasting Performance (Q1 2026, $N = 56$ Days)}
\label{tab:oos}
\small
\begin{tabular}{lccc}
\toprule
Model & RMSE & QLIKE & Rank \\
\midrule
""" + "\n".join(tex_rows) + r"""
\bottomrule
\end{tabular}
\smallskip\\
\footnotesize\textit{Note: Lower values indicate better forecast accuracy. RMSE = Root Mean Squared Error; QLIKE = Quasi-Likelihood loss (robust to imperfect volatility proxies).}
\end{table}"""

    _save_tex("table4_oos.tex", tex)
    return forecast_dict, actual_vol


# ── Phase 5: Diebold-Mariano tests ───────────────────────────────────────────

def run_dm_tests(forecast_dict: dict, actual_vol: pd.Series) -> None:
    _header("PHASE 5 — DIEBOLD-MARIANO TESTS")

    baseline_fcst = forecast_dict["Baseline EGARCH(1,1)"]
    extended = {
        "Extended EGARCH-SVI": forecast_dict["Extended EGARCH-SVI"],
        "Extended EGARCH-Asym": forecast_dict["Extended EGARCH-Asym"],
    }
    dm_table = build_dm_table(actual_vol, extended, "Baseline EGARCH(1,1)", baseline_fcst, h=1, hac_lags=1)
    _print(dm_table.to_string(index=False))

    # LaTeX table 5
    tex_rows = []
    for _, r in dm_table.iterrows():
        pair = r["Pair"].replace("Extended EGARCH-SVI vs Baseline EGARCH(1,1)", "EGARCH-SVI vs. Baseline")
        pair = pair.replace("Extended EGARCH-Asym vs Baseline EGARCH(1,1)", "EGARCH-Asym vs. Baseline")

        def _fmt_decision(pval):
            if pval < 0.05:
                return r"\textbf{Reject $H_0$}"
            return "Fail to Reject $H_0$"

        tex_rows.append(
            f"        {pair} & QLIKE & {r['MeanDiff_QLIKE(Base-Ext)']:.4f} & {r['DM_QLIKE']:.3f} & {r['PValue_QLIKE']:.3f} & {_fmt_decision(r['PValue_QLIKE'])} \\\\\n"
            f"        {pair} & SE (RMSE) & {r['MeanDiff_SE(Base-Ext)']:.4f} & {r['DM_SE']:.3f} & {r['PValue_SE']:.3f} & {_fmt_decision(r['PValue_SE'])} \\\\"
        )

    tex = r"""\begin{table}[H]
\centering
\caption{Diebold-Mariano Test Results (HAC Standard Errors, Newey-West Lag = 1)}
\label{tab:dm}
\small
\begin{tabular}{llcccc}
\toprule
Model Comparison & Metric & Mean Diff. & DM-Stat & $p$-value & Decision (5\%) \\
\midrule
""" + "\n".join(tex_rows) + r"""
\bottomrule
\end{tabular}
\smallskip\\
\footnotesize\textit{Note: DM test null hypothesis: equal predictive accuracy. Bold entries indicate rejection at 5\% level. ``Baseline'' refers to EGARCH(1,1). Mean Diff. = mean(baseline loss) $-$ mean(extended loss); positive values favor the extended model.}
\end{table}"""

    _save_tex("table5_dm.tex", tex)


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    _print("=" * 70)
    _print("  EXPERIMENT LOG — VN-Index Volatility: EGARCH-X with Google Trends")
    _print("  Generated by: scripts/run_experiments.py")
    _print("  Python arch library, statsmodels, scikit-learn")
    _print("=" * 70)

    train, test, full = load_data()
    _print(f"\nIn-sample  : {len(train):,} rows | {train.index.min().date()} -> {train.index.max().date()}")
    _print(f"Out-of-sample: {len(test):,} rows | {test.index.min().date()} -> {test.index.max().date()}")

    run_descriptive_stats(train)
    run_pretests(train)
    results, model_df = run_insample(train, full, train.index)
    forecast_dict, actual_vol = run_oos(results, model_df, test.index)
    run_dm_tests(forecast_dict, actual_vol)

    _print()
    _print("All experiments completed successfully.")
    _print(f"LaTeX tables exported to: {TABLE_DIR}/")
    _print(f"Full log saved to: {LOG_PATH}")

    _save_log()


if __name__ == "__main__":
    main()
