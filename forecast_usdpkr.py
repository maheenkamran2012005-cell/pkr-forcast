#!/usr/bin/env python
"""
Two-stage CatBoost forecast of the interbank USD/PKR rate.

Stage 1  Forecast every macro factor N months ahead (CatBoost, one model per
         factor, recursive month-by-month).
Stage 2  Train CatBoost: interbank rate  <-  macro factors (historical data).
         Feed it the Stage-1 factor forecasts to get the future interbank rate.

Usage
-----
    python forecast_usdpkr.py --data data/train_data_.xlsx --horizon 24

Outputs (in ./outputs by default)
    forecast_factors.csv        predicted factors, next N months
    forecast_interbank.csv      predicted interbank rate, next N months
    backtest_by_horizon.csv     walk-forward backtest errors
    driver_importance.csv/.png  which factors drive the rate model
    *.png                       charts
"""
import argparse
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

warnings.filterwarnings("ignore")

# --------------------------------------------------------------------------
# 1. Column configuration
# --------------------------------------------------------------------------
RENAME = {
    "Date": "date",
    "Foreign reserves": "reserves",
    "Trade Balance (Monthly)": "trade_balance",
    "Workers' Remittances (Million US$)": "remittances",
    "Capital Flows (FDI + Portfolio Investment)": "capital_flows",
    "REER (Real Effective Exchange Rate)": "reer",
    "Pakistan External Debt (Million US$)": "ext_debt",
    "Pakistan: Political Stability (Index, Recent Values)": "political_stability",
    "Fiscal Deficit - Fiscal Balance (% of GDP)": "fiscal_deficit",
    "Historical Annual Imports (USD)": "annual_imports",
    "DXY (US Dollar Index) - Close": "dxy",
    "VIX (CBOE Volatility Index) - Close": "vix",
    "Brent Crude Oil Price": "brent",
    "CDS Spread (Credit Default Swap Spread)": "cds",
    "M2 Growth (Growth in Broad Money Supply (M2))": "m2_growth",
    "Credit Rating - Numeric Score": "credit_rating",
    "IMF Status - IMF Program Active": "imf_active",
    "Import Cover Ratio": "import_cover",
    "Inflation rate differential": "infl_diff",
    "INTERBANK USD/PKR - Close": "usdpkr",
    # Not used as drivers (they are computed from / move together with the
    # target itself, so using them would be data leakage):
    "Historical/Realized FX Volatility": "fx_vol",
    "Log return of usd/pkr": "logret_given",
    "Kerb Market Premium": "kerb_premium",
}
TARGET = "usdpkr"
LEAKY = ["fx_vol", "logret_given", "kerb_premium"]

# Factors forecast with CatBoost, predicting the month-on-month CHANGE
# (trending series: a tree cannot extrapolate a level outside its training range)
DIFF_FACTORS = ["reserves", "reer", "dxy", "brent", "cds", "import_cover"]
# Factors forecast with CatBoost, predicting the LEVEL (flows / rates that mean-revert)
LEVEL_FACTORS = ["trade_balance", "remittances", "capital_flows", "ext_debt",
                 "vix", "m2_growth", "infl_diff"]
# Step-like / policy variables with only a handful of distinct values.
# Not forecastable by ML; held at the last observed value (edit if you have a view).
STEP_FACTORS = ["political_stability", "fiscal_deficit", "annual_imports",
                "credit_rating", "imf_active"]

CONT_FACTORS = DIFF_FACTORS + LEVEL_FACTORS
ALL_FACTORS = CONT_FACTORS + STEP_FACTORS


# --------------------------------------------------------------------------
# 2. Data
# --------------------------------------------------------------------------
def load_data(path):
    raw = pd.read_excel(path)
    missing = [c for c in RENAME if c not in raw.columns]
    if missing:
        raise ValueError(f"Columns not found in the Excel file: {missing}")
    df = raw.rename(columns=RENAME)[list(RENAME.values())]
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").set_index("date")
    df.index = df.index.to_period("M").to_timestamp()      # month-start
    df = df.asfreq("MS")
    if df[ALL_FACTORS + [TARGET]].isna().any().any():
        bad = df[ALL_FACTORS + [TARGET]].isna().sum()
        print("Warning: missing values found, forward-filling:\n", bad[bad > 0])
        df[ALL_FACTORS + [TARGET]] = df[ALL_FACTORS + [TARGET]].ffill()
    return df


def new_model(seed):
    return CatBoostRegressor(
        iterations=500, depth=4, learning_rate=0.03, l2_leaf_reg=5,
        loss_function="RMSE", random_seed=seed, verbose=0,
        allow_writing_files=False,
    )


# --------------------------------------------------------------------------
# 3. Stage 1 - factor forecasting models
# --------------------------------------------------------------------------
def series_features(s):
    """Features for predicting s[t] using information up to t-1 only."""
    X = pd.DataFrame(index=s.index)
    for l in (1, 2, 3, 6, 12):
        X[f"lag{l}"] = s.shift(l)
    d = s.diff()
    for l in (1, 2, 3):
        X[f"diff_lag{l}"] = d.shift(l)
    X["roll3"] = s.rolling(3).mean().shift(1)
    X["roll6"] = s.rolling(6).mean().shift(1)
    X["month"] = s.index.month
    return X


def fit_factor_models(df, seed):
    models = {}
    for f in CONT_FACTORS:
        s = df[f]
        y = s.diff() if f in DIFF_FACTORS else s
        X = series_features(s)
        m = y.notna()
        models[f] = new_model(seed).fit(X[m], y[m])
    return models


def forecast_factors(hist, models, horizon):
    """Recursive month-by-month forecast of all factors."""
    h = hist[ALL_FACTORS].copy()
    future_idx = pd.date_range(h.index[-1] + pd.offsets.MonthBegin(1),
                               periods=horizon, freq="MS")
    for dt in future_idx:
        row = {}
        for f in CONT_FACTORS:
            s = h[f]
            s_ext = pd.concat([s, pd.Series([np.nan], index=[dt])])
            x_last = series_features(s_ext).iloc[[-1]]
            pred = float(models[f].predict(x_last)[0])
            row[f] = s.iloc[-1] + pred if f in DIFF_FACTORS else pred
        for f in STEP_FACTORS:
            row[f] = h[f].iloc[-1]
        h.loc[dt, list(row)] = list(row.values())
    return h.loc[future_idx]


# --------------------------------------------------------------------------
# 4. Stage 2 - interbank rate = f(factors)
# --------------------------------------------------------------------------
def driver_features(F):
    X = F[ALL_FACTORS].copy()
    for c in CONT_FACTORS:
        X[f"{c}_chg3"] = F[c].diff(3)
    X["month"] = F.index.month
    return X


def driver_target(df, mode):
    if mode == "logret":                      # month-on-month log return
        return np.log(df[TARGET]).diff()
    return df[TARGET]                          # "level"


def fit_driver_model(df, mode, seed):
    X = driver_features(df)
    y = driver_target(df, mode)
    m = y.notna() & X.notna().any(axis=1)
    return new_model(seed).fit(X[m], y[m])


def to_level(pred, last_level, mode):
    pred = np.asarray(pred)
    return last_level * np.exp(np.cumsum(pred)) if mode == "logret" else pred


def run_pipeline(df_train, horizon, mode, seed):
    fmodels = fit_factor_models(df_train, seed)
    fut_factors = forecast_factors(df_train, fmodels, horizon)
    dmodel = fit_driver_model(df_train, mode, seed)

    full = pd.concat([df_train[ALL_FACTORS], fut_factors])
    X_fut = driver_features(full).loc[fut_factors.index]
    pred = dmodel.predict(X_fut)
    rate = pd.Series(to_level(pred, df_train[TARGET].iloc[-1], mode),
                     index=fut_factors.index, name=TARGET)
    return fut_factors, rate, dmodel


# --------------------------------------------------------------------------
# 5. Walk-forward backtest
# --------------------------------------------------------------------------
def backtest(df, mode, seed, horizon=12, n_origins=6, step=6):
    n = len(df)
    origins = [n - horizon - step * k for k in range(n_origins)][::-1]
    rows = []
    for i in origins:
        train, test = df.iloc[:i], df.iloc[i:i + horizon]
        if len(train) < 40 or len(test) < horizon:
            continue
        _, rate_fc, dmodel = run_pipeline(train, horizon, mode, seed)
        # "oracle": same rate model but fed the TRUE future factors
        X_true = driver_features(df[ALL_FACTORS]).loc[test.index]
        oracle = to_level(dmodel.predict(X_true), train[TARGET].iloc[-1], mode)
        naive = train[TARGET].iloc[-1]
        for k, dt in enumerate(test.index, start=1):
            rows.append(dict(origin=train.index[-1], h=k, date=dt,
                             actual=test[TARGET].iloc[k - 1],
                             two_stage=rate_fc.iloc[k - 1],
                             oracle_true_factors=oracle[k - 1],
                             naive_last_value=naive))
        print(f"  backtest origin {train.index[-1]:%Y-%m} done")
    bt = pd.DataFrame(rows)
    if bt.empty:
        return bt, pd.DataFrame()
    for c in ["two_stage", "oracle_true_factors", "naive_last_value"]:
        bt[f"ape_{c}"] = (bt[c] - bt["actual"]).abs() / bt["actual"] * 100
    summ = bt.groupby("h")[[c for c in bt if c.startswith("ape_")]].mean().round(2)
    summ.columns = [c.replace("ape_", "MAPE%_") for c in summ.columns]
    return bt, summ


# --------------------------------------------------------------------------
# 6. Plots
# --------------------------------------------------------------------------
def make_plots(df, fut_factors, rate, dmodel, summ, out):
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(df.index, df[TARGET], label="History")
    ax.plot(rate.index, rate.values, "r--", label="Forecast")
    ax.axvline(df.index[-1], color="grey", lw=0.8)
    ax.set_title("Interbank USD/PKR - history and forecast")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(out / "forecast_interbank.png", dpi=150); plt.close(fig)

    cols = CONT_FACTORS + STEP_FACTORS
    nr = int(np.ceil(len(cols) / 4))
    fig, axes = plt.subplots(nr, 4, figsize=(18, 3.2 * nr))
    for a, c in zip(axes.ravel(), cols):
        a.plot(df.index, df[c], lw=1)
        a.plot(fut_factors.index, fut_factors[c], "r--", lw=1)
        a.set_title(c, fontsize=9)
    for a in axes.ravel()[len(cols):]:
        a.axis("off")
    fig.tight_layout(); fig.savefig(out / "forecast_factors.png", dpi=130); plt.close(fig)

    imp = pd.Series(dmodel.get_feature_importance(), index=dmodel.feature_names_)
    imp = imp.sort_values(ascending=False)
    imp.rename("importance").to_csv(out / "driver_importance.csv")
    fig, ax = plt.subplots(figsize=(8, 7))
    imp.head(20)[::-1].plot.barh(ax=ax)
    ax.set_title("CatBoost importance - what drives the interbank rate")
    fig.tight_layout(); fig.savefig(out / "driver_importance.png", dpi=150); plt.close(fig)

    if summ is not None and len(summ):
        fig, ax = plt.subplots(figsize=(8, 4.5))
        summ.plot(ax=ax, marker="o")
        ax.set_xlabel("months ahead"); ax.set_ylabel("MAPE %")
        ax.set_title("Walk-forward backtest error"); ax.grid(alpha=0.3)
        fig.tight_layout(); fig.savefig(out / "backtest_error.png", dpi=150); plt.close(fig)


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/train_data_.xlsx")
    ap.add_argument("--horizon", type=int, default=24, help="months to forecast")
    ap.add_argument("--target-mode", choices=["logret", "level"], default="logret",
                    help="logret: model monthly log-return then rebuild level (recommended); "
                         "level: model the rate level directly")
    ap.add_argument("--outdir", default="outputs")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--skip-backtest", action="store_true")
    a = ap.parse_args()

    out = Path(a.outdir); out.mkdir(parents=True, exist_ok=True)
    df = load_data(a.data)
    print(f"Loaded {len(df)} months: {df.index[0]:%Y-%m} -> {df.index[-1]:%Y-%m}")

    summ = None
    if not a.skip_backtest:
        print("Running walk-forward backtest (12 months ahead)...")
        bt, summ = backtest(df, a.target_mode, a.seed)
        if len(summ):
            bt.to_csv(out / "backtest_detail.csv", index=False)
            summ.to_csv(out / "backtest_by_horizon.csv")
            print("\nBacktest MAPE % by months-ahead:\n", summ, "\n")

    print(f"Fitting on all data and forecasting {a.horizon} months...")
    fut_factors, rate, dmodel = run_pipeline(df, a.horizon, a.target_mode, a.seed)

    fut_factors.to_csv(out / "forecast_factors.csv", index_label="date")
    rate.to_csv(out / "forecast_interbank.csv", index_label="date", header=True)
    make_plots(df, fut_factors, rate, dmodel, summ, out)
    print(rate.round(2).to_string())
    print(f"\nAll outputs saved in: {out.resolve()}")


if __name__ == "__main__":
    main()
