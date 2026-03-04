#!/usr/bin/env python3
"""
Demo pipeline (NO real data): forecasting + stockout risk + EOQ/ROP.
Generates synthetic store-product daily data, trains simple models, and prints example decisions.
"""

from __future__ import annotations

import math
import numpy as np
import pandas as pd

from sklearn.linear_model import Lasso, LogisticRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, roc_auc_score


# -------------------- Config --------------------
RANDOM_SEED = 42
N_DAYS = 240
N_STORES = 3
N_PRODUCTS = 6

LAG_DAYS = [1, 7, 14]
ROLL_WINDOWS = [7, 14]

LEAD_TIME_DAYS = 7
SERVICE_LEVEL_Z = 1.65  # ~95%

ORDER_COST_S = 80.0      # cost per order (example)
HOLDING_COST_H = 2.0     # cost per unit-year (example)

STOCKOUT_THRESHOLD = 0


# -------------------- Helpers --------------------
def eoq(D_annual: float, S: float, H: float) -> float:
    """Economic Order Quantity."""
    if D_annual <= 0 or S <= 0 or H <= 0:
        return 0.0
    return math.sqrt((2.0 * D_annual * S) / H)


def reorder_point(mu_daily: float, lead_time_days: int, sigma_L: float, z: float) -> float:
    """ROP = mu*L + z*sigma_L."""
    return mu_daily * lead_time_days + z * sigma_L


def make_synthetic_data(seed: int = RANDOM_SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2023-01-01", periods=N_DAYS, freq="D")

    rows = []
    for s in range(1, N_STORES + 1):
        for p in range(1, N_PRODUCTS + 1):
            base = rng.uniform(20, 120)
            weekly = rng.uniform(0.0, 20.0)
            noise = rng.uniform(5.0, 25.0)

            stock = float(rng.integers(200, 600))
            for i, d in enumerate(dates):
                dow = d.dayofweek
                season = 15.0 * math.sin(2 * math.pi * (i / 365.0))
                demand = base + weekly * (1 if dow in (4, 5) else -0.3) + season + rng.normal(0, noise)
                demand = max(0.0, float(demand))

                sold = min(stock, demand + float(rng.normal(0, 3)))
                sold = max(0.0, float(sold))
                closing = stock - sold

                restock = 0.0
                if i % 21 == 0 and i > 0:
                    restock = float(rng.integers(150, 450))
                    closing += restock

                rows.append({
                    "date": d,
                    "store_id": f"S{s:03d}",
                    "product_id": f"P{p:04d}",
                    "units_sold": sold,
                    "closing_stock": float(closing),
                    "restocked": restock,
                    "day_of_week": int(dow),
                    "month": int(d.month),
                })

                stock = float(closing)

    df = pd.DataFrame(rows).sort_values(["store_id", "product_id", "date"])
    return df.reset_index(drop=True)


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    grp = df.groupby(["store_id", "product_id"], sort=False)

    for lag in LAG_DAYS:
        df[f"lag_{lag}"] = grp["units_sold"].shift(lag)

    for w in ROLL_WINDOWS:
        df[f"roll_mean_{w}"] = grp["units_sold"].shift(1).rolling(w).mean()
        df[f"roll_std_{w}"] = grp["units_sold"].shift(1).rolling(w).std()

    df["stock_lag_1"] = grp["closing_stock"].shift(1)
    df["stock_change_1"] = df["closing_stock"] - df["stock_lag_1"]

    df["target_next_day"] = grp["units_sold"].shift(-1)

    return df.dropna().reset_index(drop=True)


def build_stockout_label(df: pd.DataFrame) -> pd.Series:
    mu = df["roll_mean_7"].fillna(df["units_sold"].mean())
    projected = df["closing_stock"] - (LEAD_TIME_DAYS * mu)
    return (projected <= STOCKOUT_THRESHOLD).astype(int)


def train_forecast_model(df: pd.DataFrame):
    feature_cols = [
        "day_of_week", "month",
        "closing_stock", "stock_lag_1", "stock_change_1",
        *[f"lag_{k}" for k in LAG_DAYS],
        *[f"roll_mean_{w}" for w in ROLL_WINDOWS],
        *[f"roll_std_{w}" for w in ROLL_WINDOWS],
    ]

    X = df[feature_cols].astype(float)
    y = df["target_next_day"].astype(float)

    split_idx = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    model = Lasso(alpha=0.001, random_state=RANDOM_SEED, max_iter=20000)
    model.fit(X_train, y_train)

    pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, pred)
    rmse = math.sqrt(mean_squared_error(y_test, pred))
    r2 = r2_score(y_test, pred)

    print("\nForecasting (demo)")
    print(f"MAE  = {mae:.3f}")
    print(f"RMSE = {rmse:.3f}")
    print(f"R^2  = {r2:.3f}")

    return model


def train_stockout_model(df: pd.DataFrame, forecast_model: Lasso):
    feature_cols = [
        "day_of_week", "month",
        "closing_stock", "stock_lag_1",
        *[f"lag_{k}" for k in LAG_DAYS],
        "roll_mean_7", "roll_std_7",
    ]

    X = df[feature_cols].astype(float).copy()
    X_fore = df[[
        "day_of_week", "month",
        "closing_stock", "stock_lag_1", "stock_change_1",
        *[f"lag_{k}" for k in LAG_DAYS],
        *[f"roll_mean_{w}" for w in ROLL_WINDOWS],
        *[f"roll_std_{w}" for w in ROLL_WINDOWS],
    ]].astype(float)

    X["forecast_next_day"] = forecast_model.predict(X_fore)
    y = build_stockout_label(df)

    split_idx = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    clf = LogisticRegression(max_iter=2000)
    clf.fit(X_train, y_train)

    proba = clf.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, proba)

    print("\nStockout risk (demo)")
    print(f"ROC-AUC = {auc:.3f}")

    return clf


def example_decision(df: pd.DataFrame, forecast_model: Lasso, stockout_model: LogisticRegression) -> None:
    row = df.iloc[-1:].copy()

    feat_fore = row[[
        "day_of_week", "month",
        "closing_stock", "stock_lag_1", "stock_change_1",
        *[f"lag_{k}" for k in LAG_DAYS],
        *[f"roll_mean_{w}" for w in ROLL_WINDOWS],
        *[f"roll_std_{w}" for w in ROLL_WINDOWS],
    ]].astype(float)

    d_hat = float(forecast_model.predict(feat_fore)[0])

    feat_risk = row[[
        "day_of_week", "month",
        "closing_stock", "stock_lag_1",
        *[f"lag_{k}" for k in LAG_DAYS],
        "roll_mean_7", "roll_std_7",
    ]].astype(float).assign(forecast_next_day=d_hat)

    p_stockout = float(stockout_model.predict_proba(feat_risk)[:, 1][0])

    mu_d = float(row["roll_mean_7"].iloc[0])
    sigma_d = float(row["roll_std_7"].iloc[0])
    sigma_L = math.sqrt(LEAD_TIME_DAYS) * (sigma_d if not math.isnan(sigma_d) else 0.0)

    rop = reorder_point(mu_d, LEAD_TIME_DAYS, sigma_L, SERVICE_LEVEL_Z)
    D_annual = max(0.0, mu_d * 365.0)
    q_star = eoq(D_annual, ORDER_COST_S, HOLDING_COST_H)

    closing_stock = float(row["closing_stock"].iloc[0])
    should_reorder = closing_stock < rop

    print("\nExample decision (demo)")
    print(f"Store  : {row['store_id'].iloc[0]}")
    print(f"Product: {row['product_id'].iloc[0]}")
    print(f"Date   : {row['date'].iloc[0].date()}")
    print(f"Closing stock      : {closing_stock:.1f}")
    print(f"Forecast (next day): {d_hat:.1f}")
    print(f"Stockout risk      : {p_stockout:.3f}")
    print(f"ROP (L={LEAD_TIME_DAYS})      : {rop:.1f}")
    print(f"EOQ (Q*)           : {q_star:.1f}")
    print(f"Decision           : {'REORDER' if should_reorder else 'NO ORDER'}")


def main():
    df_raw = make_synthetic_data()
    df = add_features(df_raw)

    forecast_model = train_forecast_model(df)
    stockout_model = train_stockout_model(df, forecast_model)

    example_decision(df, forecast_model, stockout_model)


if __name__ == "__main__":
    main()
