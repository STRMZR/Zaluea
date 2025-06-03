#!/usr/bin/env python3
"""Generate trading signals using MACD and DecisionTreeClassifier."""

import argparse
import logging
from dataclasses import dataclass
from typing import List, Dict

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler


def compute_macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    """Return MACD line, signal line, and histogram."""
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    hist = macd_line - signal_line
    return macd_line, signal_line, hist


def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """Create a set of features. Extendable for additional indicators."""
    features = pd.DataFrame(index=df.index)
    features["return_1"] = df["Close"].pct_change()
    features["return_3"] = df["Close"].pct_change(3)

    volume_mean = df["Volume"].rolling(window=20).mean()
    features["vol_ratio"] = df["Volume"] / volume_mean

    ma20 = df["Close"].rolling(window=20).mean()
    std20 = df["Close"].rolling(window=20).std()
    upper = ma20 + 2 * std20
    lower = ma20 - 2 * std20
    features["bb_pos"] = (df["Close"] - lower) / (upper - lower)

    return features.fillna(0)


@dataclass
class Signal:
    timestamp: str
    ticker: str
    signal: str
    prediction: int
    macd_hist: float
    profit_after_commission: float


def sliding_window_signals(
    df: pd.DataFrame,
    ticker: str,
    commission: float = 0.25,
    train_size: int = 100,
    test_size: int = 30,
) -> List[Dict]:
    macd_line, signal_line, macd_hist = compute_macd(df["Close"])
    features = feature_engineering(df)

    scaler = StandardScaler()
    start = 0
    signals: List[Dict] = []

    while start + train_size + test_size <= len(df):
        train_slice = slice(start, start + train_size)
        test_slice = slice(start + train_size, start + train_size + test_size)

        X_train = scaler.fit_transform(features.iloc[train_slice])
        y_train = np.where(
            df["Close"].shift(-1).iloc[train_slice] > df["Close"].iloc[train_slice],
            1,
            -1,
        )

        model = DecisionTreeClassifier()
        model.fit(X_train, y_train)

        X_test = scaler.transform(features.iloc[test_slice])
        preds = model.predict(X_test)

        for idx, pred in zip(features.index[test_slice], preds):
            macd_val = macd_hist.loc[idx]
            action = None
            if macd_val > 0 and pred == 1:
                action = "BUY"
            elif macd_val < 0 and pred == -1:
                action = "SELL"

            if action:
                curr_price = df.loc[idx, "Close"]
                next_idx = df.index.get_loc(idx) + 1
                if next_idx < len(df):
                    next_price = df.iloc[next_idx]["Close"]
                    pnl = (next_price - curr_price) * (1 if action == "BUY" else -1)
                else:
                    pnl = 0
                pnl -= commission

                signals.append(
                    {
                        "timestamp": idx.strftime("%Y-%m-%d"),
                        "ticker": ticker,
                        "signal": action,
                        "prediction": int(pred),
                        "macd_hist": float(round(macd_val, 5)),
                        "profit_after_commission": float(round(pnl, 5)),
                    }
                )
                logging.info(
                    f"{idx.date()} {ticker} {action} pred={pred} MACD={macd_val:.5f} pnl={pnl:.5f}"
                )
        start += test_size

    return signals


def fetch_data(ticker: str, period: str = "2y") -> pd.DataFrame:
    """Download daily OHLCV data and flatten potential MultiIndex columns."""
    df = yf.download(ticker, period=period, interval="1d", auto_adjust=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="MACD + DecisionTree signal generator")
    parser.add_argument("--logfile", help="Optional log file path")
    parser.add_argument("--period", default="2y", help="Data period to download")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        filename=args.logfile,
        format="%(asctime)s %(message)s",
    )

    tickers = {"ES": "ES=F", "NKD": "NKD=F"}
    all_signals: List[Dict] = []
    for name, yf_ticker in tickers.items():
        df = fetch_data(yf_ticker, period=args.period)
        if df.empty:
            logging.warning(f"No data for {name}")
            continue
        sigs = sliding_window_signals(df, name)
        all_signals.extend(sigs)

    for sig in all_signals:
        print(sig)


if __name__ == "__main__":
    main()
