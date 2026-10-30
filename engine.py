# -*- coding: utf-8 -*-
"""MA × Breadth PROXY × VIX backtest engine (research tool, not advice)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

TRADING_DAYS = 252
VIX_TICKER = "^VIX"


@dataclass
class BacktestResult:
    equity: pd.Series
    returns: pd.Series
    weights: pd.DataFrame
    trades: pd.DataFrame
    metrics: dict
    breadth_proxy: Optional[pd.Series] = None
    vix: Optional[pd.Series] = None
    prices: Optional[pd.DataFrame] = None
    hold_mask: Optional[pd.DataFrame] = None


def download_prices(tickers: list[str], start: str, end: Optional[str]) -> pd.DataFrame:
    import yfinance as yf

    tickers = [t.strip().upper() for t in tickers if t and str(t).strip()]
    if not tickers:
        raise ValueError("至少需要一個股票代號")

    kwargs: dict = {"start": start, "auto_adjust": True, "progress": False, "threads": True}
    if end:
        kwargs["end"] = end

    raw = yf.download(tickers if len(tickers) > 1 else tickers[0], **kwargs)
    if raw.empty:
        raise RuntimeError("yfinance 回傳空表，請檢查網絡或 ticker。")

    if isinstance(raw.columns, pd.MultiIndex):
        level0 = raw.columns.get_level_values(0)
        if "Close" in level0:
            px = raw["Close"].copy()
        elif "Adj Close" in level0:
            px = raw["Adj Close"].copy()
        else:
            raise RuntimeError(f"未知欄位結構: {raw.columns}")
        if isinstance(px, pd.Series):
            px = px.to_frame(name=tickers[0])
    else:
        col = "Close" if "Close" in raw.columns else "Adj Close"
        px = raw[[col]].copy()
        px.columns = [tickers[0]]

    px = px.sort_index().ffill(limit=5)
    px = px.dropna(axis=1, how="all")
    # ensure column order / names match requested when present
    ordered = [t for t in tickers if t in px.columns]
    if not ordered:
        raise RuntimeError(f"無法取得任何 ticker 價格：{tickers}")
    return px[ordered]


def download_series(ticker: str, start: str, end: Optional[str], name: str) -> pd.Series:
    import yfinance as yf

    kwargs: dict = {"start": start, "auto_adjust": True, "progress": False}
    if end:
        kwargs["end"] = end
    df = yf.download(ticker, **kwargs)
    if df.empty:
        raise RuntimeError(f"無法下載 {ticker}")
    if isinstance(df.columns, pd.MultiIndex):
        s = df["Close"].iloc[:, 0] if "Close" in df.columns.get_level_values(0) else df.iloc[:, 0]
    else:
        s = df["Close"] if "Close" in df.columns else df.iloc[:, 0]
    return s.squeeze().rename(name).sort_index()


def compute_ma_signals(
    px: pd.DataFrame, ma_short: int, ma_long: int, use_cross: bool
) -> pd.DataFrame:
    sig = pd.DataFrame(index=px.index, columns=px.columns, dtype=bool)
    for col in px.columns:
        s = px[col]
        ma_s = s.rolling(ma_short, min_periods=ma_short).mean()
        ma_l = s.rolling(ma_long, min_periods=ma_long).mean()
        if use_cross:
            sig[col] = (ma_s > ma_l).fillna(False)
        else:
            sig[col] = (s > ma_l).fillna(False)
    return sig


def compute_breadth_proxy(px: pd.DataFrame, window: int) -> pd.Series:
    """PROXY only: % of universe above own MA — not official exchange breadth."""
    above = pd.DataFrame(index=px.index, columns=px.columns, dtype=float)
    for col in px.columns:
        ma = px[col].rolling(window, min_periods=window).mean()
        above[col] = (px[col] > ma).astype(float)
    valid = above.notna()
    pct = above.sum(axis=1) / valid.sum(axis=1).replace(0, np.nan)
    return pct.rename("breadth_proxy_pct_above_ma")


def combine_signals(
    ma_sig: pd.DataFrame,
    breadth: pd.Series,
    vix_s: pd.Series,
    require_breadth: bool,
    breadth_min: float,
    require_vix: bool,
    vix_off: float,
) -> pd.DataFrame:
    out = ma_sig.copy()
    if require_breadth:
        b_ok = (breadth >= breadth_min).reindex(out.index).fillna(False)
        out = out.mul(b_ok, axis=0)
    if require_vix:
        v_ok = (vix_s < vix_off).reindex(out.index).fillna(False)
        out = out.mul(v_ok, axis=0)
    return out.astype(bool)


def normalize_weights(tickers: list[str], weights: Optional[dict[str, float]] = None) -> pd.Series:
    if weights is None:
        return pd.Series(1.0 / len(tickers), index=tickers)
    w = pd.Series({t: float(weights.get(t, 0.0)) for t in tickers})
    s = w.sum()
    if s <= 0:
        raise ValueError("WEIGHTS 總和必須 > 0")
    return w / s


def run_backtest(
    px: pd.DataFrame,
    hold_mask: pd.DataFrame,
    base_weights: pd.Series,
    initial_cash: float,
    tx_cost_bps: float,
) -> BacktestResult:
    cols = list(px.columns)
    rets = px.pct_change()

    raw = hold_mask.astype(float).mul(base_weights.reindex(cols), axis=1)
    row_sum = raw.sum(axis=1)
    target_w = raw.div(row_sum.replace(0, np.nan), axis=0).fillna(0.0)

    w_lag = target_w.shift(1).fillna(0.0)
    port_ret_gross = (w_lag * rets).sum(axis=1)

    w_prev = w_lag.shift(1).fillna(0.0)
    turnover = (w_lag - w_prev).abs().sum(axis=1)
    cost = turnover * (tx_cost_bps / 10_000.0)
    port_ret = port_ret_gross - cost

    equity = (1.0 + port_ret.fillna(0.0)).cumprod() * initial_cash
    if len(equity):
        equity.iloc[0] = initial_cash

    trade_rows = []
    delta = w_lag - w_prev
    for dt, row in delta.iterrows():
        for t, d in row.items():
            if abs(d) > 1e-6:
                trade_rows.append({"date": dt, "ticker": t, "weight_change": float(d)})
    trades = pd.DataFrame(trade_rows)

    return BacktestResult(
        equity=equity.rename("equity"),
        returns=port_ret.rename("strategy_ret"),
        weights=w_lag,
        trades=trades,
        metrics={},
    )


def compute_metrics(res: BacktestResult, rf_annual: float = 0.0, trading_days: int = TRADING_DAYS) -> dict:
    eq = res.equity.dropna()
    r = res.returns.dropna()
    if len(eq) < 2 or len(r) < 2:
        return {"error": "數據不足"}

    total_return = float(eq.iloc[-1] / eq.iloc[0] - 1.0)
    n_years = (eq.index[-1] - eq.index[0]).days / 365.25
    cagr = (
        float((eq.iloc[-1] / eq.iloc[0]) ** (1 / max(n_years, 1e-9)) - 1.0)
        if n_years > 0
        else float("nan")
    )
    dd = eq / eq.cummax() - 1.0
    max_dd = float(dd.min())
    excess = r - rf_annual / trading_days
    sharpe = (
        float(np.sqrt(trading_days) * excess.mean() / excess.std())
        if excess.std() > 0
        else float("nan")
    )
    return {
        "start": str(eq.index[0].date()),
        "end": str(eq.index[-1].date()),
        "total_return": total_return,
        "cagr": cagr,
        "max_drawdown": max_dd,
        "sharpe": sharpe,
        "final_equity": float(eq.iloc[-1]),
        "n_trade_events": int(len(res.trades)),
    }


def buy_and_hold_equity(price: pd.Series, initial_cash: float) -> pd.Series:
    p = price.dropna()
    if p.empty:
        return pd.Series(dtype=float)
    eq = initial_cash * (p / p.iloc[0])
    eq.name = "buy_hold"
    return eq


def run_single_ticker_strategy(
    ticker: str,
    universe_prices: pd.DataFrame,
    vix: pd.Series,
    *,
    ma_short: int = 50,
    ma_long: int = 200,
    use_ma_cross: bool = False,
    breadth_ma: int = 200,
    breadth_min_pct: float = 0.5,
    require_breadth: bool = True,
    require_vix: bool = True,
    vix_risk_off: float = 30.0,
    initial_cash: float = 100_000.0,
    tx_cost_bps: float = 5.0,
) -> BacktestResult:
    """Single-ticker column: 100% weight when signals allow; breadth from full loaded universe."""
    ticker = ticker.upper()
    if ticker not in universe_prices.columns:
        raise ValueError(f"{ticker} 無價格數據")

    px_uni = universe_prices
    ma_sig_uni = compute_ma_signals(px_uni, ma_short, ma_long, use_ma_cross)
    breadth = compute_breadth_proxy(px_uni, breadth_ma)
    hold_uni = combine_signals(
        ma_sig_uni, breadth, vix, require_breadth, breadth_min_pct, require_vix, vix_risk_off
    )

    px = px_uni[[ticker]].dropna(how="all")
    hold = hold_uni[[ticker]].reindex(px.index).fillna(False)
    bw = normalize_weights([ticker], None)
    res = run_backtest(px, hold, bw, initial_cash, tx_cost_bps)
    res.metrics = compute_metrics(res)
    res.breadth_proxy = breadth.reindex(px.index)
    res.vix = vix.reindex(px.index)
    res.prices = px
    res.hold_mask = hold
    return res


def load_market_data(
    tickers: list[str],
    start: str = "2006-01-01",
    end: Optional[str] = None,
) -> tuple[pd.DataFrame, pd.Series]:
    tickers = [t.strip().upper() for t in tickers if t and str(t).strip()]
    prices = download_prices(tickers, start, end)
    vix = download_series(VIX_TICKER, start, end, "VIX")
    common = prices.index.intersection(vix.index)
    return prices.loc[common], vix.loc[common]
