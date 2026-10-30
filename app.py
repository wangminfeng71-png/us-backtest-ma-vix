# -*- coding: utf-8 -*-
"""美股回測前臺：均線 × 市寬 PROXY × VIX（研究工具，非投資建議）"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine import (  # noqa: E402
    buy_and_hold_equity,
    load_market_data,
    run_single_ticker_strategy,
)

st.set_page_config(
    page_title="美股回測｜均線×市寬×VIX",
    page_icon="📈",
    layout="wide",
)

st.title("美股回測工具（均線 × 市寬 PROXY × VIX）")
st.caption(
    "研究／工程用途，**唔係投資建議**。歷史回測唔代表未來表現。"
    "市寬為自選組合內部 PROXY（% 收市高過自身 MA），**唔係**交易所官方市寬。"
)

with st.sidebar:
    st.header("參數設定")
    start = st.date_input("開始日期", value=pd.Timestamp("2006-01-01").date())
    end_opt = st.checkbox("指定結束日期", value=False)
    end = st.date_input("結束日期", value=pd.Timestamp.today().date()) if end_opt else None

    st.subheader("均線")
    ma_short = st.number_input("短均線", min_value=5, max_value=250, value=50, step=1)
    ma_long = st.number_input("長均線 (MA)", min_value=10, max_value=400, value=200, step=1)
    use_ma_cross = st.checkbox("用短均 > 長均（否則：價 > 長均）", value=False)

    st.subheader("市寬 PROXY")
    breadth_ma = st.number_input("市寬用 MA 窗口", min_value=10, max_value=400, value=200, step=1)
    breadth_min = st.slider("市寬門檻（% 以上）", 0.0, 1.0, 0.5, 0.05)
    require_breadth = st.checkbox("啟用市寬過濾", value=True)

    st.subheader("VIX（恐慌指數）")
    vix_off = st.number_input("風險關閉門檻（VIX ≥）", min_value=10.0, max_value=80.0, value=30.0, step=1.0)
    require_vix = st.checkbox("啟用 VIX 過濾（VIX < 門檻先允許持倉）", value=True)

    st.subheader("資金／成本")
    initial_cash = st.number_input("初始資金 (USD)", min_value=1000.0, value=100_000.0, step=1000.0)
    tx_bps = st.number_input("單邊交易成本 (bps)", min_value=0.0, max_value=50.0, value=5.0, step=0.5)
    show_bh = st.checkbox("疊加買入持有曲線", value=True)

st.subheader("對比標的（最多 3 隻）")
c1, c2, c3 = st.columns(3)
with c1:
    t1 = st.text_input("股票 1", value="QQQ", placeholder="例如 QQQ").strip().upper()
with c2:
    t2 = st.text_input("股票 2", value="VOO", placeholder="例如 VOO").strip().upper()
with c3:
    t3 = st.text_input("股票 3", value="TSLA", placeholder="例如 TSLA").strip().upper()

run = st.button("開始回測", type="primary", use_container_width=True)


def _fmt_pct(x) -> str:
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return "—"
    return f"{x:.2%}"


def _fmt_num(x, nd=2) -> str:
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return "—"
    return f"{x:.{nd}f}"


@st.cache_data(show_spinner="下載行情中（yfinance）…", ttl=3600)
def _cached_load(tickers: tuple[str, ...], start_s: str, end_s: str | None):
    return load_market_data(list(tickers), start_s, end_s)


if run:
    tickers = [t for t in (t1, t2, t3) if t]
    # de-dupe preserve order
    seen = set()
    uniq = []
    for t in tickers:
        if t not in seen:
            seen.add(t)
            uniq.append(t)
    tickers = uniq

    if not tickers:
        st.error("請至少輸入一個股票代號。")
        st.stop()
    if len(tickers) > 3:
        st.error("最多比較 3 隻股票。")
        st.stop()
    if ma_long < ma_short and use_ma_cross:
        st.warning("短均線 ≥ 長均線時交叉訊號可能無意義，請檢查參數。")

    start_s = str(start)
    end_s = str(end) if end else None

    try:
        prices, vix = _cached_load(tuple(tickers), start_s, end_s)
    except Exception as e:
        st.error(f"數據下載失敗：{e}")
        st.stop()

    missing = [t for t in tickers if t not in prices.columns]
    if missing:
        st.warning(f"以下代號無數據（可能上市較遲或下載失敗）：{', '.join(missing)}")
    ok_tickers = [t for t in tickers if t in prices.columns]
    if not ok_tickers:
        st.error("沒有可用價格數據。")
        st.stop()

    results = {}
    errors = {}
    with st.spinner("計算訊號同回測…"):
        for t in ok_tickers:
            try:
                results[t] = run_single_ticker_strategy(
                    t,
                    prices,
                    vix,
                    ma_short=int(ma_short),
                    ma_long=int(ma_long),
                    use_ma_cross=use_ma_cross,
                    breadth_ma=int(breadth_ma),
                    breadth_min_pct=float(breadth_min),
                    require_breadth=require_breadth,
                    require_vix=require_vix,
                    vix_risk_off=float(vix_off),
                    initial_cash=float(initial_cash),
                    tx_cost_bps=float(tx_bps),
                )
            except Exception as e:
                errors[t] = str(e)

    if errors:
        for t, msg in errors.items():
            st.error(f"{t}: {msg}")
    if not results:
        st.error("全部標的回測失敗。")
        st.stop()

    st.success(
        f"完成｜區間約 {prices.index.min().date()} → {prices.index.max().date()}｜"
        f"市寬 PROXY 宇宙 = {', '.join(ok_tickers)}"
    )

    # ---- metrics side by side ----
    st.subheader("關鍵指標對比")
    metric_cols = st.columns(len(results))
    rows = []
    for i, (t, res) in enumerate(results.items()):
        m = res.metrics
        with metric_cols[i]:
            st.markdown(f"### {t}")
            if "error" in m:
                st.write(m["error"])
                continue
            st.metric("總回報", _fmt_pct(m.get("total_return")))
            st.metric("CAGR", _fmt_pct(m.get("cagr")))
            st.metric("最大回撤", _fmt_pct(m.get("max_drawdown")))
            st.metric("Sharpe", _fmt_num(m.get("sharpe")))
            st.caption(f"{m.get('start')} → {m.get('end')}")
        rows.append(
            {
                "Ticker": t,
                "總回報": m.get("total_return"),
                "CAGR": m.get("cagr"),
                "最大回撤": m.get("max_drawdown"),
                "Sharpe": m.get("sharpe"),
                "期末權益": m.get("final_equity"),
                "開始": m.get("start"),
                "結束": m.get("end"),
            }
        )

    df_m = pd.DataFrame(rows)
    st.dataframe(
        df_m.style.format(
            {
                "總回報": "{:.2%}",
                "CAGR": "{:.2%}",
                "最大回撤": "{:.2%}",
                "Sharpe": "{:.2f}",
                "期末權益": "{:,.0f}",
            },
            na_rep="—",
        ),
        use_container_width=True,
    )

    # ---- overlay equity ----
    st.subheader("權益曲線（策略）")
    fig = go.Figure()
    for t, res in results.items():
        fig.add_trace(
            go.Scatter(x=res.equity.index, y=res.equity.values, name=f"{t} 策略", mode="lines")
        )
        if show_bh and res.prices is not None:
            bh = buy_and_hold_equity(res.prices[t], float(initial_cash))
            fig.add_trace(
                go.Scatter(
                    x=bh.index,
                    y=bh.values,
                    name=f"{t} 買入持有",
                    mode="lines",
                    line=dict(dash="dot"),
                )
            )
    fig.update_layout(
        height=420,
        margin=dict(l=40, r=20, t=30, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        yaxis_title="權益 (USD)",
        xaxis_title="日期",
        hovermode="x unified",
    )
    st.plotly_chart(fig, use_container_width=True)

    # ---- breadth + VIX ----
    st.subheader("市寬 PROXY 與 VIX（共用）")
    sample = next(iter(results.values()))
    fig2 = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08,
                         subplot_titles=("市寬 PROXY（組合內 % 高過自身 MA）", "VIX"))
    if sample.breadth_proxy is not None:
        fig2.add_trace(
            go.Scatter(
                x=sample.breadth_proxy.index,
                y=sample.breadth_proxy.values,
                name="Breadth PROXY",
                line=dict(color="#2ca02c"),
            ),
            row=1,
            col=1,
        )
        fig2.add_hline(y=float(breadth_min), line_dash="dash", line_color="gray", row=1, col=1)
    if sample.vix is not None:
        fig2.add_trace(
            go.Scatter(x=sample.vix.index, y=sample.vix.values, name="VIX", line=dict(color="#ff7f0e")),
            row=2,
            col=1,
        )
        fig2.add_hline(y=float(vix_off), line_dash="dash", line_color="red", row=2, col=1)
    fig2.update_layout(height=460, margin=dict(l=40, r=20, t=40, b=40), showlegend=True)
    fig2.update_yaxes(title_text="% above MA", range=[0, 1.05], row=1, col=1)
    fig2.update_yaxes(title_text="VIX", row=2, col=1)
    st.plotly_chart(fig2, use_container_width=True)

    with st.expander("規則說明"):
        st.markdown(
            f"""
- **均線**：{'短均 > 長均' if use_ma_cross else f'收市價 > MA{int(ma_long)}'}（每隻獨立）
- **市寬 PROXY**：載入嘅對比標的中，有幾多 % 收市高過自身 MA{int(breadth_ma)}；門檻 ≥ {breadth_min:.0%}；{'已啟用' if require_breadth else '已關閉'}
- **VIX**：VIX < {vix_off:g} 先允許風險倉；{'已啟用' if require_vix else '已關閉'}
- **執行假設**：訊號日收市決定目標權重，用下一期報酬近似；單邊成本 {tx_bps} bps
- **每欄策略**：單一標的等權（允許持倉時 100% 該股，否則現金）
            """
        )
else:
    st.info(
        "左側可調參數；上方輸入最多 3 個代號（預設 QQQ / VOO / TSLA），撳「開始回測」。"
        "首次下載可能需半分鐘。"
    )
