
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go

st.set_page_config(page_title="Portfolio Lab", page_icon="📈", layout="wide")

st.title("📈 Portfolio Lab")
st.caption("Build a portfolio, test it against historical markets, and explore future what-if scenarios.")

# ---------- helpers ----------
@st.cache_data(ttl=3600, show_spinner=False)
def get_history(tickers, start, end):
    data = yf.download(
        tickers=list(tickers),
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
        group_by="column",
        threads=True,
    )
    if data.empty:
        return pd.DataFrame()
    if isinstance(data.columns, pd.MultiIndex):
        # Close is the first level after download
        if "Close" in data.columns.get_level_values(0):
            data = data["Close"]
        else:
            data = data.xs("Close", axis=1, level=0)
    elif "Close" in data.columns:
        data = data["Close"]
    return data.dropna(how="all")


def annual_returns(prices):
    return prices.resample("YE").last().pct_change().dropna()


def portfolio_value_from_original_weights(prices, weights, initial):
    """
    Buy-and-hold: original weights are set on the starting date and never
    rebalanced. Missing prices are forward-filled after the common start.
    """
    prices = prices.dropna(how="all").ffill().dropna(how="all")
    common = [t for t in weights if t in prices.columns]
    if not common:
        return pd.Series(dtype=float)
    prices = prices[common].dropna()
    if prices.empty:
        return pd.Series(dtype=float)
    w = pd.Series({t: weights[t] for t in common}, dtype=float)
    w = w / w.sum()
    shares = (initial * w) / prices.iloc[0]
    return (prices * shares).sum(axis=1)


def portfolio_stats(series):
    if len(series) < 2:
        return {}
    years = (series.index[-1] - series.index[0]).days / 365.25
    cagr = (series.iloc[-1] / series.iloc[0]) ** (1 / years) - 1 if years > 0 else np.nan
    peak = series.cummax()
    drawdown = series / peak - 1
    return {
        "final": series.iloc[-1],
        "total_return": series.iloc[-1] / series.iloc[0] - 1,
        "cagr": cagr,
        "max_drawdown": drawdown.min(),
    }


def make_growth_chart(series_map):
    fig = go.Figure()
    for name, s in series_map.items():
        fig.add_trace(go.Scatter(x=s.index, y=s.values, mode="lines", name=name))
    fig.update_layout(
        height=500,
        xaxis_title="Date",
        yaxis_title="Portfolio value (£)",
        hovermode="x unified",
        legend_title="Portfolio",
    )
    return fig


# ---------- sidebar ----------
st.sidebar.header("Portfolio setup")
initial = st.sidebar.number_input("Initial investment (£)", min_value=1000.0, value=100000.0, step=5000.0)
monthly = st.sidebar.number_input("Monthly contribution (£)", min_value=0.0, value=1000.0, step=100.0)

start_year = st.sidebar.slider("Start year", 1990, 2025, 2010)
end_year = 2026
start_date = f"{start_year}-01-01"

st.sidebar.markdown("### What-if comparison")
comparison_amount = st.sidebar.number_input("Comparison investment (£)", min_value=1000.0, value=100000.0, step=5000.0)

tabs = st.tabs(["Portfolio Builder", "Historical Results", "What If?", "Future Scenarios", "About"])

# ---------- asset universe ----------
default_assets = {
    "Stocks / Equities": "SPY",
    "Bonds": "AGG",
    "Funds / ETFs": "VTI",
    "Cash": "BIL",
    "Gold / Commodities": "GLD",
    "Property / REITs": "VNQ",
    "Crypto": "BTC-USD",
}
st.session_state.setdefault("asset_map", default_assets.copy())

with tabs[0]:
    st.subheader("1. Build your portfolio")
    st.write("Use the sliders to set the initial allocation. The app treats the allocation as a buy-and-hold portfolio: **no automatic rebalancing**.")

    selected = {}
    cols = st.columns(2)
    for i, (asset, ticker) in enumerate(st.session_state.asset_map.items()):
        with cols[i % 2]:
            pct = st.slider(asset, 0, 100, 0 if asset != "Stocks / Equities" else 50, 1, key=f"asset_{asset}")
            if pct > 0:
                selected[asset] = (ticker, pct / 100)

    total = sum(v[1] for v in selected.values())
    st.metric("Allocation total", f"{total:.0%}", delta=f"{(total-1):.0%} vs 100%")

    st.markdown("---")
    st.subheader("2. Add individual securities")
    st.caption("Examples: AAPL, MSFT, NVDA, TSLA, VUSA.L, CSPX.L. Yahoo Finance tickers are used.")

    custom_text = st.text_input("Ticker symbols (comma separated)", placeholder="AAPL, MSFT, VUSA.L")
    if custom_text:
        custom = [x.strip().upper() for x in custom_text.split(",") if x.strip()]
        st.session_state["custom_tickers"] = custom

    custom_tickers = st.session_state.get("custom_tickers", [])
    custom_weights = {}
    if custom_tickers:
        ccols = st.columns(3)
        for i, ticker in enumerate(custom_tickers):
            with ccols[i % 3]:
                custom_weights[ticker] = st.slider(ticker, 0, 100, 0, 1, key=f"custom_{ticker}")

    if custom_weights:
        for ticker, pct in custom_weights.items():
            if pct:
                selected[f"{ticker} (individual)"] = (ticker, pct / 100)

    total = sum(v[1] for v in selected.values())
    if total > 0:
        st.write("### Current allocation")
        alloc = pd.DataFrame(
            [{"Investment": k, "Ticker": v[0], "Weight": v[1]} for k, v in selected.items()]
        )
        alloc["Weight"] = alloc["Weight"] / total
        st.dataframe(
            alloc.assign(Weight=lambda x: (x["Weight"] * 100).round(1).astype(str) + "%"),
            use_container_width=True,
            hide_index=True,
        )
        if abs(total - 1) > 0.001:
            st.warning("The sliders do not currently total 100%. Historical calculations will normalise the selected weights to 100%.")
    else:
        st.info("Move at least one slider above 0% to create a portfolio.")

    st.session_state["selected"] = selected

with tabs[1]:
    st.subheader("Historical growth")
    selected = st.session_state.get("selected", {})
    if not selected:
        st.info("Build a portfolio in the Portfolio Builder first.")
    else:
        tickers = sorted(set(v[0] for v in selected.values()))
        with st.spinner("Downloading historical market data..."):
            prices = get_history(tickers, start_date, "2026-09-29")

        if prices.empty:
            st.error("No historical data was returned. Check the ticker symbols and your internet connection.")
        else:
            weights = {label: weight for label, (_, weight) in selected.items()}
            ticker_weights = {}
            for label, (ticker, weight) in selected.items():
                ticker_weights[ticker] = ticker_weights.get(ticker, 0) + weight

            p = prices[[t for t in ticker_weights if t in prices.columns]].dropna()
            if p.empty:
                st.error("There is not enough overlapping price history for the selected securities.")
            else:
                # Normalise initial weights among securities with available history.
                total_available = sum(ticker_weights[t] for t in p.columns)
                norm = {t: w / total_available for t, w in ticker_weights.items() if t in p.columns}
                portfolio = portfolio_value_from_original_weights(p, norm, initial)

                # Contribution model: approximate monthly additions by buying the same
                # original allocation at each month-end. This is separate from the
                # pure buy-and-hold result.
                monthly_prices = p.resample("ME").last().dropna(how="all").ffill()
                units = pd.Series(0.0, index=p.columns)
                values = []
                contribution = initial
                first_prices = p.iloc[0]
                units += (initial * pd.Series(norm)) / first_prices
                for dt, row in monthly_prices.iterrows():
                    if dt > p.index[0]:
                        units += (monthly * pd.Series(norm)) / row
                    values.append((dt, (units * row).sum()))
                contribution_series = pd.Series(values, index=[x[0] for x in values])
                contribution_series = pd.concat([
                    pd.Series({p.index[0]: initial}),
                    contribution_series
                ]).sort_index()
                contribution_series = contribution_series[~contribution_series.index.duplicated(keep="first")]

                stats = portfolio_stats(portfolio)
                cstats = portfolio_stats(contribution_series)

                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Starting value", f"£{initial:,.0f}")
                c2.metric("Final value", f"£{stats['final']:,.0f}")
                c3.metric("Annualised return", f"{stats['cagr']:.2%}")
                c4.metric("Maximum drawdown", f"{stats['max_drawdown']:.2%}")

                st.plotly_chart(
                    make_growth_chart({
                        "Buy & hold": portfolio,
                        f"Buy & hold + £{monthly:,.0f}/month": contribution_series
                    }),
                    use_container_width=True,
                )

                st.info(
                    "Historical prices are sourced from Yahoo Finance through yfinance. "
                    "Returns are historical and do not imply future performance. "
                    "Some securities have shorter histories than 1990, so the selected start date "
                    "is effectively limited by the earliest available data."
                )

                st.subheader("Benchmark comparison")
                benchmark_tickers = {"100% Stocks (SPY)": "SPY", "100% Bonds (AGG)": "AGG", "60/40 Stocks/Bonds": None}
                bprices = get_history(["SPY", "AGG"], start_date, "2026-09-29").dropna()
                benchmarks = {}
                if "SPY" in bprices:
                    benchmarks["100% Stocks (SPY)"] = portfolio_value_from_original_weights(
                        bprices[["SPY"]], {"SPY": 1}, comparison_amount
                    )
                if "AGG" in bprices:
                    benchmarks["100% Bonds (AGG)"] = portfolio_value_from_original_weights(
                        bprices[["AGG"]], {"AGG": 1}, comparison_amount
                    )
                if {"SPY", "AGG"}.issubset(bprices.columns):
                    benchmarks["60/40 Stocks/Bonds"] = portfolio_value_from_original_weights(
                        bprices[["SPY", "AGG"]], {"SPY": .6, "AGG": .4}, comparison_amount
                    )
                st.plotly_chart(make_growth_chart(benchmarks), use_container_width=True)

                st.subheader("Year-by-year return")
                ar = annual_returns(p)
                st.dataframe((ar * 100).round(2), use_container_width=True)

with tabs[2]:
    st.subheader("What If?")
    st.write("Compare alternative allocations using the same starting amount and historical period.")

    a, b = st.columns(2)
    with a:
        stock_pct = st.slider("Portfolio A — Stocks", 0, 100, 60, 1, key="wia_stock")
        bond_pct = 100 - stock_pct
        st.metric("Portfolio A — Bonds", f"{bond_pct}%")
    with b:
        stock_pct_b = st.slider("Portfolio B — Stocks", 0, 100, 80, 1, key="wib_stock")
        bond_pct_b = 100 - stock_pct_b
        st.metric("Portfolio B — Bonds", f"{bond_pct_b}%")

    bprices = get_history(["SPY", "AGG"], start_date, "2026-09-29").dropna()
    if not bprices.empty:
        A = portfolio_value_from_original_weights(
            bprices, {"SPY": stock_pct/100, "AGG": bond_pct/100}, comparison_amount
        )
        B = portfolio_value_from_original_weights(
            bprices, {"SPY": stock_pct_b/100, "AGG": bond_pct_b/100}, comparison_amount
        )
        st.plotly_chart(make_growth_chart({"Portfolio A": A, "Portfolio B": B}), use_container_width=True)
        sa, sb = portfolio_stats(A), portfolio_stats(B)
        table = pd.DataFrame({
            "Metric": ["Final value", "Total return", "Annualised return", "Maximum drawdown"],
            "Portfolio A": [sa["final"], sa["total_return"], sa["cagr"], sa["max_drawdown"]],
            "Portfolio B": [sb["final"], sb["total_return"], sb["cagr"], sb["max_drawdown"]],
        })
        table.loc[0, ["Portfolio A", "Portfolio B"]] = table.loc[0, ["Portfolio A", "Portfolio B"]].map(lambda x: f"£{x:,.0f}")
        table.loc[1:, ["Portfolio A", "Portfolio B"]] = table.loc[1:, ["Portfolio A", "Portfolio B"]].map(lambda x: f"{x:.2%}")
        st.dataframe(table, use_container_width=True, hide_index=True)

with tabs[3]:
    st.subheader("🔮 Future Scenario Lab")
    st.write(
        "Explore hypothetical 5-year outcomes under different inflation and interest-rate environments. "
        "These are scenario assumptions, not forecasts."
    )

    col1, col2 = st.columns(2)
    with col1:
        scenario = st.selectbox(
            "Economic scenario",
            [
                "Baseline",
                "High inflation + high rates",
                "Falling inflation + falling rates",
                "Stagflation",
                "Low inflation + low rates",
                "Custom"
            ]
        )
        inflation = st.slider("Average inflation", 0.0, 15.0, 2.5, 0.1) / 100
        rate = st.slider("Average interest rate", 0.0, 15.0, 3.5, 0.1) / 100
    with col2:
        years = st.slider("Projection period (years)", 1, 10, 5)
        future_initial = st.number_input("Future starting portfolio (£)", min_value=1000.0, value=100000.0, step=5000.0)
        future_monthly = st.number_input("Future monthly contribution (£)", min_value=0.0, value=1000.0, step=100.0)

    scenario_defaults = {
        "Baseline": (0.025, 0.035),
        "High inflation + high rates": (0.060, 0.065),
        "Falling inflation + falling rates": (0.020, 0.025),
        "Stagflation": (0.070, 0.060),
        "Low inflation + low rates": (0.015, 0.020),
    }
    if scenario != "Custom":
        inflation, rate = scenario_defaults[scenario]

    st.write(f"**Assumptions:** inflation {inflation:.1%} | interest rate {rate:.1%}")

    # Illustrative asset-return model. Clearly labelled as assumptions.
    # Higher rates/inflation affect asset classes differently.
    assumptions = pd.DataFrame({
        "Asset class": ["Stocks", "Bonds", "Funds / ETFs", "Cash", "Gold / commodities", "Property / REITs", "Crypto"],
        "Expected nominal return": [
            0.075 + 0.15*(0.025-inflation),
            rate + 0.015 - 0.35*(inflation-0.025),
            0.065 + 0.12*(0.025-inflation),
            rate,
            0.045 + 0.40*(inflation-0.025),
            0.055 + 0.10*(0.025-inflation) - 0.10*(rate-0.035),
            0.10
        ]
    })
    assumptions["Real return after inflation"] = (1 + assumptions["Expected nominal return"]) / (1 + inflation) - 1
    st.dataframe(
        assumptions.assign(
            **{
                "Expected nominal return": assumptions["Expected nominal return"].map(lambda x: f"{x:.2%}"),
                "Real return after inflation": assumptions["Real return after inflation"].map(lambda x: f"{x:.2%}")
            }
        ),
        use_container_width=True,
        hide_index=True,
    )

    # Use current builder allocation if available; otherwise 60/40.
    selected = st.session_state.get("selected", {})
    if selected:
        current = {}
        for label, (ticker, weight) in selected.items():
            # map to broad class where possible
            key = label.split(" (individual)")[0]
            current[key] = current.get(key, 0) + weight
        total = sum(current.values())
        current = {k: v/total for k,v in current.items()}
    else:
        current = {"Stocks / Equities": .6, "Bonds": .4}

    class_map = {
        "Stocks / Equities": "Stocks",
        "Bonds": "Bonds",
        "Funds / ETFs": "Funds / ETFs",
        "Cash": "Cash",
        "Gold / Commodities": "Gold / commodities",
        "Property / REITs": "Property / REITs",
        "Crypto": "Crypto",
    }
    returns = dict(zip(assumptions["Asset class"], assumptions["Expected nominal return"]))
    nominal = sum(
        w * returns[class_map[k]]
        for k, w in current.items()
        if k in class_map
    )

    # Monthly future simulation using constant scenario return.
    months = years * 12
    monthly_r = (1 + nominal) ** (1/12) - 1
    vals = [future_initial]
    for _ in range(months):
        vals.append(vals[-1] * (1 + monthly_r) + future_monthly)
    future = pd.Series(vals, index=pd.date_range(pd.Timestamp.today().normalize(), periods=months+1, freq="ME"))

    real_vals = future / ((1 + inflation) ** np.arange(len(future) / 12, len(future) / 12 + 1e-9, 1/12)[:len(future)])

    f1, f2, f3 = st.columns(3)
    f1.metric("Illustrative nominal return", f"{nominal:.2%}")
    f2.metric("Projected nominal value", f"£{future.iloc[-1]:,.0f}")
    f3.metric("Approx. value in today's money", f"£{real_vals.iloc[-1]:,.0f}")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=future.index, y=future.values, mode="lines", name="Nominal £"))
    fig.add_trace(go.Scatter(x=future.index, y=real_vals.values, mode="lines", name="Today's £"))
    fig.update_layout(height=450, xaxis_title="Date", yaxis_title="Value (£)", hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)

    st.warning(
        "The future model is deliberately a scenario tool. Its asset-class return assumptions are illustrative "
        "and should not be interpreted as financial advice or a prediction of actual market returns."
    )

with tabs[4]:
    st.subheader("About Portfolio Lab")
    st.markdown("""
    **Purpose:** learn how asset allocation, historical market performance, contributions,
    inflation and interest-rate scenarios can affect a portfolio.

    **Historical mode:** uses market-price history supplied by Yahoo Finance via `yfinance`.
    The portfolio starts with the selected allocation and is then held without rebalancing.

    **Future mode:** lets you change inflation and interest-rate assumptions and see how a
    simple illustrative return model changes the outcome.

    **Important:** this is an educational modelling tool, not investment advice. Historical
    performance does not guarantee future performance, and scenario assumptions are not forecasts.
    """)
