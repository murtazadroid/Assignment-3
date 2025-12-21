# frontend/app.py

from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

st.set_page_config(page_title="Kalshi RL Replay Dashboard", layout="wide")
st.title("Kalshi RL Replay Dashboard (Synthetic Replay)")

# -------------------------
# Load replay log
# -------------------------
LOG_PATH = st.text_input("Replay log CSV", value="demo_logs/replay_trades.csv")
p = Path(LOG_PATH)
if not p.exists():
    st.warning("Log not found. Run: python demo/run_replay.py")
    st.stop()

df = pd.read_csv(p)

# Episode list
episodes = sorted(df["episode"].unique().tolist())
st.sidebar.header("Replay Controls")
ep = st.sidebar.selectbox("Episode", episodes, index=0)

# Episode dataframe
dfe = df[df["episode"] == ep].sort_values("step").reset_index(drop=True)

# Derived columns
START_BAL = 10_000.0
dfe["cum_pnl"] = dfe["portfolio_value"] - START_BAL
dfe["pnl_step"] = dfe["delta_pv"]  # change in PV at each step

max_step = int(dfe["step"].max())
step_k = st.sidebar.slider("Replay step", min_value=0, max_value=max_step, value=min(0, max_step), step=1)

# Slice up to step_k for "replay"
dfe_k = dfe[dfe["step"] <= step_k].copy()

# Current row (the active step)
cur = dfe[dfe["step"] == step_k].iloc[0]

# -------------------------
# Helper: action labels
# -------------------------
def trade_side_label(x: int) -> str:
    # Your env uses trade_side in info. If it's 0/1/2 etc, map here.
    # If your env uses: 0=hold, 1=buy_yes, 2=buy_no (or sell), adjust accordingly.
    if x == 0:
        return "HOLD"
    if x == 1:
        return "TRADE (side=1)"
    if x == 2:
        return "TRADE (side=2)"
    return f"TRADE (side={x})"

# -------------------------
# Top KPIs (current step)
# -------------------------
c1, c2, c3, c4, c5 = st.columns(5)

c1.metric("Step", f"{int(cur['step'])} / {max_step}")
c2.metric("Hour", f"{int(cur['hour'])}" if "hour" in cur and not pd.isna(cur["hour"]) else "-")
c3.metric("Action", trade_side_label(int(cur["trade_side"])))
c4.metric("ΔPnL (this step)", f"{float(cur['pnl_step']):+.4f}")
c5.metric("Cum PnL (episode)", f"{float(cur['cum_pnl']):+.4f}")

# Color hint
if float(cur["pnl_step"]) >= 0:
    st.success("This step was PROFIT (green).")
else:
    st.error("This step was LOSS (red).")

# -------------------------
# Layout: side-by-side panels
# -------------------------
left, right = st.columns([1.3, 1.0])

# -------------------------
# LEFT: Price + Threshold + cursor + trade markers
# -------------------------
with left:
    st.subheader("Replay: Decision Price vs Threshold (up to current step)")

    # marker colors by step pnl (green profit / red loss)
    marker_colors = np.where(dfe_k["pnl_step"] >= 0, "green", "red")

    fig_price = go.Figure()

    fig_price.add_trace(go.Scatter(
        x=dfe_k["step"], y=dfe_k["decision_price"],
        mode="lines+markers",
        name="Decision Price",
        line=dict(width=2),
        marker=dict(size=7, color=marker_colors),
        hovertemplate="step=%{x}<br>decision_price=%{y:.2f}<extra></extra>"
    ))

    fig_price.add_trace(go.Scatter(
        x=dfe_k["step"], y=dfe_k["threshold"],
        mode="lines",
        name="Threshold",
        line=dict(dash="dash", width=2),
        hovertemplate="step=%{x}<br>threshold=%{y:.2f}<extra></extra>"
    ))

    # Trade markers (non-hold only)
    trades_k = dfe_k[dfe_k["trade_side"] != 0]
    if len(trades_k) > 0:
        fig_price.add_trace(go.Scatter(
            x=trades_k["step"],
            y=trades_k["decision_price"],
            mode="markers",
            name="Trades",
            marker=dict(size=12, symbol="x"),
            hovertemplate="step=%{x}<br>trade_side=%{customdata}<extra></extra>",
            customdata=trades_k["trade_side"].astype(int).values
        ))

    # Vertical cursor line at current step
    fig_price.add_vline(
        x=step_k,
        line_width=2,
        line_dash="dot",
        line_color="black"
    )

    fig_price.update_layout(
        height=430,
        xaxis_title="Step (hourly event)",
        yaxis_title="Price",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=10, r=10, t=35, b=10),
    )
    st.plotly_chart(fig_price, use_container_width=True)

# -------------------------
# RIGHT: Volatility + PnL curve (both replay up to current step)
# -------------------------
with right:
    st.subheader("Replay: Volatility + PnL (up to current step)")

    # Volatility curve
    fig_vol = go.Figure()
    fig_vol.add_trace(go.Scatter(
        x=dfe_k["step"], y=dfe_k["vol_30m"],
        mode="lines+markers",
        name="vol_30m",
        hovertemplate="step=%{x}<br>vol_30m=%{y:.6f}<extra></extra>"
    ))
    fig_vol.add_vline(x=step_k, line_width=2, line_dash="dot", line_color="black")
    fig_vol.update_layout(
        height=200,
        xaxis_title="Step",
        yaxis_title="vol_30m",
        margin=dict(l=10, r=10, t=25, b=10),
        showlegend=False
    )
    st.plotly_chart(fig_vol, use_container_width=True)

    # Cumulative PnL curve
    fig_pnl = go.Figure()
    pnl_colors = np.where(dfe_k["pnl_step"] >= 0, "green", "red")
    fig_pnl.add_trace(go.Scatter(
        x=dfe_k["step"], y=dfe_k["cum_pnl"],
        mode="lines+markers",
        name="Cum PnL",
        marker=dict(size=7, color=pnl_colors),
        hovertemplate="step=%{x}<br>cum_pnl=%{y:.4f}<extra></extra>"
    ))
    fig_pnl.add_vline(x=step_k, line_width=2, line_dash="dot", line_color="black")
    fig_pnl.update_layout(
        height=200,
        xaxis_title="Step",
        yaxis_title="Cumulative PnL",
        margin=dict(l=10, r=10, t=25, b=10),
        showlegend=False
    )
    st.plotly_chart(fig_pnl, use_container_width=True)

# -------------------------
# Step details panel (exactly what happened at this step)
# -------------------------
st.subheader("Current Step Details")
detail_cols = st.columns(4)

detail_cols[0].write({
    "decision_price": float(cur.get("decision_price", np.nan)),
    "threshold": float(cur.get("threshold", np.nan)),
    "resolve_price": float(cur.get("resolve_price", np.nan)),
})
detail_cols[1].write({
    "payoff_yes": float(cur.get("payoff_yes", np.nan)),
    "payoff_no": float(cur.get("payoff_no", np.nan)),
    "trade_cost": float(cur.get("trade_cost", np.nan)),
})
detail_cols[2].write({
    "yes_midprice": float(cur.get("yes_midprice", np.nan)),
    "no_midprice": float(cur.get("no_midprice", np.nan)),
    "moneyness": float(cur.get("moneyness", np.nan)),
})
detail_cols[3].write({
    "reward": float(cur.get("reward", np.nan)),
    "delta_pv": float(cur.get("delta_pv", np.nan)),
    "portfolio_value": float(cur.get("portfolio_value", np.nan)),
})

# -------------------------
# Table (up to current step)
# -------------------------
st.subheader("Step-by-step Table (up to current step)")
show_trades_only = st.checkbox("Show only trade steps", value=False)

table_df = dfe_k.copy()
if show_trades_only:
    table_df = table_df[table_df["trade_side"] != 0]

cols_to_show = [
    "episode","step","hour",
    "decision_price","threshold","resolve_price",
    "payoff_yes","payoff_no",
    "yes_midprice","no_midprice",
    "trade_side","trade_cost",
    "reward","delta_pv","portfolio_value",
    "vol_30m","moneyness"
]
cols_to_show = [c for c in cols_to_show if c in table_df.columns]

st.dataframe(table_df[cols_to_show], use_container_width=True)
