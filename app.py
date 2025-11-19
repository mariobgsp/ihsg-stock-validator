import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import data_loader 
import core_analysis as core
import reporting as rep
import backtest_engine as bt 

# --- HELPER ---
def analyze_change(df):
    if len(df) < 2: return 0
    return (df.iloc[-1]['close'] - df.iloc[-2]['close']) / df.iloc[-2]['close'] * 100

# --- PAGE CONFIG ---
st.set_page_config(page_title="Stock Trade Planner", layout="wide", page_icon="📈")
st.title("📈 Ultimate Stock Trade Planner")

# --- SIDEBAR ---
with st.sidebar:
    st.header("1. Data Configuration")
    data_source = st.radio("Source", ["Yahoo Finance", "CSV Upload"])
    df = None 
    
    if data_source == "Yahoo Finance":
        ticker = st.text_input("Ticker Symbol", value="BBCA.JK").upper()
        fetch_period = st.selectbox("Data History (Back-end)", ["1y", "2y", "5y", "max"], index=1, help="How much historical data to download for indicator calculation.")
        
        if st.button("Fetch Data"):
            with st.spinner(f"Fetching {ticker}..."):
                df = data_loader.get_stock_data_from_yfinance(ticker, period=fetch_period)
                if df is None:
                    st.error("Failed to fetch data. Check ticker symbol.")
                else:
                    st.session_state['data'] = df
    else:
        uploaded_file = st.file_uploader("Upload Stock CSV", type=['csv'], key="stock_uploader_main")
        if uploaded_file is not None:
            df = data_loader.get_stock_data_from_csv(uploaded_file)
            if df is not None:
                 st.session_state['data'] = df

    if 'data' in st.session_state:
        df = st.session_state['data']

    st.divider()
    
    st.header("2. Analysis View")
    analysis_days = st.number_input("Zoom Analysis Window (Days)", min_value=30, max_value=1000, value=120, step=10, help="How many recent days to show on the chart.")
    
    st.markdown("**Visualization Options**")
    # RENAMED: Boxes to Lines
    show_sr_lines = st.checkbox("Show S/R Lines", value=True)
    show_fibs = st.checkbox("Show Fibonacci", value=True)
    show_supertrend = st.checkbox("Show Supertrend", value=True)
    show_bb = st.checkbox("Show Bollinger Bands", value=False)

# --- MAIN LOGIC ---
if df is not None:
    try:
        # 1. PREPARE DATA
        df_hist = df.dropna(subset=['close'])
        if analysis_days > len(df_hist): df_view = df_hist
        else: df_view = df_hist.iloc[-analysis_days:]
        last_bar = df_hist.iloc[-1]
        
        # 2. RUN ANALYSIS
        s_cluster, r_cluster, _, _ = core.find_support_resistance(df_hist)
        structure = core.detect_market_structure(df_hist)
        divergence = core.detect_divergence(df_hist)
        fibs = core.calculate_fibonacci_levels(df_view) 
        trade_plan = rep.recommend_trade(df_hist, s_cluster, r_cluster, structure, divergence)
        candle_shape = bt.classify_candle_shape(last_bar, abs(df_hist['close'] - df_hist['open']).mean())

        # --- TABS LAYOUT ---
        tab1, tab2, tab3 = st.tabs(["📊 Full Analysis", "⚙️ Backtest Engine", "🧪 Optimization Lab"])

        # ==========================================
        # TAB 1: CHARTING & ANALYSIS
        # ==========================================
        with tab1:
            # Metrics
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Price", f"{last_bar['close']:.0f}", f"{analyze_change(df_hist):.2f}%")
            m2.metric("Candle Shape", candle_shape) 
            st_val = last_bar.get('supertrend', 0)
            m3.metric("Supertrend", "BULLISH" if last_bar['close'] > st_val else "BEARISH", f"{st_val:.0f}")
            rec_label = trade_plan['Action']
            m4.metric("Trade Call", rec_label, trade_plan['Type'], delta_color="inverse" if "BUY" in rec_label else "normal")
            
            st.divider()
            k1, k2, k3, k4 = st.columns(4)
            k1.metric("ATR (Volatility)", f"{last_bar.get('atrr_14',0):.0f}")
            k2.metric("ADX (Trend)", f"{last_bar.get('adx_14',0):.1f}")
            k3.metric("RVOL", f"{last_bar.get('rvol',0):.2f}x")
            k4.metric("Nearest Support", trade_plan['Entry Zone'].split('-')[0].strip())

            if trade_plan['Action'] != "WAIT":
                st.info(f"### 🎯 Active Setup: {trade_plan['Type']}")
                c1, c2, c3, c4 = st.columns(4)
                c1.markdown(f"**Entry Zone**\n\n`{trade_plan['Entry Zone']}`")
                c2.markdown(f"**Stop Loss**\n\n`{trade_plan['SL']}`")
                c3.markdown(f"**Take Profit**\n\n`{trade_plan['TP']}`")
                c4.markdown(f"**Risk/Reward**\n\n`1 : {trade_plan['RR']}`")
                with st.expander("Analysis Reasons"):
                    for r in trade_plan['Reason']: st.write(f"- {r}")

            # --- FULL CHART ---
            st.subheader("Chart Analysis")
            row_heights = [0.55, 0.1, 0.15, 0.2] 
            fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=row_heights,
                                subplot_titles=("Price Action", "Volume", "MACD", "RSI (14)"))

            # ROW 1
            fig.add_trace(go.Candlestick(x=df_view.index, open=df_view['open'], high=df_view['high'],
                low=df_view['low'], close=df_view['close'], name='Price'), row=1, col=1)
            
            fig.add_trace(go.Scatter(x=df_view.index, y=df_view['sma_50'], line=dict(color='orange', width=1), name='MA 50'), row=1, col=1)
            fig.add_trace(go.Scatter(x=df_view.index, y=df_view['sma_200'], line=dict(color='blue', width=2), name='MA 200'), row=1, col=1)
            
            if show_supertrend and 'supertrend' in df_view.columns:
                st_color = "rgba(0, 200, 255, 0.8)"
                fig.add_trace(go.Scatter(x=df_view.index, y=df_view['supertrend'], 
                                         line=dict(color=st_color, width=2), name='Supertrend'), row=1, col=1)

            if show_bb and 'bb_upper' in df_view.columns:
                fig.add_trace(go.Scatter(x=df_view.index, y=df_view['bb_upper'], 
                                         line=dict(width=1, color='gray'), name='Upper BB'), row=1, col=1)
                fig.add_trace(go.Scatter(x=df_view.index, y=df_view['bb_lower'], 
                                         line=dict(width=1, color='gray'), fill='tonexty', fillcolor='rgba(128,128,128,0.1)', name='Lower BB'), row=1, col=1)

            # --- SMART SCALING & LINE VISUALIZATION ---
            view_min = df_view['low'].min()
            view_max = df_view['high'].max()
            limit_lower = view_min * 0.85
            limit_upper = view_max * 1.15

            if show_sr_lines and not s_cluster.empty:
                # Filter & Sort
                s_plot = s_cluster[(s_cluster['Level'] >= limit_lower) & (s_cluster['Level'] <= limit_upper)]
                s_plot = s_plot.sort_values(by='Hits', ascending=False).head(5)
                
                for _, row in s_plot.iterrows():
                    # LINE Visualization (Green Dashed)
                    fig.add_shape(type="line", 
                        x0=df_view.index[0], x1=df_view.index[-1], 
                        y0=row['Level'], y1=row['Level'],
                        line=dict(color="green", width=2, dash="dash"), 
                        layer="below", row=1, col=1
                    )
                    # Label with Range
                    label_text = f"Sup {row['Level']:.0f} [{row['Min']:.0f}-{row['Max']:.0f}]"
                    fig.add_annotation(x=df_view.index[-1], y=row['Level'], text=label_text, 
                                       showarrow=False, xanchor="left", font=dict(color="green", size=11), row=1, col=1)

            if show_sr_lines and not r_cluster.empty:
                # Filter & Sort
                r_plot = r_cluster[(r_cluster['Level'] >= limit_lower) & (r_cluster['Level'] <= limit_upper)]
                r_plot = r_plot.sort_values(by='Hits', ascending=False).head(5)
                
                for _, row in r_plot.iterrows():
                    # LINE Visualization (Red Dashed)
                    fig.add_shape(type="line", 
                        x0=df_view.index[0], x1=df_view.index[-1], 
                        y0=row['Level'], y1=row['Level'],
                        line=dict(color="red", width=2, dash="dash"), 
                        layer="below", row=1, col=1
                    )
                    # Label with Range
                    label_text = f"Res {row['Level']:.0f} [{row['Min']:.0f}-{row['Max']:.0f}]"
                    fig.add_annotation(x=df_view.index[-1], y=row['Level'], text=label_text, 
                                       showarrow=False, xanchor="left", font=dict(color="red", size=11), row=1, col=1)

            if show_fibs and fibs:
                colors = {"Ret 0.618": "gold", "Ret 0.5": "white", "Ret 0.382": "silver"}
                for label, price in fibs.items():
                    if "Swing" in label: continue
                    if limit_lower < price < limit_upper:
                        fig.add_hline(y=price, line_dash="dot", line_color=colors.get(label, "gray"), line_width=1, 
                                      annotation_text=f"{label}: {price:.0f}", row=1, col=1)

            # ROW 2-4 (Volume, MACD, RSI)
            vol_colors = ['#00cc96' if c >= o else '#ef553b' for c, o in zip(df_view['close'], df_view['open'])]
            fig.add_trace(go.Bar(x=df_view.index, y=df_view['volume'], name='Volume', marker_color=vol_colors), row=2, col=1)

            macd_col = next((c for c in df.columns if 'macd_' in c and 'h' not in c and 's' not in c), None)
            macdh_col = next((c for c in df.columns if 'macdh_' in c), None)
            macds_col = next((c for c in df.columns if 'macds_' in c), None)
            if macdh_col:
                hist_colors = ['#00cc96' if h >= 0 else '#ef553b' for h in df_view[macdh_col]]
                fig.add_trace(go.Bar(x=df_view.index, y=df_view[macdh_col], name='MACD Hist', marker_color=hist_colors), row=3, col=1)
                fig.add_trace(go.Scatter(x=df_view.index, y=df_view[macd_col], line=dict(color='cyan', width=1), name='MACD'), row=3, col=1)
                fig.add_trace(go.Scatter(x=df_view.index, y=df_view[macds_col], line=dict(color='orange', width=1), name='Signal'), row=3, col=1)

            fig.add_trace(go.Scatter(x=df_view.index, y=df_view['rsi_14'], name='RSI', line=dict(color='purple')), row=4, col=1)
            fig.add_hline(y=70, line_dash="dash", line_color="red", row=4, col=1)
            fig.add_hline(y=30, line_dash="dash", line_color="green", row=4, col=1)

            fig.update_layout(height=1000, xaxis_rangeslider_visible=False, template="plotly_dark", hovermode="x unified")
            fig.update_yaxes(range=[view_min * 0.95, view_max * 1.05], row=1, col=1)
            st.plotly_chart(fig, use_container_width=True)

            # --- BOTTOM TABLES ---
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Signal Backtest History")
                st.dataframe(rep.scan_for_signals(df_hist, s_cluster, r_cluster), use_container_width=True)
            with col2:
                st.subheader("Oscillator Reliability")
                st.dataframe(core.backtest_oscillator_signals(df_hist), use_container_width=True)

        # ==========================================
        # TAB 2: BACKTEST ENGINE
        # ==========================================
        with tab2:
            st.subheader("⚙️ Advanced Strategy Simulator")
            c1, c2, c3 = st.columns(3)
            strategy = c1.selectbox("Strategy Mode", ["SWING", "POSITION"])
            risk_pct = c2.number_input("Risk per Trade (%)", 0.5, 5.0, 2.0) / 100
            capital = c3.number_input("Initial Capital", 1_000_000, 1_000_000_000, 100_000_000)
            
            if st.button("Run Backtest Simulation"):
                trade_log, stats, df_equity = bt.run_advanced_backtest(df_hist, strategy, risk_pct, capital)
                if not trade_log.empty:
                    st.success(f"Backtest Completed for {strategy}")
                    s1, s2, s3, s4 = st.columns(4)
                    s1.metric("Net Profit", f"{stats['Total PnL']:,.0f}", f"{stats['Net Profit %']:.2%}")
                    s2.metric("Win Rate", f"{stats['Win Rate']:.1%}")
                    s3.metric("Expectancy", f"{stats['Expectancy']:,.0f}")
                    s4.metric("Final Balance", f"{stats['Final Balance']:,.0f}")
                    r1, r2, r3, r4 = st.columns(4)
                    r1.metric("Sharpe", f"{stats['Sharpe Ratio']:.2f}")
                    r2.metric("CAGR", f"{stats['CAGR']:.2%}")
                    r3.metric("Drawdown", f"{stats['Max Drawdown']:.2%}", delta_color="inverse")
                    r4.metric("Volatility", f"{stats['Volatility']:.2%}", delta_color="inverse")
                    st.divider()
                    if not df_equity.empty:
                        st.subheader("Equity Curve")
                        st.line_chart(df_equity['Equity'])
                    st.subheader("Trade Log")
                    st.dataframe(trade_log.sort_values("Entry Date", ascending=False), use_container_width=True)
                else:
                    st.warning("No trades triggered.")

        # ==========================================
        # TAB 3: OPTIMIZATION LAB
        # ==========================================
        with tab3:
            st.subheader("🧪 Automated Optimization")
            with st.expander("1. Parameter Grid Search", expanded=True):
                col_p1, col_p2 = st.columns(2)
                ema_fast_opts = col_p1.multiselect("Test EMA Fast", [10, 20, 30], default=[10, 20])
                ema_slow_opts = col_p2.multiselect("Test EMA Slow", [30, 50, 100], default=[30, 50])
                sl_opts = st.slider("Test SL Multiplier", 1.0, 4.0, (1.5, 3.0))
                
                if st.button("Run Optimization"):
                    param_grid = {'ema_fast': ema_fast_opts, 'ema_slow': ema_slow_opts, 'sl_mult': [sl_opts[0], sl_opts[1]]}
                    with st.spinner("Optimizing..."):
                        best_results = bt.run_grid_search(df_hist, param_grid)
                    if not best_results.empty:
                        st.success("Optimization Complete!")
                        st.dataframe(best_results.style.highlight_max(axis=0, subset=['Sharpe']), use_container_width=True)
                        st.session_state['best_params'] = best_results.iloc[0].to_dict()
                    else:
                        st.warning("Optimization failed.")

            with st.expander("2. Walk-Forward Validation"):
                if 'best_params' in st.session_state:
                    params = st.session_state['best_params']
                    clean_params = {k: v for k, v in params.items() if k in ['ema_fast', 'ema_slow', 'sl_mult']}
                    if st.button("Run Walk-Forward Analysis"):
                        wfo_results = bt.run_walk_forward(df_hist, clean_params)
                        if not wfo_results.empty:
                            st.dataframe(wfo_results, use_container_width=True)
                            st.bar_chart(wfo_results.set_index("Period End")['Test PnL'])
                        else:
                            st.warning("Not enough data.")
                else:
                    st.info("Run Optimization first.")

    except Exception as e:
        st.error(f"An error occurred: {e}")