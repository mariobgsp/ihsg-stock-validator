import pandas as pd
import numpy as np
import pandas_ta as ta
import itertools

def classify_candle_shape(row, avg_body_size):
    open_p, close_p = row['open'], row['close']
    high_p, low_p = row['high'], row['low']
    
    body_size = abs(close_p - open_p)
    total_range = high_p - low_p
    
    if total_range == 0: return "Doji"

    upper_shadow = high_p - max(open_p, close_p)
    lower_shadow = min(open_p, close_p) - low_p
    body_ratio = body_size / total_range
    
    shape = "Normal"
    if body_ratio > 0.8:
        shape = "Marubozu (Strong)"
    elif body_ratio < 0.1:
        shape = "Doji (Indecision)"
    elif lower_shadow > (body_size * 2) and upper_shadow < body_size:
        shape = "Hammer/Pinbar (Bullish Rejection)"
    elif upper_shadow > (body_size * 2) and lower_shadow < body_size:
        shape = "Shooting Star (Bearish Rejection)"
    elif body_size > (avg_body_size * 1.5):
        shape = "Big Candle (Momentum)"
        
    return shape

def run_advanced_backtest(df, strategy_type="SWING", risk_per_trade=0.02, initial_capital=100000000, params=None, start_date=None):
    """
    Mesin Backtest dengan dukungan Parameter Dinamis & Start Date Filter.
    """
    df = df.copy()
    
    # Default Parameters
    default_params = {
        'ema_fast': 20, 'ema_slow': 50, 'ema_trend': 200,
        'rsi_min': 40, 'rsi_max': 75,
        'sl_mult': 1.5, 'tp1_mult': 1.5, 'tp2_mult': 3.0,
        'use_trend_filter': False
    }
    
    if strategy_type == "POSITION":
        default_params.update({
            'sl_mult': 3.0, 'tp1_mult': 2.0, 'tp2_mult': 5.0,
            'use_trend_filter': True, 'rsi_min': 45, 'rsi_max': 80
        })
    
    p = default_params.copy()
    if params:
        p.update(params)

    # --- 1. Ensure Indicators ---
    if f"ema_{p['ema_fast']}" not in df.columns: 
        df[f"ema_{p['ema_fast']}"] = ta.ema(close=df['close'], length=p['ema_fast'])
    if f"ema_{p['ema_slow']}" not in df.columns: 
        df[f"ema_{p['ema_slow']}"] = ta.ema(close=df['close'], length=p['ema_slow'])
    if f"ema_{p['ema_trend']}" not in df.columns: 
        df[f"ema_{p['ema_trend']}"] = ta.ema(close=df['close'], length=p['ema_trend'])
        
    if 'atrr_14' not in df.columns: df['atrr_14'] = ta.atr(high=df['high'], low=df['low'], close=df['Close'], length=14)
    if 'rsi_14' not in df.columns: df['rsi_14'] = ta.rsi(close=df['close'], length=14)
    
    stoch_exists = any(c.startswith('stochk') or c.startswith('STOCHk') for c in df.columns)
    if not stoch_exists:
        stoch = df.ta.stoch(high=df['high'], low=df['low'], close=df['close'], k=14, d=3, smooth_k=3)
        if stoch is not None:
            df['stoch_k'] = stoch.iloc[:, 0]
            df['stoch_d'] = stoch.iloc[:, 1]
    else:
        k_col = next((c for c in df.columns if c.lower().startswith('stochk')), None)
        d_col = next((c for c in df.columns if c.lower().startswith('stochd')), None)
        if k_col: df['stoch_k'] = df[k_col]
        if d_col: df['stoch_d'] = df[d_col]

    df.fillna(0, inplace=True)

    col_fast = f"ema_{p['ema_fast']}"
    col_slow = f"ema_{p['ema_slow']}"
    col_trend = f"ema_{p['ema_trend']}"

    capital = initial_capital
    trades = []
    equity_curve = []
    in_position = False
    
    # State
    entry_price = 0
    entry_date = None
    position_size = 0 
    stop_loss = 0
    take_profit_1 = 0
    take_profit_2 = 0
    tp1_hit = False
    signal_reason = ""
    entry_rsi = 0
    entry_stoch = 0
    
    # --- Loop Logic ---
    start_idx = 200
    if len(df) < 200: start_idx = 50 

    for i in range(start_idx, len(df)):
        bar = df.iloc[i]
        curr_date = df.index[i]
        
        # WFO Filter
        if start_date and curr_date < pd.to_datetime(start_date):
            continue
        
        # Equity
        current_equity = capital
        if in_position:
            current_equity = capital + (position_size * bar['close'])
        equity_curve.append({"Date": curr_date, "Equity": current_equity})

        # Exit
        if in_position:
            if bar['low'] <= stop_loss:
                exit_price = stop_loss if bar['open'] > stop_loss else bar['open']
                proceeds = position_size * exit_price
                capital += proceeds
                pnl = proceeds - (position_size * entry_price)
                
                trades.append({
                    "Entry Date": entry_date, "Exit Date": curr_date, "Signal": signal_reason,
                    "Type": "SL Hit", "Entry": entry_price, "Exit": exit_price,
                    "PnL": pnl, "R-Multiple": -1.0, "Balance": capital,
                    "RSI": f"{entry_rsi:.1f}", "Stoch": f"{entry_stoch:.1f}"
                })
                in_position = False
                position_size = 0
                continue
            
            if not tp1_hit and bar['high'] >= take_profit_1:
                exit_price = take_profit_1
                sold_size = int(position_size * 0.5)
                proceeds = sold_size * exit_price
                capital += proceeds
                pnl = proceeds - (sold_size * entry_price)
                position_size -= sold_size 
                tp1_hit = True
                stop_loss = entry_price 
                
                trades.append({
                    "Entry Date": entry_date, "Exit Date": curr_date, "Signal": signal_reason,
                    "Type": "TP 1 (Partial)", "Entry": entry_price, "Exit": exit_price,
                    "PnL": pnl, "R-Multiple": p['tp1_mult'], "Balance": capital,
                    "RSI": f"{entry_rsi:.1f}", "Stoch": f"{entry_stoch:.1f}"
                })

            if tp1_hit and bar['high'] >= take_profit_2:
                exit_price = take_profit_2
                proceeds = position_size * exit_price
                capital += proceeds
                pnl = proceeds - (position_size * entry_price)
                
                trades.append({
                    "Entry Date": entry_date, "Exit Date": curr_date, "Signal": signal_reason,
                    "Type": "TP 2 (Full)", "Entry": entry_price, "Exit": exit_price,
                    "PnL": pnl, "R-Multiple": p['tp2_mult'], "Balance": capital,
                    "RSI": f"{entry_rsi:.1f}", "Stoch": f"{entry_stoch:.1f}"
                })
                in_position = False
                position_size = 0
                continue

            new_sl = bar['close'] - (bar['atrr_14'] * p['sl_mult'])
            if new_sl > stop_loss: stop_loss = new_sl 

        # Entry
        if not in_position:
            trend_ok = True
            if p['use_trend_filter']: trend_ok = bar['close'] > bar[col_trend]
            
            ema_bullish = (bar[col_fast] > bar[col_slow]) and (bar['close'] > bar[col_fast])
            
            curr_rsi = bar.get('rsi_14', 50)
            curr_stoch_k = bar.get('stoch_k', 50)
            curr_stoch_d = bar.get('stoch_d', 50)
            
            active_signal = None
            if trend_ok and ema_bullish and (p['rsi_min'] < curr_rsi < p['rsi_max']) and (curr_stoch_k > curr_stoch_d):
                active_signal = "Trend Pullback + Momentum"
            elif not ema_bullish and (df.iloc[i-1]['rsi_14'] < 30 and curr_rsi >= 30):
                 active_signal = "Reversal: RSI Oversold Bounce"

            if active_signal:
                entry_price = bar['close']
                entry_date = curr_date
                entry_rsi = curr_rsi
                entry_stoch = curr_stoch_k
                signal_reason = active_signal
                
                atr_val = bar.get('atrr_14', entry_price*0.02)
                risk_amt = capital * risk_per_trade
                dist_sl = atr_val * p['sl_mult']
                
                stop_loss = entry_price - dist_sl
                take_profit_1 = entry_price + (dist_sl * p['tp1_mult']) 
                take_profit_2 = entry_price + (dist_sl * p['tp2_mult']) 
                
                risk_per_share = entry_price - stop_loss
                if risk_per_share > 0:
                    potential_size = int(risk_amt / risk_per_share)
                    cost = potential_size * entry_price
                    
                    if cost < capital:
                        capital -= cost
                        position_size = potential_size
                        in_position = True
                        tp1_hit = False

    df_trades = pd.DataFrame(trades)
    
    if equity_curve:
        df_equity = pd.DataFrame(equity_curve)
        if 'Date' in df_equity.columns: df_equity = df_equity.set_index("Date")
    else:
        df_equity = pd.DataFrame(columns=['Equity'])

    # --- STATS INITIALIZATION (FIXED) ---
    # Initialize with default values to prevent KeyErrors
    stats = {
        "Total PnL": 0, 
        "Win Rate": 0, 
        "Sharpe Ratio": 0, 
        "Max Drawdown": 0, 
        "Final Balance": capital, 
        "Net Profit %": 0,
        "Expectancy": 0,
        "CAGR": 0,
        "Volatility": 0,
        "Total Trades": 0
    }
    
    if not df_trades.empty and not df_equity.empty:
        wins = df_trades[df_trades['PnL'] > 0]
        losses = df_trades[df_trades['PnL'] <= 0]
        stats["Win Rate"] = len(wins) / len(df_trades)
        stats["Total PnL"] = df_trades['PnL'].sum()
        stats["Total Trades"] = len(df_trades)
        
        # Expectancy
        avg_win = wins['PnL'].mean() if not wins.empty else 0
        avg_loss = losses['PnL'].mean() if not losses.empty else 0
        loss_rate = 1 - stats["Win Rate"]
        stats["Expectancy"] = (stats["Win Rate"] * avg_win) + (loss_rate * avg_loss)

        # CAGR & Volatility
        df_equity['Returns'] = df_equity['Equity'].pct_change()
        days = (df_equity.index[-1] - df_equity.index[0]).days
        years = days / 365.25
        final_eq = df_equity['Equity'].iloc[-1]
        
        stats["Final Balance"] = final_eq
        stats["Net Profit %"] = (final_eq - initial_capital)/initial_capital
        
        stats["CAGR"] = (final_eq / initial_capital) ** (1 / years) - 1 if years > 0 else 0
        
        volatility = df_equity['Returns'].std() * np.sqrt(252)
        stats["Volatility"] = volatility
        
        mean_return = df_equity['Returns'].mean() * 252
        stats["Sharpe Ratio"] = mean_return / volatility if volatility != 0 else 0
        
        df_equity['Peak'] = df_equity['Equity'].cummax()
        dd = (df_equity['Equity'] - df_equity['Peak']) / df_equity['Peak']
        stats["Max Drawdown"] = dd.min()
        
    return df_trades, stats, df_equity

def run_grid_search(df, param_grid):
    keys, values = zip(*param_grid.items())
    combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    
    results = []
    max_runs = 20 
    if len(combinations) > max_runs:
        import random
        combinations = random.sample(combinations, max_runs)
    
    print(f"Running optimization on {len(combinations)} combinations...")
    
    for params in combinations:
        _, stats, _ = run_advanced_backtest(df, strategy_type="CUSTOM", params=params)
        res = params.copy()
        res['Sharpe'] = stats['Sharpe Ratio']
        res['Net Profit'] = stats['Total PnL']
        res['DD'] = stats['Max Drawdown']
        res['Trades'] = stats.get('Total Trades', 0)
        results.append(res)
        
    df_results = pd.DataFrame(results)
    if not df_results.empty:
        return df_results.sort_values(by='Sharpe', ascending=False).head(5)
    return pd.DataFrame()

def run_walk_forward(df, params, train_size_days=180, test_size_days=60):
    total_days = (df.index[-1] - df.index[0]).days
    segments = []
    current_start = df.index[0]
    warmup_days = 365 
    
    while True:
        train_end = current_start + pd.Timedelta(days=train_size_days)
        test_end = train_end + pd.Timedelta(days=test_size_days)
        
        if test_end > df.index[-1]: break
        
        train_slice_start = current_start - pd.Timedelta(days=warmup_days)
        if train_slice_start < df.index[0]: train_slice_start = df.index[0]
        df_train_context = df[train_slice_start:train_end]
        
        test_slice_start = train_end - pd.Timedelta(days=warmup_days)
        if test_slice_start < df.index[0]: test_slice_start = df.index[0]
        df_test_context = df[test_slice_start:test_end]
        
        if len(df_train_context) < 200 or len(df_test_context) < 200:
            current_start += pd.Timedelta(days=test_size_days)
            continue
            
        _, stats_train, _ = run_advanced_backtest(df_train_context, params=params, start_date=current_start)
        _, stats_test, _ = run_advanced_backtest(df_test_context, params=params, start_date=train_end)
        
        segments.append({
            "Period Start": train_end.date(),
            "Period End": test_end.date(),
            "Train Sharpe": stats_train.get('Sharpe Ratio', 0),
            "Test Sharpe": stats_test.get('Sharpe Ratio', 0),
            "Test PnL": stats_test.get('Total PnL', 0)
        })
        
        current_start += pd.Timedelta(days=test_size_days)
        
    return pd.DataFrame(segments)