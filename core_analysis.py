import pandas as pd
import numpy as np
from scipy.signal import find_peaks

def _to_scalar(val):
    if isinstance(val, pd.Series): return val.iloc[0]
    return val

def cluster_levels(levels, cluster_percent=0.025):
    """
    Mengelompokkan level harga menjadi ZONA.
    Fixed: Increased minimum box thickness for visibility.
    """
    if levels.empty:
        return pd.DataFrame(columns=['Level', 'Min', 'Max', 'Hits', 'Kekuatan'])
    
    sorted_levels = np.sort(levels.values)
    clusters = []
    
    i = 0
    while i < len(sorted_levels):
        current_cluster = [sorted_levels[i]]
        base_price = sorted_levels[i]
        
        j = i + 1
        while j < len(sorted_levels):
            level = sorted_levels[j]
            if (level - base_price) / base_price <= cluster_percent:
                current_cluster.append(level)
                j += 1
            else:
                break 
        
        cluster_mean = np.mean(current_cluster)
        cluster_min = np.min(current_cluster)
        cluster_max = np.max(current_cluster)
        
        # FIX: Make box thicker (1.5% padding) if min ~= max
        # This ensures the box is visible on the chart
        if (cluster_max - cluster_min) / cluster_mean < 0.005:
            padding = cluster_mean * 0.015 
            cluster_min = cluster_mean - padding
            cluster_max = cluster_mean + padding
            
        cluster_hits = len(current_cluster)
        
        clusters.append({
            'Level': cluster_mean, 
            'Min': cluster_min, 
            'Max': cluster_max, 
            'Hits': cluster_hits
        })
        i = j 
        
    df_clusters = pd.DataFrame(clusters)
    if not df_clusters.empty:
        df_clusters = df_clusters.sort_values(by='Hits', ascending=False)
        df_clusters['Level'] = df_clusters['Level'].round(0)
        
        def strength_label(hits):
            if hits >= 4: return "Sangat Kuat"
            elif hits >= 2: return "Kuat"
            else: return "Moderat"
        
        df_clusters['Kekuatan'] = df_clusters['Hits'].apply(strength_label)
    else:
        return pd.DataFrame(columns=['Level', 'Min', 'Max', 'Hits', 'Kekuatan'])
    
    return df_clusters

def find_support_resistance(df):
    df_clean = df.dropna(subset=['high', 'low'])
    if df_clean.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.Series(), pd.Series()

    prominence_h = df_clean['high'].std() * 0.4 
    prominence_l = df_clean['low'].std() * 0.4

    try:
        peaks_idx, _ = find_peaks(df_clean['high'], distance=5, prominence=prominence_h)
        valleys_idx, _ = find_peaks(-df_clean['low'], distance=5, prominence=prominence_l)

        raw_supports = df_clean['low'].iloc[valleys_idx]
        raw_resistances = df_clean['high'].iloc[peaks_idx]
        
        clustered_supports = cluster_levels(raw_supports, cluster_percent=0.02) 
        clustered_resistances = cluster_levels(raw_resistances, cluster_percent=0.02)
        
        return clustered_supports, clustered_resistances, raw_supports, raw_resistances
    except Exception as e:
        print(f"Error detecting S/R: {e}")
        return pd.DataFrame(columns=['Level']), pd.DataFrame(columns=['Level']), pd.Series(), pd.Series()

def calculate_fibonacci_levels(detail_df):
    clean_df = detail_df.dropna(subset=['close'])
    if clean_df.empty: return {}

    swing_low = clean_df['low'].min()
    swing_high = clean_df['high'].max()
    diff = swing_high - swing_low
    
    levels = {
        "Swing High": swing_high,
        "Swing Low": swing_low,
        "Ret 0.382": swing_high - (diff * 0.382),
        "Ret 0.5": swing_high - (diff * 0.5),
        "Ret 0.618": swing_high - (diff * 0.618),
        "Ext 1.272": swing_high + (diff * 0.272),
        "Ext 1.618": swing_high + (diff * 0.618)
    }
    return levels

def detect_market_structure(df):
    df_clean = df.dropna(subset=['close'])
    if len(df_clean) < 200: return "DATA KURANG"
    last_bar = df_clean.iloc[-1]
    price = _to_scalar(last_bar['close'])
    ma50 = _to_scalar(last_bar.get('sma_50', 0))
    ma200 = _to_scalar(last_bar.get('sma_200', 0))
    
    if price > ma50 and ma50 > ma200: return "STRONG UPTREND"
    if price < ma50 and ma50 < ma200: return "STRONG DOWNTREND"
    if price > ma200 and price < ma50: return "WEAK UPTREND (Correction)"
    if price < ma200 and price > ma50: return "WEAK DOWNTREND (Recovery)"
    if price > ma200: return "UPTREND (Early)"
    return "DOWNTREND"

def detect_divergence(df, lookback=30):
    df_slice = df.iloc[-lookback:].copy()
    if len(df_slice) < 10: return "Normal"
    price_lows_idx, _ = find_peaks(-df_slice['low'], distance=5)
    rsi_lows_idx, _ = find_peaks(-df_slice['rsi_14'], distance=5)
    if len(price_lows_idx) < 2 or len(rsi_lows_idx) < 2: return "Normal"
    p1, p2 = price_lows_idx[-2], price_lows_idx[-1]
    if df_slice['low'].iloc[p2] < df_slice['low'].iloc[p1] and \
       df_slice['rsi_14'].iloc[p2] > df_slice['rsi_14'].iloc[p1]:
        return "BULLISH DIVERGENCE"
    return "Normal"

def backtest_oscillator_signals(df, lookahead=5):
    df = df.copy()
    if 'rsi_14' not in df.columns: return pd.DataFrame()
    
    stoch_k_col = next((c for c in df.columns if 'stochk' in c), None)
    if not stoch_k_col: return pd.DataFrame()

    rsi_buy = (df['rsi_14'].shift(1) < 30) & (df['rsi_14'] >= 30)
    rsi_sell = (df['rsi_14'].shift(1) > 70) & (df['rsi_14'] <= 70)
    stoch_buy = (df[stoch_k_col].shift(1) < 20) & (df[stoch_k_col] >= 20)
    stoch_sell = (df[stoch_k_col].shift(1) > 80) & (df[stoch_k_col] <= 80)

    scenarios = {
        "RSI Oversold Bounce (<30)": rsi_buy,
        "Stoch Oversold Bounce (<20)": stoch_buy,
        "RSI Overbought Drop (>70)": rsi_sell,
        "Stoch Overbought Drop (>80)": stoch_sell
    }

    results = []
    for name, signal_series in scenarios.items():
        entry_dates = df[signal_series].index
        returns = []
        for date in entry_dates:
            try:
                idx = df.index.get_loc(date)
                if idx + lookahead >= len(df): continue
                entry_price = _to_scalar(df.iloc[idx]['close'])
                exit_price = _to_scalar(df.iloc[idx + lookahead]['close'])
                ret = (exit_price - entry_price) / entry_price
                if "Drop" in name: ret = -ret 
                returns.append(ret)
            except: continue
            
        if returns:
            avg_ret = np.mean(returns)
            win_rate = np.sum(np.array(returns) > 0) / len(returns)
            results.append({
                "Setup": name,
                "Count": len(returns),
                "Win Rate (5 Days)": win_rate,
                "Avg Return": avg_ret
            })

    res_df = pd.DataFrame(results)
    if not res_df.empty:
        res_df['Win Rate (5 Days)'] = res_df['Win Rate (5 Days)'].apply(lambda x: f"{x:.0%}")
        res_df['Avg Return'] = res_df['Avg Return'].apply(lambda x: f"{x:+.2%}")
    return res_df