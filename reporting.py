import pandas as pd
import numpy as np

def _to_scalar(val):
    if isinstance(val, pd.Series): return val.iloc[0]
    return val

def scan_for_signals(df, clustered_supports, clustered_resistances):
    # (Code scanning logic remains similar, just cleaner)
    df = df.copy()
    signals = []
    df['ma50_prev'] = df['sma_50'].shift(1)
    df['ma200_prev'] = df['sma_200'].shift(1)
    
    # Golden Cross
    golden_cross = (df['ma50_prev'] < df['ma200_prev']) & (df['sma_50'] > df['sma_200'])
    gc_dates = df[golden_cross].index
    for date in gc_dates:
        price = _to_scalar(df.loc[date, 'close'])
        signals.append({"Tanggal": date, "Sinyal": "GOLDEN CROSS", "Harga": price, "Detail": "Trend Start"})

    # Buy The Dip (Supertrend Version)
    # If Supertrend is UP (1) and Price touches near it
    if 'supertrend' in df.columns:
        st_uptrend = df['close'] > df['supertrend']
        near_st = (df['low'] <= df['supertrend'] * 1.02) & (df['close'] > df['supertrend'])
        btd_st = st_uptrend & near_st
        for date in df[btd_st].index:
            price = _to_scalar(df.loc[date, 'close'])
            signals.append({"Tanggal": date, "Sinyal": "SUPERTREND DIP", "Harga": price, "Detail": "Bounce off Supertrend"})

    return pd.DataFrame(signals).sort_values('Tanggal', ascending=False) if signals else pd.DataFrame()

def recommend_trade(df, clustered_s, clustered_r, structure, divergence_status):
    last_bar = df.iloc[-1]
    current_price = _to_scalar(last_bar['close'])
    
    rec = {
        "Action": "WAIT", "Type": "N/A", "Entry Zone": "N/A",
        "SL": 0, "TP": 0, "RR": 0, "Score": 0, "Reason": []
    }
    
    # --- Indicators ---
    adx = _to_scalar(last_bar.get('adx_14', 0))
    rsi = _to_scalar(last_bar.get('rsi_14', 50))
    rvol = _to_scalar(last_bar.get('rvol', 1))
    supertrend = _to_scalar(last_bar.get('supertrend', 0))
    
    # Bollinger Bands
    bbu = _to_scalar(last_bar.get('bbu_20_2.0', 0))
    bbl = _to_scalar(last_bar.get('bbl_20_2.0', 0))
    
    # --- 1. Context & Supertrend Logic ---
    is_supertrend_bullish = current_price > supertrend if supertrend > 0 else False
    
    if adx > 20 and "UPTREND" in structure:
        trade_type = "SWING (Trend Follow)"
    elif rsi < 35 or "Divergence" in divergence_status:
        trade_type = "QUICK (Reversal)"
    else:
        trade_type = "SCALPING / WAIT"

    # --- 2. Support Selection ---
    candidates = []
    # A. Supertrend (Dynamic Support)
    if is_supertrend_bullish:
        candidates.append({"Level": supertrend, "Type": "Supertrend Support", "Priority": 1})
        
    # B. Cluster
    valid_s = clustered_s[clustered_s['Level'] < current_price]
    if not valid_s.empty:
        # Get closest cluster
        nearest_c = valid_s.sort_values('Level', ascending=False).iloc[0]
        candidates.append({"Level": nearest_c.get('Max', nearest_c['Level']), "Type": "Structure Cluster", "Priority": 2})
        
    # C. Bollinger Lower Band (For Quick Trades)
    if trade_type == "QUICK (Reversal)" and bbl > 0:
        candidates.append({"Level": bbl, "Type": "Bollinger Lower Band", "Priority": 3})

    # Pick Best Support
    best_s = None
    min_dist = float('inf')
    for cand in candidates:
        dist = (current_price - cand['Level']) / current_price
        # Filter out invalid (price below support) or too far
        if 0 < dist < min_dist:
            min_dist = dist
            best_s = cand
    
    if not best_s: 
         # Fallback
         best_s = {"Level": current_price*0.95, "Type": "Fallback"}
         
    support_level = best_s['Level']
    s_source = best_s['Type']

    # --- 3. Scoring ---
    score = 0
    reasons = []
    
    # Trend Points
    if is_supertrend_bullish: score += 2; reasons.append("Supertrend is Bullish")
    elif score > 0: score -= 1 # Penalize if Supertrend is bearish
    
    if "UPTREND" in structure: score += 1
    
    # Momentum Points
    if 40 <= rsi <= 65: score += 1
    if rvol > 1.2: score += 1; reasons.append("Volume Spike")
    
    # Location Points
    dist_to_s = (current_price - support_level) / current_price
    if dist_to_s < 0.025: score += 2; reasons.append(f"Perfect Entry at {s_source}")
    elif dist_to_s < 0.06: score += 1; reasons.append(f"Near {s_source}")
    
    # Bollinger Logic (Warning)
    if bbu > 0 and current_price > bbu * 0.99:
        score -= 2
        reasons.append("Warning: Price touching Upper Bollinger Band (Expensive)")
    
    if "DIVERGENCE" in divergence_status: score += 2; reasons.append("Divergence Detected")

    # --- 4. Trade Plan ---
    atr = _to_scalar(last_bar.get('atrr_14', current_price*0.02))
    
    # Stop Loss: If Supertrend is support, SL is just below it
    if "Supertrend" in s_source:
        sl_price = support_level * 0.99 # Supertrend is already a stop level
    else:
        sl_price = support_level - (1.0 * atr)
        
    risk = current_price - sl_price
    tp_price = current_price + (risk * 1.5)
    
    # Resistance Check
    valid_r = clustered_r[clustered_r['Level'] > current_price]
    if not valid_r.empty:
        nearest_r = valid_r['Level'].min()
        if nearest_r < tp_price:
            tp_price = nearest_r * 0.99
            reasons.append(f"Target limited by Resistance {nearest_r:.0f}")

    rr = (tp_price - current_price) / risk if risk > 0 else 0
    
    rec['Entry Zone'] = f"{support_level:.0f} - {current_price:.0f}"
    rec['SL'] = int(sl_price)
    rec['TP'] = int(tp_price)
    rec['RR'] = round(rr, 2)
    rec['Score'] = score
    rec['Reason'] = reasons
    rec['Type'] = trade_type

    if score >= 6 and rr >= 1.5: rec['Action'] = "STRONG BUY"
    elif score >= 4 and rr >= 2.0: rec['Action'] = "SPECULATIVE BUY"
    else: rec['Action'] = "WAIT"

    return rec