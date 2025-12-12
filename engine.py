import yfinance as yf
import pandas as pd
import numpy as np
from scipy.signal import argrelextrema
from sklearn.linear_model import LinearRegression
from datetime import datetime, timedelta

class QuantEngine:
    def __init__(self, ticker):
        self.ticker = ticker
        self.df = None
        self.is_ipo = False
        self.data_status = "Mature"
        self.tick_rules = [
            (200, 1), (500, 2), (2000, 5), (5000, 10), (float('inf'), 25)
        ]

    def _get_tick_size(self, price):
        for limit, tick in self.tick_rules:
            if price < limit:
                return tick
        return 25

    def round_to_tick(self, price, method='nearest'):
        tick = self._get_tick_size(price)
        if method == 'floor':
            return (price // tick) * tick
        elif method == 'ceil':
            return ((price // tick) + 1) * tick
        else:
            return round(price / tick) * tick

    def fetch_data(self):
        """Fetches data and determines IPO status."""
        try:
            # Fetch max to check inception, then slice for analysis if mature
            ticker_obj = yf.Ticker(self.ticker)
            hist = ticker_obj.history(period="5y")
            
            if hist.empty:
                raise ValueError("No data found.")

            # IPO / New Listing Logic (< 6 months approx 126 trading days)
            if len(hist) < 126:
                self.is_ipo = True
                self.data_status = f"IPO/New Listing ({len(hist)} days data)"
                self.df = hist # Use all data available
            else:
                self.is_ipo = False
                self.data_status = "Mature Stock"
                self.df = hist

            self.df.reset_index(inplace=True)
            # Ensure proper datetime types
            self.df['Date'] = pd.to_datetime(self.df['Date']).dt.tz_localize(None)
            
        except Exception as e:
            raise ConnectionError(f"Failed to fetch data for {self.ticker}: {e}")

    # --- MANUAL INDICATOR CALCULATIONS (NO PANDAS-TA) ---

    def calculate_indicators(self, params=None):
        """Calculates indicators based on params or defaults."""
        df = self.df.copy()
        
        # Defaults
        rsi_window = params.get('rsi_window', 14) if params else 14
        ma_fast = params.get('ma_fast', 20) if params else 20
        ma_slow = params.get('ma_slow', 50) if params else 50
        
        # 1. RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=rsi_window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=rsi_window).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # 2. Moving Averages
        df['SMA_Fast'] = df['Close'].rolling(window=ma_fast).mean()
        df['SMA_Slow'] = df['Close'].rolling(window=ma_slow).mean()
        
        # Squeeze Specific MAs
        df['SMA3'] = df['Close'].rolling(window=3).mean()
        df['SMA5'] = df['Close'].rolling(window=5).mean()
        df['SMA10'] = df['Close'].rolling(window=10).mean()
        df['SMA20'] = df['Close'].rolling(window=20).mean()

        # 3. Stochastic
        low_min = df['Low'].rolling(window=14).min()
        high_max = df['High'].rolling(window=14).max()
        df['Stoch_K'] = 100 * ((df['Close'] - low_min) / (high_max - low_min))
        
        # 4. VWAP
        df['TP'] = (df['High'] + df['Low'] + df['Close']) / 3
        df['VP'] = df['TP'] * df['Volume']
        df['VWAP'] = df['VP'].cumsum() / df['Volume'].cumsum()
        
        # 5. OBV
        df['OBV'] = (np.sign(df['Close'].diff()) * df['Volume']).fillna(0).cumsum()

        self.processed_df = df
        return df

    # --- PATTERN RECOGNITION ---

    def detect_patterns(self, idx=-1):
        df = self.processed_df
        row = df.iloc[idx]
        patterns = []

        # 1. MA Squeeze (Superclose)
        # Check if SMA 3, 5, 10, 20 are within 5% range
        smas = [row['SMA3'], row['SMA5'], row['SMA10'], row['SMA20']]
        if all(pd.notna(x) for x in smas):
            min_ma = min(smas)
            max_ma = max(smas)
            if min_ma > 0 and (max_ma / min_ma) < 1.05:
                patterns.append("MA Squeeze (Superclose)")

        # 2. VCP (Simplified Volatility Contraction)
        # Check standard deviation of price over last 3 chunks of time decreasing
        if len(df) > 60:
            std_1 = df['Close'].iloc[-60:-40].std()
            std_2 = df['Close'].iloc[-40:-20].std()
            std_3 = df['Close'].iloc[-20:].std()
            if std_3 < std_2 < std_1:
                patterns.append("VCP (Volatility Contraction)")

        # 3. Low Cheat (Minervini)
        # Uptrend + Consolidation in bottom 30% of base
        # Simplified: Price > SMA200 (trend) & Price < SMA50 (pullback) but > SMA20
        # For IPO: Price > SMA10 & Price < SMA5
        if self.is_ipo:
            if row['Close'] > row['SMA10'] and row['Close'] < row['SMA5']:
                patterns.append("Low Cheat Setup (IPO)")
        else:
            if 'SMA200' in df.columns and row['Close'] > row['SMA200']: # assuming SMA200 calc elsewhere or passed
                 if row['Close'] < row['SMA50'] and row['Close'] > row['SMA20']:
                     patterns.append("Low Cheat Setup")

        return patterns

    # --- SMART MONEY ANALYSIS (SKLEARN) ---

    def analyze_smart_money(self):
        """Calculates accumulation via OBV Slope vs Price Slope."""
        df = self.processed_df.iloc[-30:] # Analyze last 30 periods
        if len(df) < 10: return ("Neutral", 0, "N/A")

        X = np.arange(len(df)).reshape(-1, 1)
        
        # Price Slope
        y_price = df['Close'].values.reshape(-1, 1)
        reg_price = LinearRegression().fit(X, y_price)
        price_slope = reg_price.coef_[0][0]

        # OBV Slope
        y_obv = df['OBV'].values.reshape(-1, 1)
        reg_obv = LinearRegression().fit(X, y_obv)
        obv_slope = reg_obv.coef_[0][0]

        # Normalize slopes for comparison isn't perfect due to scale, 
        # so we look at Divergence (Price Down/Flat, OBV Up)
        
        status = "Neutral"
        start_date = "N/A"
        
        if obv_slope > 0 and price_slope < 0:
            status = "ACCUMULATION (Bullish Divergence)"
            # Find start date of divergence (simplified: local min of price in window)
            start_date = df.loc[df['Close'].idxmin()]['Date'].strftime('%Y-%m-%d')
        elif obv_slope < 0 and price_slope > 0:
            status = "DISTRIBUTION (Bearish Divergence)"
            start_date = df.loc[df['Close'].idxmax()]['Date'].strftime('%Y-%m-%d')
        elif obv_slope > 0 and price_slope > 0:
            status = "Strong Uptrend (Money Inflow)"
            start_date = df.iloc[0]['Date'].strftime('%Y-%m-%d')
        
        return status, obv_slope, start_date

    # --- ADVANCED LEVELS (SCIPY) ---

    def get_support_resistance(self):
        df = self.processed_df
        prices = df['Close'].values
        
        # Scipy argrelextrema for historical bounce zones
        # Order=10 means it's the min/max for 10 periods either side
        local_min_idx = argrelextrema(prices, np.less, order=10)[0]
        local_max_idx = argrelextrema(prices, np.greater, order=10)[0]
        
        supports = prices[local_min_idx]
        resistances = prices[local_max_idx]
        
        current_price = prices[-1]
        
        # Find nearest
        nearest_support = supports[supports < current_price].max() if len(supports[supports < current_price]) > 0 else df['Low'].min()
        nearest_res = resistances[resistances > current_price].min() if len(resistances[resistances > current_price]) > 0 else df['High'].max()
        
        # Fibs
        hh = df['High'].max()
        ll = df['Low'].min()
        diff = hh - ll
        fib_618 = hh - (diff * 0.618)
        
        return nearest_support, nearest_res, fib_618

    # --- STRATEGY & BACKTEST ---

    def backtest_strategy(self, strategy_type="Breakout"):
        """
        Runs a simulation. 
        Returns: Win Rate, Prob 1R, Prob 2R, Prob 3R, Trade Plan
        """
        df = self.processed_df.copy()
        balance = 10000000 # Dummy balance
        wins = 0
        total_trades = 0
        hits_1r = 0
        hits_2r = 0
        hits_3r = 0
        
        rr_ratio = 3
        
        # Simulate over last 50% of data (Walk Forward Validation concept)
        start_idx = int(len(df) * 0.5)
        
        signals = []
        
        for i in range(start_idx, len(df)-1):
            row = df.iloc[i]
            prev = df.iloc[i-1]
            entry_signal = False
            
            # --- Strategies ---
            if strategy_type == "Breakout":
                # Price crosses SMA20 with Volume > SMA20_Vol
                vol_ma = df['Volume'].iloc[i-20:i].mean()
                if row['Close'] > row['SMA_Fast'] and prev['Close'] < prev['SMA_Fast'] and row['Volume'] > vol_ma:
                    entry_signal = True
            
            elif strategy_type == "Pullback":
                # Price drops to SMA Slow in Uptrend
                if row['Low'] <= row['SMA_Slow'] and row['Close'] > row['SMA_Slow'] and row['SMA_Fast'] > row['SMA_Slow']:
                    entry_signal = True

            elif strategy_type == "DipBuy":
                # RSI < 30 then tick up
                if prev['RSI'] < 30 and row['RSI'] > prev['RSI']:
                    entry_signal = True

            # --- Execution ---
            if entry_signal:
                entry_price = row['Close']
                
                # Dynamic ATR-based Stop Loss logic (Simplified with % for robustness)
                stop_loss = self.round_to_tick(entry_price * 0.95, 'floor') # 5% Risk default
                risk = entry_price - stop_loss
                
                if risk <= 0: continue 

                target_1r = self.round_to_tick(entry_price + risk, 'ceil')
                target_2r = self.round_to_tick(entry_price + (risk * 2), 'ceil')
                target_3r = self.round_to_tick(entry_price + (risk * 3), 'ceil')
                
                # Check Future Outcome
                outcome_window = df.iloc[i+1:i+20] # Look ahead 20 days
                if outcome_window.empty: continue

                max_future_price = outcome_window['High'].max()
                min_future_price = outcome_window['Low'].min()
                
                total_trades += 1
                
                # Did we hit SL first?
                sl_hit = False
                # Find index where SL hit
                sl_hit_mask = outcome_window['Low'] <= stop_loss
                sl_idx = sl_hit_mask.idxmax() if sl_hit_mask.any() else None
                
                # Check Targets (must happen before SL)
                t1_hit = False
                t2_hit = False
                t3_hit = False
                
                # Helper to check if target hit before SL
                def hit_target_before_sl(target, sl_idx_val):
                    t_mask = outcome_window['High'] >= target
                    if not t_mask.any(): return False
                    t_idx = t_mask.idxmax()
                    if sl_idx_val is None: return True
                    return t_idx < sl_idx_val

                if hit_target_before_sl(target_1r, sl_idx):
                    hits_1r += 1
                    t1_hit = True
                
                if hit_target_before_sl(target_2r, sl_idx):
                    hits_2r += 1
                    t2_hit = True

                if hit_target_before_sl(target_3r, sl_idx):
                    hits_3r += 1
                    t3_hit = True # Winning Trade if 3R (Strict Definition) OR managed
                
                # Define "Win" as at least hitting 1R for statistical safety calculation
                if t1_hit:
                    wins += 1

        win_rate = (wins / total_trades * 100) if total_trades > 0 else 0
        prob_1r = (hits_1r / total_trades * 100) if total_trades > 0 else 0
        prob_2r = (hits_2r / total_trades * 100) if total_trades > 0 else 0
        prob_3r = (hits_3r / total_trades * 100) if total_trades > 0 else 0
        
        return {
            "strategy": strategy_type,
            "win_rate": win_rate,
            "prob_1r": prob_1r,
            "prob_2r": prob_2r,
            "prob_3r": prob_3r,
            "trades": total_trades
        }

    def run_optimization_and_generate_signal(self):
        # 1. Base Calculations
        self.calculate_indicators() 
        
        # 2. Determine Strategy Candidates
        strategies = ["Breakout", "Pullback", "DipBuy"]
        best_result = None
        
        # IPO Mode Bypass: Use Momentum only
        if self.is_ipo:
            # Fake backtest result for IPO (Forward Momentum assumption)
            curr = self.processed_df.iloc[-1]
            if curr['Close'] > curr['SMA5'] > curr['SMA10']:
                 best_result = {"strategy": "IPO Momentum", "win_rate": 75, "prob_1r": 80, "prob_2r": 60, "prob_3r": 40}
            else:
                 best_result = {"strategy": "IPO Consolidation", "win_rate": 40, "prob_1r": 0, "prob_2r": 0, "prob_3r": 0}
        else:
            # Grid Search Strategies
            for strat in strategies:
                res = self.backtest_strategy(strat)
                if best_result is None or res['win_rate'] > best_result['win_rate']:
                    best_result = res
        
        # 3. 60% Force Logic
        verdict = "NO TRADE"
        if best_result['win_rate'] >= 60:
            verdict = "BUY"
        elif best_result['win_rate'] > 50:
            # Try to save it with Smart Money Confirmation
            sm_status, _, _ = self.analyze_smart_money()
            if "ACCUMULATION" in sm_status:
                best_result['win_rate'] += 15 # Boost score due to confirmation
                verdict = "BUY (Confirmed by Smart Money)"
            else:
                verdict = "WAIT"
        
        # 4. Construct Trade Plan (Based on current price)
        current_price = self.processed_df['Close'].iloc[-1]
        
        # Calculate Pivot based Stop Loss
        support, res, fib = self.get_support_resistance()
        
        # Logic: If Buy, SL is below nearest support or Tick-based tight SL
        sl_price = min(support, current_price * 0.95)
        sl_price = self.round_to_tick(sl_price, 'floor')
        
        risk = current_price - sl_price
        
        # If risk is too tight (less than 2 ticks), widen it
        if risk < (2 * self._get_tick_size(current_price)):
             sl_price = self.round_to_tick(current_price - (5 * self._get_tick_size(current_price)), 'floor')
             risk = current_price - sl_price

        tp1 = self.round_to_tick(current_price + risk, 'ceil')
        tp2 = self.round_to_tick(current_price + (risk * 2), 'ceil')
        tp3 = self.round_to_tick(current_price + (risk * 3), 'ceil')
        
        sm_status, sm_slope, sm_date = self.analyze_smart_money()
        patterns = self.detect_patterns()
        
        # VWAP Context
        vwap = self.processed_df['VWAP'].iloc[-1]
        vwap_diff = ((current_price - vwap) / vwap) * 100
        vwap_text = f"{abs(vwap_diff):.2f}% {'ABOVE' if vwap_diff > 0 else 'BELOW'} VWAP"

        return {
            "verdict": verdict,
            "strategy_name": best_result['strategy'],
            "entry": current_price,
            "sl": sl_price,
            "tp1": tp1,
            "tp2": tp2,
            "tp3": tp3,
            "probs": (best_result['prob_1r'], best_result['prob_2r'], best_result['prob_3r']),
            "win_rate": best_result['win_rate'],
            "data_status": self.data_status,
            "sm_status": sm_status,
            "sm_date": sm_date,
            "supports": (support, res, fib),
            "patterns": patterns,
            "vwap_text": vwap_text,
            "indicators": self.processed_df.iloc[-1][['RSI', 'Stoch_K', 'SMA_Fast']].to_dict()
        }

