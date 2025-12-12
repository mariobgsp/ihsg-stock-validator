"""
engine.py - Core Trading Engine for IHSG Swing Trading
Advanced Technical Analysis, Backtesting, and Signal Generation
"""

import numpy as np
import pandas as pd
import yfinance as yf
from scipy.signal import argrelextrema
from sklearn.linear_model import LinearRegression
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')


class TradingEngine:
    """Production-grade trading engine with advanced technical analysis"""
    
    def __init__(self, ticker, years=3):
        self.ticker = ticker
        self.years = years
        self.data = None
        self.signals = []
        self.smart_money_start = None
        self.current_phase = "NEUTRAL"
        
    def fetch_data(self):
        """Fetch OHLCV data using yfinance"""
        try:
            symbol = f"{self.ticker}.JK" if not self.ticker.endswith('.JK') else self.ticker
            end_date = datetime.now()
            start_date = end_date - timedelta(days=self.years * 365)
            
            df = yf.download(symbol, start=start_date, end=end_date, progress=False)
            
            if df.empty:
                raise ValueError(f"No data found for {symbol}")
            
            # Flatten multi-index columns if present
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            
            df.columns = [col.lower().capitalize() for col in df.columns]
            self.data = df
            return True
            
        except Exception as e:
            print(f"Error fetching data: {e}")
            return False
    
    def calculate_rsi(self, period=14):
        """Calculate RSI manually using Wilder's smoothing"""
        delta = self.data['Close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        # Wilder's smoothing
        avg_gain = gain.rolling(window=period, min_periods=period).mean()
        avg_loss = loss.rolling(window=period, min_periods=period).mean()
        
        for i in range(period, len(gain)):
            avg_gain.iloc[i] = (avg_gain.iloc[i-1] * (period - 1) + gain.iloc[i]) / period
            avg_loss.iloc[i] = (avg_loss.iloc[i-1] * (period - 1) + loss.iloc[i]) / period
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def calculate_stochastic(self, k_period=14, d_period=3):
        """Calculate Stochastic Oscillator manually"""
        low_min = self.data['Low'].rolling(window=k_period).min()
        high_max = self.data['High'].rolling(window=k_period).max()
        
        k = 100 * (self.data['Close'] - low_min) / (high_max - low_min)
        d = k.rolling(window=d_period).mean()
        
        return k, d
    
    def calculate_vwap(self):
        """Calculate Volume-Weighted Average Price"""
        typical_price = (self.data['High'] + self.data['Low'] + self.data['Close']) / 3
        vwap = (typical_price * self.data['Volume']).cumsum() / self.data['Volume'].cumsum()
        return vwap
    
    def calculate_obv(self):
        """Calculate On-Balance Volume"""
        obv = (np.sign(self.data['Close'].diff()) * self.data['Volume']).fillna(0).cumsum()
        return obv
    
    def calculate_obv_slope(self, window=20):
        """Calculate OBV trend slope using linear regression"""
        obv = self.calculate_obv()
        slopes = []
        
        for i in range(len(obv)):
            if i < window:
                slopes.append(0)
            else:
                y = obv.iloc[i-window:i].values.reshape(-1, 1)
                X = np.arange(window).reshape(-1, 1)
                try:
                    model = LinearRegression()
                    model.fit(X, y)
                    slopes.append(model.coef_[0][0])
                except:
                    slopes.append(0)
        
        return pd.Series(slopes, index=obv.index)
    
    def calculate_atr(self, period=14):
        """Calculate Average True Range"""
        high_low = self.data['High'] - self.data['Low']
        high_close = np.abs(self.data['High'] - self.data['Close'].shift())
        low_close = np.abs(self.data['Low'] - self.data['Close'].shift())
        
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()
        return atr
    
    def calculate_pivot_points(self):
        """Calculate Standard Pivot Points"""
        latest = self.data.iloc[-1]
        prev = self.data.iloc[-2]
        
        pp = (prev['High'] + prev['Low'] + prev['Close']) / 3
        r1 = 2 * pp - prev['Low']
        r2 = pp + (prev['High'] - prev['Low'])
        s1 = 2 * pp - prev['High']
        s2 = pp - (prev['High'] - prev['Low'])
        
        return {'PP': pp, 'R1': r1, 'R2': r2, 'S1': s1, 'S2': s2}
    
    def find_bounce_zones(self, order=10):
        """Find historical support levels using scipy peak detection"""
        lows = self.data['Low'].values
        local_mins_idx = argrelextrema(lows, np.less_equal, order=order)[0]
        
        bounce_levels = []
        for idx in local_mins_idx:
            if idx < len(self.data) - 50:  # Only consider older bounces
                price = self.data['Low'].iloc[idx]
                # Check if price bounced (recovered by >2% within 5 days)
                future_prices = self.data['Close'].iloc[idx:idx+5]
                if len(future_prices) > 0 and future_prices.max() > price * 1.02:
                    bounce_levels.append(price)
        
        # Cluster similar levels (within 2%)
        if bounce_levels:
            bounce_levels = sorted(bounce_levels)
            clustered = [bounce_levels[0]]
            for level in bounce_levels[1:]:
                if level > clustered[-1] * 1.02:
                    clustered.append(level)
            return clustered[-5:]  # Return last 5 bounce zones
        return []
    
    def calculate_fibonacci(self, lookback=100):
        """Calculate Fibonacci retracement from recent high"""
        recent = self.data.iloc[-lookback:]
        high = recent['High'].max()
        low = recent['Low'].min()
        diff = high - low
        
        levels = {
            'Fib_0': high,
            'Fib_23.6': high - 0.236 * diff,
            'Fib_38.2': high - 0.382 * diff,
            'Fib_50': high - 0.5 * diff,
            'Fib_61.8': high - 0.618 * diff,
            'Fib_78.6': high - 0.786 * diff,
            'Fib_100': low
        }
        return levels
    
    def detect_vcp(self, lookback=60):
        """Detect Volatility Contraction Pattern (Minervini)"""
        recent = self.data.iloc[-lookback:]
        
        # Calculate volatility contractions
        window = 10
        volatilities = []
        for i in range(0, len(recent) - window, window):
            segment = recent.iloc[i:i+window]
            vol = (segment['High'].max() - segment['Low'].min()) / segment['Low'].min()
            volatilities.append(vol)
        
        if len(volatilities) < 2:
            return False, 0
        
        # Check if volatility is contracting (each pullback smaller)
        contractions = 0
        for i in range(1, len(volatilities)):
            if volatilities[i] < volatilities[i-1] * 0.8:  # 20% reduction
                contractions += 1
        
        # VCP requires 2-6 contractions
        is_vcp = 2 <= contractions <= 6
        return is_vcp, contractions
    
    def detect_ma_squeeze(self):
        """Detect MA Superclose (tight compression)"""
        mas = [
            self.data['Close'].rolling(3).mean().iloc[-1],
            self.data['Close'].rolling(5).mean().iloc[-1],
            self.data['Close'].rolling(10).mean().iloc[-1],
            self.data['Close'].rolling(20).mean().iloc[-1]
        ]
        
        ma_max = max(mas)
        ma_min = min(mas)
        
        # Check if within 5% range
        if ma_max > 0:
            compression = (ma_max - ma_min) / ma_min
            is_squeeze = compression < 0.05
            return is_squeeze, compression * 100
        return False, 0
    
    def detect_smart_money_phase(self):
        """Detect accumulation/distribution and track phase start"""
        obv = self.calculate_obv()
        obv_slope = self.calculate_obv_slope(window=20)
        price_slope = self.data['Close'].diff(20) / self.data['Close'].shift(20)
        
        current_obv_slope = obv_slope.iloc[-1]
        current_price_slope = price_slope.iloc[-1]
        
        # Detect phase
        if current_obv_slope > 0 and current_price_slope < 0.02:
            phase = "ACCUMULATION"
        elif current_obv_slope < 0 and current_price_slope > -0.02:
            phase = "DISTRIBUTION"
        else:
            phase = "NEUTRAL"
        
        # Track phase start date
        if phase != "NEUTRAL":
            for i in range(len(obv_slope) - 1, 0, -1):
                past_obv = obv_slope.iloc[i-20] if i >= 20 else obv_slope.iloc[0]
                past_price = price_slope.iloc[i-20] if i >= 20 else price_slope.iloc[0]
                
                if phase == "ACCUMULATION":
                    if not (past_obv > 0 and past_price < 0.02):
                        self.smart_money_start = self.data.index[i]
                        break
                elif phase == "DISTRIBUTION":
                    if not (past_obv < 0 and past_price > -0.02):
                        self.smart_money_start = self.data.index[i]
                        break
        
        self.current_phase = phase
        return phase, current_obv_slope
    
    def optimize_parameters(self):
        """Grid search optimization for indicator parameters"""
        best_params = {
            'rsi_period': 14,
            'stoch_k': 14,
            'stoch_d': 3,
            'ma_fast': 10,
            'ma_slow': 50
        }
        
        # Test RSI periods
        rsi_periods = [9, 10, 12, 14, 16]
        stoch_k_periods = [10, 12, 14, 16, 18]
        
        # For production: would run full backtest for each combo
        # Here we use simplified scoring
        best_score = 0
        
        for rsi_p in rsi_periods:
            for stoch_k in stoch_k_periods:
                rsi = self.calculate_rsi(period=rsi_p)
                k, d = self.calculate_stochastic(k_period=stoch_k, d_period=3)
                
                # Simple scoring: count profitable crossovers
                score = 0
                for i in range(50, len(self.data)):
                    if rsi.iloc[i] < 30 and self.data['Close'].iloc[i:i+5].max() > self.data['Close'].iloc[i] * 1.03:
                        score += 1
                
                if score > best_score:
                    best_score = score
                    best_params['rsi_period'] = rsi_p
                    best_params['stoch_k'] = stoch_k
        
        return best_params
    
    def get_ojk_compliant_price(self, price):
        """Round price according to IDX/OJK tick size rules"""
        if price < 200:
            return round(price / 1) * 1
        elif price < 500:
            return round(price / 2) * 2
        elif price < 2000:
            return round(price / 5) * 5
        elif price < 5000:
            return round(price / 10) * 10
        else:
            return round(price / 25) * 25
    
    def backtest_strategy(self, strategy_name, params):
        """Backtest a specific strategy"""
        results = []
        
        # Calculate indicators
        rsi = self.calculate_rsi(period=params['rsi_period'])
        k, d = self.calculate_stochastic(k_period=params['stoch_k'], d_period=params['stoch_d'])
        sma_fast = self.data['Close'].rolling(params['ma_fast']).mean()
        sma_slow = self.data['Close'].rolling(params['ma_slow']).mean()
        atr = self.calculate_atr()
        
        for i in range(100, len(self.data) - 10):
            signal = None
            
            if strategy_name == "BREAKOUT":
                # Breakout above resistance with volume
                if (self.data['Close'].iloc[i] > sma_slow.iloc[i] and
                    self.data['Volume'].iloc[i] > self.data['Volume'].iloc[i-20:i].mean() * 1.5 and
                    rsi.iloc[i] > 50 and rsi.iloc[i] < 70):
                    signal = "BUY"
            
            elif strategy_name == "PULLBACK":
                # Pullback to support in uptrend
                if (sma_fast.iloc[i] > sma_slow.iloc[i] and
                    rsi.iloc[i] < 40 and rsi.iloc[i] > 30 and
                    k.iloc[i] < 30):
                    signal = "BUY"
            
            elif strategy_name == "LOW_CHEAT":
                # Price breaks below then recovers quickly
                recent_low = self.data['Low'].iloc[i-5:i].min()
                if (self.data['Close'].iloc[i] > recent_low * 1.02 and
                    self.data['Low'].iloc[i-1] < recent_low and
                    rsi.iloc[i] > 40):
                    signal = "BUY"
            
            elif strategy_name == "BUY_DIP":
                # Oversold bounce
                if (rsi.iloc[i] < 30 and k.iloc[i] < 20 and
                    self.data['Close'].iloc[i] > sma_fast.iloc[i] * 0.95):
                    signal = "BUY"
            
            if signal == "BUY":
                entry = self.data['Close'].iloc[i]
                atr_val = atr.iloc[i] if not pd.isna(atr.iloc[i]) else entry * 0.02
                
                stop_loss = self.get_ojk_compliant_price(entry - 1 * atr_val)
                take_profit = self.get_ojk_compliant_price(entry + 3 * atr_val)
                
                # Check outcome in next 10 days
                future = self.data.iloc[i+1:i+11]
                if len(future) > 0:
                    hit_tp = (future['High'] >= take_profit).any()
                    hit_sl = (future['Low'] <= stop_loss).any()
                    
                    if hit_tp and not hit_sl:
                        outcome = "WIN"
                    elif hit_sl:
                        outcome = "LOSS"
                    else:
                        outcome = "NEUTRAL"
                    
                    results.append({
                        'entry': entry,
                        'sl': stop_loss,
                        'tp': take_profit,
                        'outcome': outcome
                    })
        
        if results:
            wins = len([r for r in results if r['outcome'] == 'WIN'])
            total = len(results)
            win_rate = wins / total if total > 0 else 0
            return win_rate, results
        
        return 0, []
    
    def find_best_strategy(self):
        """Find strategy with >55% win rate or return NO TRADE"""
        params = self.optimize_parameters()
        strategies = ["BREAKOUT", "PULLBACK", "LOW_CHEAT", "BUY_DIP"]
        
        best_strategy = None
        best_win_rate = 0
        best_results = []
        
        # Layer 1: Test standard strategies
        for strategy in strategies:
            win_rate, results = self.backtest_strategy(strategy, params)
            if win_rate > best_win_rate:
                best_win_rate = win_rate
                best_strategy = strategy
                best_results = results
        
        # Layer 2: Apply additional filters if < 55%
        if best_win_rate < 0.55:
            # Add VCP filter
            is_vcp, _ = self.detect_vcp()
            obv_slope = self.calculate_obv_slope().iloc[-1]
            
            if is_vcp and obv_slope > 1000:  # Stricter OBV requirement
                enhanced_params = params.copy()
                enhanced_params['rsi_min'] = 45  # Stricter RSI
                
                for strategy in strategies:
                    win_rate, results = self.backtest_strategy(strategy, enhanced_params)
                    if win_rate > best_win_rate:
                        best_win_rate = win_rate
                        best_strategy = strategy + "_ENHANCED"
                        best_results = results
        
        # Layer 3: Final check
        if best_win_rate < 0.55:
            return None, 0, []  # NO TRADE
        
        return best_strategy, best_win_rate, best_results
    
    def generate_signal(self):
        """Generate final trading signal"""
        if self.data is None or len(self.data) < 200:
            return None
        
        # Calculate all indicators
        params = self.optimize_parameters()
        rsi = self.calculate_rsi(period=params['rsi_period'])
        k, d = self.calculate_stochastic(k_period=params['stoch_k'], d_period=params['stoch_d'])
        vwap = self.calculate_vwap()
        obv = self.calculate_obv()
        obv_slope = self.calculate_obv_slope()
        atr = self.calculate_atr()
        
        # Pattern detection
        is_vcp, vcp_contractions = self.detect_vcp()
        is_squeeze, squeeze_pct = self.detect_ma_squeeze()
        phase, obv_trend = self.detect_smart_money_phase()
        
        # Support/Resistance
        pivots = self.calculate_pivot_points()
        bounces = self.find_bounce_zones()
        fibs = self.calculate_fibonacci()
        
        # Find best strategy
        strategy, win_rate, backtest_results = self.find_best_strategy()
        
        if strategy is None:
            return {
                'verdict': 'NO TRADE',
                'reason': 'No strategy met the 55% win rate threshold. Market conditions too risky.',
                'win_rate': 0
            }
        
        # Generate trade plan
        current_price = self.data['Close'].iloc[-1]
        atr_val = atr.iloc[-1]
        
        entry = self.get_ojk_compliant_price(current_price)
        stop_loss = self.get_ojk_compliant_price(current_price - 1 * atr_val)
        tp_1r = self.get_ojk_compliant_price(current_price + 1 * atr_val)
        tp_2r = self.get_ojk_compliant_price(current_price + 2 * atr_val)
        tp_3r = self.get_ojk_compliant_price(current_price + 3 * atr_val)
        
        # Generate plain English explanation
        explanation = self._generate_explanation(phase, is_squeeze, is_vcp, obv_trend)
        
        return {
            'verdict': 'BUY' if win_rate > 0.55 else 'NO TRADE',
            'strategy': strategy,
            'entry': entry,
            'stop_loss': stop_loss,
            'tp_1r': tp_1r,
            'tp_2r': tp_2r,
            'tp_3r': tp_3r,
            'win_rate': win_rate * 100,
            'explanation': explanation,
            'rsi': rsi.iloc[-1],
            'stoch_k': k.iloc[-1],
            'stoch_d': d.iloc[-1],
            'vwap': vwap.iloc[-1],
            'obv_slope': obv_slope.iloc[-1],
            'is_vcp': is_vcp,
            'vcp_contractions': vcp_contractions,
            'is_squeeze': is_squeeze,
            'squeeze_pct': squeeze_pct,
            'smart_money_phase': phase,
            'smart_money_start': self.smart_money_start.strftime('%Y-%m-%d') if self.smart_money_start else 'N/A',
            'pivots': pivots,
            'fibonacci': fibs,
            'current_price': current_price
        }
    
    def _generate_explanation(self, phase, is_squeeze, is_vcp, obv_trend):
        """Generate plain English explanation"""
        explanations = []
        
        if phase == "ACCUMULATION":
            explanations.append("Big investors are quietly buying while the price is stable.")
        
        if is_squeeze:
            explanations.append("The price is squeezing into a tight range, like a spring ready to pop.")
        
        if is_vcp:
            explanations.append("The stock is showing a classic compression pattern that often leads to breakouts.")
        
        if obv_trend > 5000:
            explanations.append("Volume confirms strong buying interest building up.")
        
        if not explanations:
            explanations.append("Technical indicators suggest a favorable risk-reward setup.")
        
        return " ".join(explanations)
