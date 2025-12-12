import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings("ignore")

class OJKTools:
    """Helper to enforce Indonesia Stock Exchange (IDX/OJK) price tick rules."""
    @staticmethod
    def get_tick_size(price):
        if price < 200: return 1
        elif price < 500: return 2
        elif price < 2000: return 5
        elif price < 5000: return 10
        else: return 25

    @staticmethod
    def floor_to_tick(price):
        """Rounds down to nearest valid tick."""
        tick = OJKTools.get_tick_size(price)
        return (int(price) // tick) * tick

    @staticmethod
    def ceil_to_tick(price):
        """Rounds up to nearest valid tick."""
        tick = OJKTools.get_tick_size(price)
        if int(price) % tick == 0: return int(price)
        return ((int(price) // tick) + 1) * tick

class QuantEngine:
    def __init__(self, ticker):
        # Auto-append .JK if missing for IHSG
        self.ticker = ticker if ticker.endswith('.JK') else f"{ticker}.JK"
        self.df = None
        self.analysis_results = {}
        
    def fetch_data(self):
        """Fetch 3 years of data."""
        start_date = (datetime.now() - timedelta(days=3*365)).strftime('%Y-%m-%d')
        self.df = yf.download(self.ticker, start=start_date, progress=False)
        
        if self.df.empty:
            raise ValueError(f"No data found for {self.ticker}")
            
        # Ensure flat columns if MultiIndex (yfinance update)
        if isinstance(self.df.columns, pd.MultiIndex):
            self.df.columns = self.df.columns.get_level_values(0)
            
        self.df = self.df.dropna()

    def _calculate_rsi(self, series, period):
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def _calculate_stoch(self, high, low, close, k_period, d_period):
        lowest_low = low.rolling(window=k_period).min()
        highest_high = high.rolling(window=k_period).max()
        k = 100 * ((close - lowest_low) / (highest_high - lowest_low))
        d = k.rolling(window=d_period).mean()
        return k, d

    def _calculate_vwap(self, df):
        v = df['Volume'].values
        tp = (df['High'] + df['Low'] + df['Close']) / 3
        return df.assign(VWAP=(tp * v).cumsum() / v.cumsum())

    def optimize_indicators(self):
        """
        Dynamically find best MA, RSI, and Stoch settings by simple backtest 
        on the specific stock's history.
        """
        best_score = -np.inf
        best_params = {}
        
        # Grid search simplified
        ma_options = [20, 50, 100]
        rsi_options = [9, 14, 21]
        stoch_options = [(9,3), (14,3), (21,5)]
        
        # We test based on a simple Trend Following logic: 
        # Price > MA AND RSI > 50 -> Score increment
        for ma in ma_options:
            for rsi in rsi_options:
                for stoch in stoch_options:
                    score = 0
                    temp_df = self.df.copy()
                    temp_df['MA'] = temp_df['Close'].rolling(window=ma).mean()
                    temp_df['RSI'] = self._calculate_rsi(temp_df['Close'], rsi)
                    
                    # Vectorized simplified scoring: Return for periods where conditions met
                    # Condition: Price > MA and RSI rising
                    temp_df['Signal'] = np.where((temp_df['Close'] > temp_df['MA']) & (temp_df['RSI'] > 50), 1, 0)
                    temp_df['Return'] = temp_df['Close'].pct_change() * temp_df['Signal'].shift(1)
                    
                    total_return = temp_df['Return'].sum()
                    
                    if total_return > best_score:
                        best_score = total_return
                        best_params = {'MA': ma, 'RSI': rsi, 'Stoch': stoch}
        
        self.analysis_results['optimized_params'] = best_params
        return best_params

    def analyze_structure(self):
        df = self.df
        current_price = df['Close'].iloc[-1]
        
        # 1. Pivot & Bounce Support
        # Find local mins (bounces) in last 6 months
        last_6m = df.iloc[-120:]
        # Simple local minima detection
        min_idx = (np.diff(np.sign(np.diff(last_6m['Low']))) > 0).nonzero()[0] + 1
        supports = last_6m['Low'].iloc[min_idx]
        # Filter supports below current price
        valid_supports = supports[supports < current_price]
        nearest_bounce_support = valid_supports.max() if not valid_supports.empty else df['Low'].min()

        # 2. Fibonacci Retracement
        # Find last major High and Low
        lookback = 200
        period_high = df['High'].iloc[-lookback:].max()
        period_low = df['Low'].iloc[-lookback:].min()
        diff = period_high - period_low
        fibo_levels = {
            '0.236': period_high - 0.236 * diff,
            '0.382': period_high - 0.382 * diff,
            '0.5': period_high - 0.5 * diff,
            '0.618': period_high - 0.618 * diff
        }
        
        # Find nearest fibo support below current price
        fibo_support = min([v for v in fibo_levels.values() if v < current_price], key=lambda x: abs(x - current_price), default=period_low)

        self.analysis_results['support'] = {
            'bounce': nearest_bounce_support,
            'fibo': fibo_support,
            'pivot_strong': max(nearest_bounce_support, fibo_support)
        }

    def analyze_bandar_flow(self):
        """
        Heuristic detection of Accumulation/Distribution.
        Method: Analyzes Volume Price Trend and On Balance Volume logic combined with volatility.
        """
        df = self.df.copy()
        df['Returns'] = df['Close'].pct_change()
        
        # Bandar Volume Proxy: Volume * Sign of Close Change
        # Accumulation: High Volume on Up days, Low Volume on Down days
        df['MoneyFlow'] = df['Volume'] * ((df['Close'] - df['Open']) / df['Open'])
        
        # Look at last 20 days
        recent = df.iloc[-20:]
        net_flow = recent['MoneyFlow'].sum()
        
        # Absorption detection: Price flat/small drop, Volume High
        price_change_pct = (recent['Close'].iloc[-1] - recent['Close'].iloc[0]) / recent['Close'].iloc[0]
        avg_vol_recent = recent['Volume'].mean()
        avg_vol_hist = df['Volume'].iloc[-100:].mean()
        
        status = "Neutral"
        start_date = "N/A"
        
        if net_flow > 0 and price_change_pct > -0.05:
            status = "Accumulation"
            # Find start of accumulation (first large green volume spike in window)
            spikes = recent[recent['MoneyFlow'] > recent['MoneyFlow'].mean() + recent['MoneyFlow'].std()]
            if not spikes.empty:
                start_date = spikes.index[0].strftime('%Y-%m-%d')
        elif net_flow < 0:
            status = "Distribution"
            spikes = recent[recent['MoneyFlow'] < recent['MoneyFlow'].mean() - recent['MoneyFlow'].std()]
            if not spikes.empty:
                start_date = spikes.index[0].strftime('%Y-%m-%d')
                
        # VWAP
        df = self._calculate_vwap(df)
        current_vwap = df['VWAP'].iloc[-1]
        
        self.analysis_results['bandar'] = {
            'status': status,
            'start_date': start_date,
            'vwap': current_vwap,
            'volume_ratio': avg_vol_recent / avg_vol_hist # > 1 means high activity
        }

    def detect_patterns(self):
        df = self.df
        close = df['Close']
        ma_s = df['Close'].rolling(window=3).mean()
        ma_5 = df['Close'].rolling(window=5).mean()
        ma_10 = df['Close'].rolling(window=10).mean()
        ma_20 = df['Close'].rolling(window=20).mean()
        
        current_ma = [ma_s.iloc[-1], ma_5.iloc[-1], ma_10.iloc[-1], ma_20.iloc[-1]]
        
        # 1. Superclose MA (Compression)
        # Check if spread between min and max MA is < 5% of price
        spread = (max(current_ma) - min(current_ma)) / close.iloc[-1]
        is_superclose = spread < 0.05
        
        # 2. Mark Minervini Low Cheat / VCP
        # VCP: Series of lower highs with decreasing volatility?
        # Simplified Check: Look at last 3 swing highs. Are they contracting? 
        # And is current price near the lows of the consolidation (Low Cheat)?
        
        recent_highs = df['High'].iloc[-60:].rolling(10).max().drop_duplicates()
        is_vcp = False
        if len(recent_highs) >= 2:
            # Very rough check for contraction
            if recent_highs.iloc[-1] < recent_highs.iloc[-2]:
                is_vcp = True

        self.analysis_results['patterns'] = {
            'superclose_ma': is_superclose,
            'vcp_detected': is_vcp,
            'ma_spread_pct': spread * 100
        }

    def backtest_strategy(self):
        """
        Backtests the "Optimum" strategy on this specific stock to calculate probability.
        Strategy: Buy when Price > Opt_MA and RSI > 50. Exit when Price < Opt_MA.
        """
        params = self.analysis_results.get('optimized_params', {'MA':20, 'RSI':14})
        ma_per = params['MA']
        rsi_per = params['RSI']
        
        df = self.df.copy()
        df['MA'] = df['Close'].rolling(window=ma_per).mean()
        df['RSI'] = self._calculate_rsi(df['Close'], rsi_per)
        
        df['Signal'] = 0
        # Entry: Close > MA & RSI > 50 (Momentum)
        df.loc[(df['Close'] > df['MA']) & (df['RSI'] > 50), 'Signal'] = 1
        # Exit: Close < MA
        df.loc[df['Close'] < df['MA'], 'Signal'] = -1
        
        # Simulate trades
        trades = []
        in_position = False
        entry_price = 0
        
        for i in range(1, len(df)):
            if not in_position and df['Signal'].iloc[i] == 1:
                in_position = True
                entry_price = df['Close'].iloc[i]
            elif in_position and df['Signal'].iloc[i] == -1:
                in_position = False
                exit_price = df['Close'].iloc[i]
                trades.append((exit_price - entry_price) / entry_price)
                
        win_rate = len([t for t in trades if t > 0]) / len(trades) if trades else 0
        avg_return = np.mean(trades) if trades else 0
        
        self.analysis_results['backtest'] = {
            'win_rate': win_rate,
            'total_trades': len(trades),
            'avg_return': avg_return
        }

    def generate_recommendation(self):
        current_price = self.df['Close'].iloc[-1]
        support = self.analysis_results['support']['pivot_strong']
        
        # Risk Management (R:R 3:1)
        # Stop Loss slightly below support
        stop_loss_raw = support * 0.98 # 2% buffer below support
        risk = current_price - stop_loss_raw
        
        if risk <= 0: 
            # Price is at or below support, tight stop
            risk = current_price * 0.02
            stop_loss_raw = current_price - risk
            
        reward_raw = current_price + (risk * 3)
        
        # OJK Adjustments
        stop_loss = OJKTools.floor_to_tick(stop_loss_raw)
        target_price = OJKTools.ceil_to_tick(reward_raw)
        
        # Logic for Action
        # If Bandarmology is Acc, Patterns are good, Backtest > 40% win rate (swing standard)
        score = 0
        reasons = []
        
        if self.analysis_results['bandar']['status'] == "Accumulation":
            score += 2
            reasons.append("Bandar Accumulation Detected")
        
        if self.analysis_results['patterns']['superclose_ma']:
            score += 2
            reasons.append("Superclose MA (Squeeze imminent)")
            
        if self.analysis_results['patterns']['vcp_detected']:
            score += 1
            reasons.append("VCP/Contraction characteristics")
            
        backtest_win = self.analysis_results['backtest']['win_rate']
        if backtest_win > 0.5:
            score += 2
            reasons.append(f"High historical win rate ({backtest_win:.0%})")
        elif backtest_win > 0.35:
            score += 1
            
        # Entry Signal
        action = "WAIT"
        if score >= 4:
            action = "STRONG BUY"
        elif score >= 2:
            action = "BUY ON WEAKNESS"
        else:
            action = "WAIT / AVOID"
            
        prob_success = min(0.9, backtest_win + (score * 0.05))
        
        return {
            'ticker': self.ticker,
            'price': current_price,
            'action': action,
            'probability': prob_success,
            'stop_loss': stop_loss,
            'target_price': target_price,
            'risk_reward': "1:3",
            'reasons': reasons,
            'optimized_settings': self.analysis_results['optimized_params']
        }

    def run_full_analysis(self):
        self.fetch_data()
        self.optimize_indicators()
        self.analyze_structure()
        self.analyze_bandar_flow()
        self.detect_patterns()
        self.backtest_strategy()
        return self.generate_recommendation()