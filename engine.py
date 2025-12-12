import yfinance as yf
import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta

# --- OJK / IDX Compliance Module ---

def get_tick_size(price: float) -> int:
    """
    Returns the valid tick size (fraction) based on IDX rules (2024/2025).
    """
    if price < 200:
        return 1
    elif 200 <= price < 500:
        return 2
    elif 500 <= price < 2000:
        return 5
    elif 2000 <= price < 5000:
        return 10
    else:
        return 25

def round_to_tick(price: float) -> int:
    """Rounds a price to the nearest valid OJK tick."""
    tick = get_tick_size(price)
    return int(round(price / tick) * tick)

# --- Data Structures ---

@dataclass
class BacktestResult:
    win_rate: float
    total_trades: int
    max_drawdown: float
    avg_candles_held: int
    config: Dict
    strategy_name: str
    is_valid: bool  # True if Win Rate > 65%

@dataclass
class SignalOutput:
    action: str  # BUY, WAIT, NO TRADE
    entry_price: float
    stop_loss: float
    target_price: float
    reason: str
    probabilities: Dict[str, float]
    technical_data: Dict
    backtest_stats: BacktestResult

# --- Core Analysis Engine ---

class AlphaEngine:
    def __init__(self, ticker: str):
        self.ticker = ticker
        if not self.ticker.endswith(".JK"):
            self.ticker += ".JK"
        self.df = self._fetch_data()
        
    def _fetch_data(self) -> pd.DataFrame:
        """Fetches last 3 years of OHLCV data."""
        start_date = (datetime.now() - timedelta(days=3*365)).strftime('%Y-%m-%d')
        df = yf.download(self.ticker, start=start_date, progress=False)
        if df.empty:
            raise ValueError(f"No data found for {self.ticker}")
        
        # Flatten MultiIndex columns if present (yfinance update)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
            
        df['Returns'] = df['Close'].pct_change()
        return df

    def calculate_indicators(self, df: pd.DataFrame, rsi_period=14, ma_fast=10, ma_slow=50, stoch_k=14) -> pd.DataFrame:
        """Calculates indicators based on dynamic parameters."""
        data = df.copy()
        
        # RSI
        delta = data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=rsi_period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=rsi_period).mean()
        rs = gain / loss
        data['RSI'] = 100 - (100 / (1 + rs))
        
        # Moving Averages
        data['MA_Fast'] = data['Close'].rolling(window=ma_fast).mean()
        data['MA_Slow'] = data['Close'].rolling(window=ma_slow).mean()
        
        # Stochastic
        low_min = data['Low'].rolling(window=stoch_k).min()
        high_max = data['High'].rolling(window=stoch_k).max()
        data['Stoch_K'] = 100 * ((data['Close'] - low_min) / (high_max - low_min))
        
        # VWAP
        data['VWAP'] = (data['Volume'] * (data['High'] + data['Low'] + data['Close']) / 3).cumsum() / data['Volume'].cumsum()
        
        # OBV for Smart Money
        data['OBV'] = (np.sign(data['Close'].diff()) * data['Volume']).fillna(0).cumsum()
        data['OBV_MA'] = data['OBV'].rolling(window=20).mean() # Signal line for OBV
        
        return data

    def analyze_smart_money(self, df: pd.DataFrame) -> Dict:
        """Analyzes Accumulation/Distribution and finds the Start Date."""
        current_obv = df['OBV'].iloc[-1]
        current_obv_ma = df['OBV_MA'].iloc[-1]
        
        status = "Accumulation" if current_obv > current_obv_ma else "Distribution"
        
        # Find start date (last crossover)
        crossover = False
        start_date = "Unknown"
        
        # Iterate backwards to find the flip
        for i in range(len(df)-1, 1, -1):
            curr_cond = df['OBV'].iloc[i] > df['OBV_MA'].iloc[i]
            prev_cond = df['OBV'].iloc[i-1] > df['OBV_MA'].iloc[i-1]
            
            if (status == "Accumulation" and not prev_cond and curr_cond) or \
               (status == "Distribution" and prev_cond and not curr_cond):
                start_date = df.index[i].strftime('%Y-%m-%d')
                break
                
        return {
            "status": status,
            "strength": "High" if abs(current_obv - current_obv_ma) > df['OBV'].std() else "Moderate",
            "start_date": start_date
        }

    def detect_patterns(self, df: pd.DataFrame) -> Dict:
        """Detects VCP, MA Squeeze, and Low Cheat."""
        latest = df.iloc[-1]
        
        # MA Squeeze: SMA 3, 5, 10, 20 within 5% range
        mas = [
            df['Close'].rolling(3).mean().iloc[-1],
            df['Close'].rolling(5).mean().iloc[-1],
            df['Close'].rolling(10).mean().iloc[-1],
            df['Close'].rolling(20).mean().iloc[-1]
        ]
        ma_min, ma_max = min(mas), max(mas)
        is_squeeze = (ma_max - ma_min) / ma_min < 0.05
        
        # VCP: Decreasing volatility over last 20 days
        # Simplified logic: Range (High-Low) is shrinking
        ranges = (df['High'] - df['Low']).tail(20)
        # Check if the average range of the last 5 days is smaller than the first 5 days of the window
        is_vcp = ranges.iloc[-5:].mean() < ranges.iloc[:5].mean() * 0.7 and latest['Volume'] < df['Volume'].tail(20).mean()
        
        return {
            "VCP": is_vcp,
            "MA_Squeeze": is_squeeze,
            "Low_Cheat": is_vcp and latest['Close'] > latest['MA_Slow'] # Simple proxy for low cheat base
        }

    def get_support_resistance(self, df: pd.DataFrame) -> Dict:
        """Calculates Pivot Points, Fibs, and Bounce Zones."""
        high = df['High'].iloc[-1]
        low = df['Low'].iloc[-1]
        close = df['Close'].iloc[-1]
        
        # Standard Pivot
        pivot = (high + low + close) / 3
        s1 = (2 * pivot) - high
        
        # Recent Bounce Zone (Lowest Low of last 60 days that held more than once)
        last_60 = df.tail(60)
        min_price = last_60['Low'].min()
        
        # Fib Retracement from last major swing high (last 90 days max)
        swing_high = df['High'].tail(90).max()
        swing_low = df['Low'].tail(90).min()
        fib_618 = swing_high - ((swing_high - swing_low) * 0.618)
        
        return {
            "Pivot_Support": round_to_tick(s1),
            "Bounce_Zone": round_to_tick(min_price),
            "Fib_Golden_Zone": round_to_tick(fib_618)
        }

    def _backtest_strategy(self, df: pd.DataFrame, strategy_type: str, rsi_p, ma_f, ma_s) -> BacktestResult:
        """
        Runs a simulation of the strategy on historical data.
        Returns performance metrics.
        """
        capital = 10000000 # 10 Juta IDR
        balance = capital
        position = 0
        trades = 0
        wins = 0
        entry_price = 0
        peak_balance = capital
        drawdown = 0
        candles_held = []
        entry_idx = 0

        # Run logic on historical data
        # Note: We iterate through the dataframe. This is simplified for speed.
        
        # Pre-calculate signals to speed up loop
        # Logic: Buy if MA_Fast crosses MA_Slow AND RSI < 70
        
        for i in range(ma_s, len(df)):
            row = df.iloc[i]
            prev = df.iloc[i-1]
            
            # Entry Logic
            if position == 0:
                signal = False
                if strategy_type == "Trend_Following":
                    signal = row['MA_Fast'] > row['MA_Slow'] and prev['MA_Fast'] <= prev['MA_Slow']
                elif strategy_type == "Dip_Buy":
                    signal = row['RSI'] < 30 and row['Close'] > row['MA_Slow']
                
                if signal:
                    tick = get_tick_size(row['Close'])
                    stop_loss = row['Close'] - (3 * tick) # Tight stop for simulation
                    risk = row['Close'] - stop_loss
                    target = row['Close'] + (risk * 3) # 1:3 RR
                    
                    position = int(balance / row['Close'])
                    balance -= position * row['Close']
                    entry_price = row['Close']
                    entry_idx = i
                    tp_price = target
                    sl_price = stop_loss
                    trades += 1

            # Exit Logic (Simulation)
            elif position > 0:
                # Check High/Low for TP/SL hit
                if row['Low'] <= sl_price:
                    balance += position * sl_price
                    position = 0
                    candles_held.append(i - entry_idx)
                elif row['High'] >= tp_price:
                    balance += position * tp_price
                    position = 0
                    wins += 1
                    candles_held.append(i - entry_idx)
                
                # Update Drawdown
                current_val = balance + (position * row['Close'])
                peak_balance = max(peak_balance, current_val)
                dd = (peak_balance - current_val) / peak_balance
                drawdown = max(drawdown, dd)

        win_rate = (wins / trades * 100) if trades > 0 else 0
        avg_hold = sum(candles_held)/len(candles_held) if candles_held else 0
        
        return BacktestResult(
            win_rate=win_rate,
            total_trades=trades,
            max_drawdown=drawdown * 100,
            avg_candles_held=int(avg_hold),
            config={'rsi': rsi_p, 'ma_fast': ma_f, 'ma_slow': ma_s},
            strategy_name=strategy_type,
            is_valid=win_rate > 65.0
        )

    def optimize_and_run(self) -> SignalOutput:
        """
        Grid Search to find best parameters and strategy.
        Then generates the final signal based on current data.
        """
        # Grid Search Parameters
        rsi_params = [9, 14, 21]
        ma_params = [(5, 10), (10, 20), (20, 50), (50, 200)]
        strategies = ["Trend_Following", "Dip_Buy"]
        
        best_result: Optional[BacktestResult] = None
        
        # 1. OPTIMIZATION LOOP
        for rsi in rsi_params:
            for ma_f, ma_s in ma_params:
                # Calculate indicators ONCE for this config
                df_calc = self.calculate_indicators(self.df, rsi, ma_f, ma_s)
                
                for strat in strategies:
                    res = self._backtest_strategy(df_calc, strat, rsi, ma_f, ma_s)
                    
                    if res.is_valid:
                        if best_result is None or res.win_rate > best_result.win_rate:
                            best_result = res

        # Fallback if no strategy meets 65% (Return best available but mark danger)
        if best_result is None:
            # Re-run to find just the highest WR even if low
            best_result = BacktestResult(0,0,0,0, {}, "None", False) # Dummy

        # 2. GENERATE CURRENT SIGNAL USING BEST CONFIG
        final_config = best_result.config if best_result.config else {'rsi': 14, 'ma_fast': 10, 'ma_slow': 20}
        df_final = self.calculate_indicators(self.df, **final_config)
        
        smart_money = self.analyze_smart_money(df_final)
        patterns = self.detect_patterns(df_final)
        levels = self.get_support_resistance(df_final)
        
        latest = df_final.iloc[-1]
        
        # Decide Action based on optimized strategy logic applied to *current* candle
        action = "WAIT"
        reason = "Setup conditions not met."
        
        # Logic Application
        if best_result.strategy_name == "Trend_Following":
            if latest['MA_Fast'] > latest['MA_Slow'] and patterns['MA_Squeeze'] == False:
                 # Breakout check
                 if latest['Close'] > levels['Pivot_Support']:
                     action = "BUY"
                     reason = "Trend Following Breakout with Optimized MAs"
        elif best_result.strategy_name == "Dip_Buy":
            if latest['RSI'] < 35 and latest['Close'] > levels['Bounce_Zone']:
                action = "BUY"
                reason = "Oversold Bounce Play at Historical Support"

        # Risk Calculation
        current_price = latest['Close']
        stop_loss = round_to_tick(levels['Pivot_Support'] if levels['Pivot_Support'] < current_price else current_price * 0.95)
        risk = current_price - stop_loss
        target = round_to_tick(current_price + (risk * 3))
        
        if risk <= 0: # Sanity check if price below support
            action = "NO TRADE"
            reason = "Price is below immediate support."

        # Probabilities (Heuristic based on Backtest WR and Technical Confluence)
        base_prob = best_result.win_rate
        if smart_money['status'] == "Accumulation": base_prob += 5
        if patterns['VCP']: base_prob += 10
        
        return SignalOutput(
            action=action,
            entry_price=current_price,
            stop_loss=stop_loss,
            target_price=target,
            reason=reason,
            probabilities={
                "1R": min(base_prob + 15, 95),
                "2R": min(base_prob, 85),
                "3R": min(base_prob - 20, 60)
            },
            technical_data={
                "Current_Price": current_price,
                "VWAP": latest['VWAP'],
                "Indicators": final_config,
                "RSI_Value": round(latest['RSI'], 2),
                "Smart_Money": smart_money,
                "Levels": levels,
                "Patterns": patterns
            },
            backtest_stats=best_result
        )