import yfinance as yf
import pandas as pd
import numpy as np
import time
import random
import requests
from bs4 import BeautifulSoup
from textblob import TextBlob
from datetime import datetime, timedelta
# --- XGBoost Imports ---
from xgboost import XGBClassifier, XGBRegressor
from sklearn.model_selection import train_test_split

# ==========================================
# 1. DEFAULT CONFIGURATION
# ==========================================
DEFAULT_CONFIG = {
    "BACKTEST_PERIOD": "2y",
    "MAX_HOLD_DAYS": 60,
    "FIB_LOOKBACK_DAYS": 120,
    "RSI_PERIOD": 14,
    "RSI_LOWER": 30,
    "ATR_PERIOD": 14,
    "SL_MULTIPLIER": 3.0, 
    "TP_MULTIPLIER": 6.0,
    "CMF_PERIOD": 20,
    "MFI_PERIOD": 14,
    "VOL_MA_PERIOD": 20,
    "MIN_MARKET_CAP": 500_000_000_000, 
    "MIN_DAILY_VOL": 1_000_000_000,
    "MIN_ADTV_IDR": 5_000_000_000,
    "ACCOUNT_BALANCE": 100_000_000,
    "RISK_PER_TRADE_PCT": 1.0
}

MA_TEST_PAIRS = [(5, 20), (20, 50), (50, 200)] 
STOCH_K_PERIOD = 14
STOCH_D_PERIOD = 3
STOCH_OVERSOLD = 20
OBV_LOOKBACK_DAYS = 10
TREND_EMA_DEFAULT = 200

# Financial Dictionary for Better Sentiment
FIN_BULLISH = {
    'naik', 'lonjakan', 'rekor', 'laba', 'untung', 'dividen', 'buyback', 
    'akuisisi', 'merger', 'tumbuh', 'menguat', 'bullish', 'hijau', 
    'positif', 'tertinggi', 'cuan', 'proyeksi', 'ekspansi', 'divestasi',
    'surplus', 'pendapatan naik', 'laba bersih'
}

FIN_BEARISH = {
    'turun', 'anjlok', 'rugi', 'merugi', 'bangkrut', 'pailit', 'utang', 
    'gagal', 'batal', 'melemah', 'bearish', 'merah', 'negatif', 
    'terendah', 'boncos', 'suspen', 'gugat', 'sanksi', 'denda',
    'penggelapan', 'korupsi', 'phk', 'defisit'
}

class StockAnalyzer:
    def __init__(self, ticker, user_config=None):
        self.ticker = self._format_ticker(ticker)
        self.market_ticker = "^JKSE"
        self.df = None
        self.market_df = None
        self.info = {}
        self.news_analysis = {"sentiment": "Neutral", "score": 0, "headlines": []}
        self.active_trend_col = f"EMA_{TREND_EMA_DEFAULT}"
        self.data_len = 0
        self.config = DEFAULT_CONFIG.copy()
        if user_config:
            self.config.update(user_config)

    def _format_ticker(self, ticker):
        ticker = ticker.upper().strip()
        if not ticker.endswith(".JK") and not ticker.startswith("^"):
            ticker += ".JK"
        return ticker

    def fetch_data(self):
        try:
            period = self.config["BACKTEST_PERIOD"]
            # Ensure at least 2y for valid ML/Quant metrics
            if "y" in period:
                 y = int(period.replace("y",""))
                 if y < 2: period = "2y"
            
            self.df = yf.download(self.ticker, period=period, progress=False, auto_adjust=True, multi_level_index=False)
            try:
                self.market_df = yf.download(self.market_ticker, period=period, progress=False, auto_adjust=True, multi_level_index=False)
            except: self.market_df = None

            if self.df.empty: return False
            self.data_len = len(self.df)

            ticker_obj = yf.Ticker(self.ticker)
            try:
                self.info = ticker_obj.info
                if 'longName' not in self.info: self.info['longName'] = self.ticker
            except: self.info['longName'] = self.ticker
            return True
        except Exception as e:
            print(f"Error fetching data: {e}")
            return False

    def analyze_news_sentiment(self):
        headlines = []
        try:
            query = self.ticker.replace(".JK", "")
            long_name = self.info.get('longName', '')
            if long_name and long_name != self.ticker:
                query = long_name.replace("PT ", "").replace(" Tbk", "").strip()
            rss_url = f"https://news.google.com/rss/search?q={query}+Indonesia+saham&hl=id-ID&gl=ID&ceid=ID:id"
            response = requests.get(rss_url, timeout=5)
            if response.status_code == 200:
                soup = BeautifulSoup(response.content, features="xml")
                items = soup.findAll('item')
                for item in items[:5]: headlines.append(item.find('title').text)

            if not headlines:
                self.news_analysis = {"sentiment": "Neutral", "score": 0, "headlines": []}
                return

            score = 0
            for title in headlines:
                t_lower = title.lower()
                bull = sum(1 for w in FIN_BULLISH if w in t_lower)
                bear = sum(1 for w in FIN_BEARISH if w in t_lower)
                score += (bull * 1.5) - (bear * 1.5)

            final_score = score / len(headlines) if headlines else 0
            sentiment = "Positive" if final_score > 0.2 else "Negative" if final_score < -0.2 else "Neutral"
            self.news_analysis = {"sentiment": sentiment, "score": round(final_score, 3), "headlines": headlines[:3]}
        except Exception as e:
            self.news_analysis = {"sentiment": "Error", "score": 0, "headlines": [str(e)]}

    # --- MATH HELPERS ---
    def calc_rsi(self, series, period):
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def calc_ema(self, series, period):
        return series.ewm(span=period, adjust=False).mean()

    def calc_sma(self, series, period):
        return series.rolling(window=period).mean()

    def calc_std(self, series, period):
        return series.rolling(window=period).std()

    def calc_slope(self, series, period=20):
        if len(series) < period: return 0
        y = series.iloc[-period:].values
        x = np.arange(len(y))
        try:
            # Normalize
            y_norm = (y - np.mean(y)) / (np.std(y) + 1e-9)
            slope, _ = np.polyfit(x, y_norm, 1)
            return slope
        except: return 0

    def calc_atr(self, high, low, close, period):
        tr1 = high - low
        tr2 = (high - close.shift(1)).abs()
        tr3 = (low - close.shift(1)).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.rolling(window=period).mean()

    def calc_obv(self, close, volume):
        obv = (np.sign(close.diff()) * volume).fillna(0).cumsum()
        return obv

    def calc_mfi(self, high, low, close, volume, period):
        typical_price = (high + low + close) / 3
        raw_money_flow = typical_price * volume
        positive_flow = raw_money_flow.where(typical_price > typical_price.shift(1), 0)
        negative_flow = raw_money_flow.where(typical_price < typical_price.shift(1), 0)
        mf_ratio = positive_flow.rolling(window=period).sum() / negative_flow.rolling(window=period).sum()
        return 100 - (100 / (1 + mf_ratio))

    def calc_cmf(self, high, low, close, volume, period):
        mf_multiplier = ((close - low) - (high - close)) / (high - low)
        mf_volume = mf_multiplier * volume
        return mf_volume.rolling(window=period).sum() / volume.rolling(window=period).sum()

    def calc_force_index(self, close, volume, period):
        fi = close.diff(1) * volume
        return self.calc_ema(fi, period)

    def calc_stoch(self, high, low, close, k_period, d_period):
        lowest_low = low.rolling(window=k_period).min()
        highest_high = high.rolling(window=k_period).max()
        k = 100 * ((close - lowest_low) / (highest_high - lowest_low))
        d = k.rolling(window=d_period).mean()
        return k, d
        
    def calc_amihud(self, close, volume, period):
        ret = close.pct_change().abs()
        dol_vol = close * volume
        amihud = (ret / dol_vol) * 1000000000
        return amihud.rolling(window=period).mean()

    def calc_ad_line(self, high, low, close, volume):
        mfm = ((close - low) - (high - close)) / (high - low)
        mfm = mfm.fillna(0) 
        mfv = mfm * volume
        ad_line = mfv.cumsum()
        return ad_line

    def calc_pvt(self, close, volume):
        pct_change = close.pct_change().fillna(0)
        pvt = (pct_change * volume).cumsum()
        return pvt

    # --- HURST EXPONENT (Regime Filter) ---
    def calc_hurst(self, series, max_lag=20):
        try:
            lags = range(2, max_lag)
            tau = [np.std(series.diff(lag)) for lag in lags]
            # Avoid log(0)
            tau = [t if t > 0 else 0.000001 for t in tau]
            poly = np.polyfit(np.log(lags), np.log(tau), 1)
            return poly[0] * 2.0 
        except: return 0.5

    # --- Relative Strength vs Market ---
    def calc_relative_strength_score(self):
        try:
            if self.market_df is None or len(self.market_df) < 100: return 50
            
            stock_close = self.df['Close']
            market_close = self.market_df['Close']
            
            aligned = pd.concat([stock_close, market_close], axis=1, keys=['Stock', 'Market']).dropna()
            if len(aligned) < 252: return 50 
            
            s = aligned['Stock']
            m = aligned['Market']
            
            def safe_roc(series, period):
                if series.iloc[-period] == 0: return 1.0
                return series.iloc[-1] / series.iloc[-period]

            roc3 = safe_roc(s, 63) / safe_roc(m, 63)
            roc6 = safe_roc(s, 126) / safe_roc(m, 126)
            roc12 = safe_roc(s, 252) / safe_roc(m, 252)
            
            rs_val = (roc3 * 0.4) + (roc6 * 0.3) + (roc12 * 0.3)
            return round(rs_val * 100, 1) 
        except: return 50

    def prepare_indicators(self):
        if self.df is None or self.df.empty: return

        self.df['EMA_20'] = self.calc_ema(self.df['Close'], 20)
        self.df['EMA_50'] = self.calc_ema(self.df['Close'], 50)
        self.df['EMA_100'] = self.calc_ema(self.df['Close'], 100)
        self.df['EMA_150'] = self.calc_ema(self.df['Close'], 150)
        self.df['EMA_200'] = self.calc_ema(self.df['Close'], 200)
        self.active_trend_col = 'EMA_200'

        self.df['RSI'] = self.calc_rsi(self.df['Close'], self.config["RSI_PERIOD"])
        
        k, d = self.calc_stoch(self.df['High'], self.df['Low'], self.df['Close'], 14, 3)
        self.df[f"STOCHk"] = k
        self.df[f"STOCHd"] = d

        self.df['OBV'] = self.calc_obv(self.df['Close'], self.df['Volume'])
        self.df['AD_Line'] = self.calc_ad_line(self.df['High'], self.df['Low'], self.df['Close'], self.df['Volume'])
        self.df['PVT'] = self.calc_pvt(self.df['Close'], self.df['Volume'])
        
        self.df['CMF'] = self.calc_cmf(self.df['High'], self.df['Low'], self.df['Close'], self.df['Volume'], self.config["CMF_PERIOD"])
        self.df['MFI'] = self.calc_mfi(self.df['High'], self.df['Low'], self.df['Close'], self.df['Volume'], self.config["MFI_PERIOD"])
        self.df['VOL_MA'] = self.df['Volume'].rolling(window=self.config["VOL_MA_PERIOD"]).mean()
        self.df['RVOL'] = self.df['Volume'] / self.df['VOL_MA']
        self.df['TxValue'] = self.df['Close'] * self.df['Volume']
        self.df['EFI'] = self.calc_force_index(self.df['Close'], self.df['Volume'], 13)
        self.df['AMIHUD'] = self.calc_amihud(self.df['Close'], self.df['Volume'], 20)

        tp = (self.df['High'] + self.df['Low'] + self.df['Close']) / 3
        self.df['VWAP'] = (tp * self.df['Volume']).rolling(20).sum() / self.df['Volume'].rolling(20).sum()
        
        roc = self.df['Close'].pct_change()
        vol_down = self.df['Volume'] < self.df['Volume'].shift(1)
        nvi = [1000]
        for i in range(1, len(self.df)):
            if vol_down.iloc[i]:
                prev = nvi[-1]
                change = prev * roc.iloc[i]
                nvi.append(prev + change)
            else:
                nvi.append(nvi[-1])
        self.df['NVI'] = pd.Series(nvi, index=self.df.index)
        self.df['NVI_EMA'] = self.df['NVI'].ewm(span=255).mean()

        self.df['ATR'] = self.calc_atr(self.df['High'], self.df['Low'], self.df['Close'], self.config["ATR_PERIOD"])

    def check_liquidity_quality(self):
        try:
            adtv = self.df['TxValue'].rolling(20).mean().iloc[-1]
            min_adtv = self.config["MIN_ADTV_IDR"]
            if adtv < min_adtv:
                return {"status": "FAIL", "msg": f"Low Liquidity ({adtv/1e9:.2f}B IDR). High Risk.", "adtv": adtv}
            return {"status": "PASS", "msg": f"Healthy Liquidity ({adtv/1e9:.2f}B IDR)", "adtv": adtv}
        except: return {"status": "UNKNOWN", "msg": "Calc Error", "adtv": 0}

    def get_market_regime(self):
        regime = "NEUTRAL"
        if self.market_df is not None and not self.market_df.empty:
            m_close = self.market_df['Close']
            m_ma200 = m_close.rolling(200).mean().iloc[-1]
            if m_close.iloc[-1] > m_ma200:
                regime = "BULLISH"
            else:
                regime = "BEARISH"
        return regime

    def check_weekly_trend(self):
        try:
            logic = {'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'}
            weekly_df = self.df.resample('W').apply(logic).dropna()
            if len(weekly_df) < 30: return "UNKNOWN"
            weekly_df['EMA_30'] = weekly_df['Close'].ewm(span=30, adjust=False).mean()
            curr = weekly_df['Close'].iloc[-1]
            ema = weekly_df['EMA_30'].iloc[-1]
            if curr > ema: return "UPTREND"
            return "DOWNTREND"
        except: return "UNKNOWN"

    def check_trend_template(self):
        res = {"status": "FAIL", "score": 0, "max_score": 6, "details": []}
        try:
            has_200 = self.data_len >= 200
            if self.data_len < 50:
                 res["details"].append("Insufficient data (Need > 50 days)")
                 return res

            curr = self.df['Close'].iloc[-1]
            ema_50 = self.df['EMA_50'].iloc[-1]
            
            regime = self.get_market_regime()
            if regime == "BEARISH":
                res["details"].append("NOTE: Market (IHSG) is Bearish. Be cautious.")

            if not has_200:
                res["status"] = "IPO / NEW LISTING"
                res["max_score"] = 3
                ath = self.df['High'].max()
                atl = self.df['Low'].min()
                c1 = curr > ema_50
                c2 = curr >= (0.75 * ath)
                c3 = curr >= (1.25 * atl)
                score = sum([c1, c2, c3])
                res["score"] = score
                if score == 3: res["status"] = "IPO POWER TREND"
                elif score >= 1: res["status"] = "IPO UPTREND"
                if c1: res["details"].append("Price > EMA 50 (Short Term Trend)")
                if c2: res["details"].append("Near All-Time Highs")
                res["details"].append(f"Note: No EMA 200 yet ({self.data_len} days listed)")
                return res

            lookback = min(self.data_len, 260)
            ema_150 = self.df['EMA_150'].iloc[-1]
            ema_200 = self.df['EMA_200'].iloc[-1]
            year_high = self.df['High'].iloc[-lookback:].max()
            year_low = self.df['Low'].iloc[-lookback:].min()
            
            c1 = curr > ema_150 and curr > ema_200
            c2 = ema_150 > ema_200
            slope_200 = self.calc_slope(self.df['EMA_200'], 20)
            c3 = slope_200 > 0
            c4 = curr > ema_50
            c5 = curr >= (1.25 * year_low)
            c6 = curr >= (0.75 * year_high)
            
            score = sum([c1, c2, c3, c4, c5, c6])
            res["score"] = score
            if score == 6: res["status"] = "PERFECT UPTREND (Stage 2)"
            elif score >= 4: res["status"] = "STRONG UPTREND"
            elif score == 3: res["status"] = "WEAK UPTREND / RECOVERY"
            elif score <= 2: res["status"] = "DOWNTREND / BASE"
            
            if c1 and c2: res["details"].append("MA Alignment (Price > 150 > 200)")
            if c3: res["details"].append("200-Day MA Rising")
            if c5: res["details"].append(f"> 25% Off {lookback}-Day Lows")
            if c6: res["details"].append(f"Near {lookback}-Day Highs")
            if not c4: res["details"].append("WARNING: Price below 50 EMA")
        except Exception as e: res["details"].append(f"Error: {str(e)}")
        return res

    # --- VECTORIZED BACKTEST SLICING ---
    def _get_backtest_start_index(self):
        """Calculates integer index where backtest period starts."""
        try:
            period_str = self.config["BACKTEST_PERIOD"] # e.g. "2y"
            years = int(period_str.replace("y", ""))
            cutoff = self.df.index[-1] - pd.DateOffset(years=years)
            # Find closest index after cutoff
            return len(self.df[self.df.index < cutoff])
        except:
            return max(0, self.data_len - 500)

    def optimize_stock(self, days_min, days_max):
        """Vectorized Optimization with Return Stats"""
        best_res = {
            "strategy": None, "win_rate": -1, "details": "N/A", 
            "hold_days": 0, "is_triggered_today": False,
            "avg_win": 0, "avg_loss": 0
        }
        
        start_idx = self._get_backtest_start_index()
        close_np = self.df['Close'].values
        total_len = len(close_np)

        # Helper for vectorized calculation
        def fast_sim(signal_indices_np):
            best_wr = -1
            best_d = 0
            stats = {"avg_win": 0, "avg_loss": 0}
            
            # Loop hold days (scalar loop is fast if inner ops are vectorized)
            for d in range(days_min, days_max + 1):
                exit_indices = signal_indices_np + d
                # Filter indices that go out of bounds
                valid_mask = exit_indices < total_len
                if not np.any(valid_mask): continue
                
                valid_entries = signal_indices_np[valid_mask]
                valid_exits = exit_indices[valid_mask]
                
                entry_prices = close_np[valid_entries]
                exit_prices = close_np[valid_exits]
                
                # vectorized return calc
                returns = (exit_prices - entry_prices) / entry_prices
                wins = returns > 0
                win_count = np.sum(wins)
                count = len(returns)
                
                if count > 0:
                    wr = (win_count / count) * 100
                    if wr > best_wr:
                        best_wr = wr
                        best_d = d
                        # Calc average returns
                        win_rets = returns[wins]
                        loss_rets = returns[~wins]
                        
                        stats["avg_win"] = np.mean(win_rets) if len(win_rets) > 0 else 0
                        stats["avg_loss"] = np.mean(loss_rets) if len(loss_rets) > 0 else 0
                        
            return best_wr, best_d, stats

        # 1. RSI Strategy
        if 'RSI' in self.df.columns:
            rsi_np = self.df['RSI'].values
            for level in [self.config["RSI_LOWER"], self.config["RSI_LOWER"] + 10]:
                # Identify signal indices within backtest window
                # np.where returns tuple, get [0]
                sig_idxs = np.where((rsi_np < level))[0]
                sig_idxs = sig_idxs[sig_idxs >= start_idx] # Filter for backtest period
                
                if len(sig_idxs) > 0:
                    wr, h_days, stats = fast_sim(sig_idxs)
                    if wr > best_res['win_rate']:
                        best_res = {
                            "strategy": "RSI Reversal", 
                            "details": f"RSI < {level}", 
                            "win_rate": wr, 
                            "hold_days": h_days, 
                            "is_triggered_today": self.df['RSI'].iloc[-1] < level,
                            "avg_win": stats['avg_win'],
                            "avg_loss": stats['avg_loss']
                        }

        # 2. MA Trend Strategy
        if 'EMA_50' in self.df.columns and 'EMA_200' in self.df.columns:
            ema50 = self.df['EMA_50'].values
            ema200 = self.df['EMA_200'].values
            
            sig_idxs = np.where(ema50 > ema200)[0]
            sig_idxs = sig_idxs[sig_idxs >= start_idx]
            
            if len(sig_idxs) > 0:
                wr, h_days, stats = fast_sim(sig_idxs)
                if wr > best_res['win_rate']:
                    best_res = {
                        "strategy": "MA Trend", 
                        "details": "Trend Following (50 > 200)", 
                        "win_rate": wr, 
                        "hold_days": h_days, 
                        "is_triggered_today": (ema50[-1] > ema200[-1]),
                        "avg_win": stats['avg_win'],
                        "avg_loss": stats['avg_loss']
                    }
                    
        return best_res

    # --- MONTE CARLO SIMULATION ---
    def simulate_monte_carlo(self, best_strategy):
        """
        Runs a Monte Carlo simulation based on the backtested win rate and avg returns.
        Simulates 1000 equity curves for a 1-year period (approx 50 trades).
        """
        res = {"risk_of_ruin": 0, "median_return": 0, "worst_case": 0}
        try:
            win_rate = best_strategy.get('win_rate', 0) / 100.0
            avg_win = best_strategy.get('avg_win', 0)
            avg_loss = best_strategy.get('avg_loss', 0)
            
            if win_rate <= 0 or avg_win <= 0: return res
            
            # Simulation Parameters
            num_sims = 1000
            num_trades = 50 # Approx trades per year
            start_equity = 1.0 # Normalized
            
            # Pre-generate random outcomes (1 = win, 0 = loss)
            # shape: (1000, 50)
            outcomes = np.random.choice([1, 0], size=(num_sims, num_trades), p=[win_rate, 1-win_rate])
            
            # Map outcomes to returns
            # If 1, return is avg_win. If 0, return is avg_loss.
            returns = np.where(outcomes == 1, avg_win, avg_loss)
            
            # Calculate Equity Curves (Compound)
            # equity = start * (1 + r1) * (1 + r2) ...
            equity_curves = np.cumprod(1 + returns, axis=1)
            final_equities = equity_curves[:, -1]
            
            # Metrics
            median_eq = np.median(final_equities)
            res['median_return'] = (median_eq - 1) * 100
            
            # Worst Case (5th percentile - VaR 95)
            worst_eq = np.percentile(final_equities, 5)
            res['worst_case'] = (worst_eq - 1) * 100
            
            # Risk of Ruin (Drawdown > 50% at any point)
            # Check min equity along the path
            min_equities = np.min(equity_curves, axis=1)
            ruin_count = np.sum(min_equities < 0.5) # Ruin defined as 50% loss
            res['risk_of_ruin'] = (ruin_count / num_sims) * 100
            
        except: pass
        return res

    # --- CALCULATE TARGET PROBABILITIES ---
    def calculate_target_probabilities(self, entry_price, stop_loss_price, atr):
        """
        Calculates the probability of hitting 1R, 2R, 3R, 4R, 5R targets and the stop loss.
        Based on historical volatility (ATR) and recent price action.
        This is a simplified probabilistic model, not a full backtest of every target.
        """
        probs = {"1R": 0, "2R": 0, "3R": 0, "4R": 0, "5R": 0, "Stop Loss": 0}
        try:
            risk = entry_price - stop_loss_price
            if risk <= 0: return probs

            # Get historical data for simulation (last 2 years)
            start_idx = self._get_backtest_start_index()
            hist_df = self.df.iloc[start_idx:].copy()
            
            if len(hist_df) < 50: return probs

            # Identify similar setups: Price > EMA20, RSI > 50 (Trend Following Context)
            # We filter for days where the trend was broadly similar to today
            # This is a basic similarity search
            
            current_rsi = self.df['RSI'].iloc[-1]
            current_close = self.df['Close'].iloc[-1]
            current_ema50 = self.df['EMA_50'].iloc[-1]
            
            # Filter condition: Similar Trend Strength
            # 1. Price vs EMA50 relationship (Above/Below)
            is_above_ema50 = current_close > current_ema50
            
            # 2. RSI Regime (Bullish > 50, Bearish < 50)
            is_rsi_bullish = current_rsi > 50
            
            # Create mask for similar days
            mask = (
                (hist_df['Close'] > hist_df['EMA_50']) == is_above_ema50
            ) & (
                (hist_df['RSI'] > 50) == is_rsi_bullish
            )
            
            similar_days = hist_df[mask]
            
            if len(similar_days) < 20: 
                # Fallback to all data if not enough similar days
                similar_days = hist_df
            
            # Simulate trades from these similar days
            # We check max excursion (High - Open) vs max adverse excursion (Open - Low) over next 40 days
            
            outcomes = [] # Stores max R multiple reached before hitting -1R
            
            for idx in similar_days.index:
                # Need future data
                loc = self.df.index.get_loc(idx)
                if loc > (self.data_len - 41): continue
                
                # Assume entry at Close of that day (approximation)
                sim_entry = self.df['Close'].iloc[loc]
                sim_risk = self.df['ATR'].iloc[loc] * 1.5 # Using dynamic risk based on that day's ATR
                sim_stop = sim_entry - sim_risk
                
                # Look forward 40 days to allow for larger moves
                future_window = self.df.iloc[loc+1 : loc+41]
                
                # Check if stop hit first
                lows = future_window['Low'].values
                highs = future_window['High'].values
                
                hit_stop = False
                max_r = 0
                
                for i in range(len(lows)):
                    if lows[i] <= sim_stop:
                        hit_stop = True
                        break # Stopped out
                    
                    # Calculate current R multiple
                    current_gain = highs[i] - sim_entry
                    current_r = current_gain / sim_risk
                    if current_r > max_r:
                        max_r = current_r
                
                if hit_stop:
                    outcomes.append(-1) # Hit stop
                else:
                    outcomes.append(max_r) # Max R reached without stopping out
            
            if not outcomes: return probs
            
            outcomes = np.array(outcomes)
            total_sims = len(outcomes)
            
            # Calculate probabilities
            # Prob(Stop Loss) is roughly estimated by trades that hit -1R
            # However, in real trading, you hold until target or stop.
            # Let's use the empirical data:
            
            probs['Stop Loss'] = np.mean(outcomes == -1) * 100
            probs['1R'] = np.mean(outcomes >= 1) * 100
            probs['2R'] = np.mean(outcomes >= 2) * 100
            probs['3R'] = np.mean(outcomes >= 3) * 100
            probs['4R'] = np.mean(outcomes >= 4) * 100
            probs['5R'] = np.mean(outcomes >= 5) * 100
            
        except: pass
        return probs

    def backtest_smart_money_predictivity(self):
        res = {"accuracy": "N/A", "avg_return": 0, "count": 0, "verdict": "Unproven", "best_horizon": 0}
        try:
            start_idx = self._get_backtest_start_index()
            # Vectorized Signals
            # Conditions: CMF > 0.05, MFI < 80, Close > VWAP
            cond = (
                (self.df['CMF'].values > 0.05) & 
                (self.df['MFI'].values < 80) & 
                (self.df['Close'].values > self.df['VWAP'].values)
            )
            # Apply Backtest Period Filter
            all_indices = np.where(cond)[0]
            valid_indices = all_indices[all_indices >= start_idx]
            
            if len(valid_indices) < 5: return res
            
            close_np = self.df['Close'].values
            high_np = self.df['High'].values
            total_len = len(close_np)
            
            best_win_rate = -1
            best_stats = None

            # Check horizons
            for h in range(1, 41):
                # Filter indices that allow for h-day lookahead
                usable_indices = valid_indices[valid_indices < (total_len - h - 1)]
                if len(usable_indices) == 0: continue

                # Vectorized Entry/Exit
                entry_prices = close_np[usable_indices]
                exit_prices = close_np[usable_indices + h]
                
                # Check for "Win" (High > Entry * 1.02 within h days)
                # This part is harder to fully vectorize perfectly without a stride trick or loop
                # Simplification: Check if Exit > Entry (Profit) OR just check max high
                # We will stick to the loop logic but optimized with numpy slicing
                
                wins = 0
                for i, idx in enumerate(usable_indices):
                    # Only check every 5th signal to avoid overlap bias (simple sampling)
                    if i % 5 != 0: continue
                        
                    future_highs = high_np[idx+1 : idx+h+1]
                    if np.max(future_highs) > (entry_prices[i] * 1.02):
                        wins += 1
                
                # Sample size reduction due to skip logic
                valid_count = len(usable_indices) // 5
                if valid_count > 0:
                    wr = (wins / valid_count) * 100
                    # Quick avg return calc (simple exit)
                    returns = (exit_prices[::5] - entry_prices[::5]) / entry_prices[::5]
                    avg_ret = np.mean(returns) * 100
                    
                    if wr > best_win_rate:
                        best_win_rate = wr
                        best_stats = {
                            "accuracy": f"{wr:.1f}%", 
                            "avg_return": f"{avg_ret:.1f}%", 
                            "count": valid_count, 
                            "best_horizon": h
                        }

            if best_stats:
                verdict = "POOR"
                if best_win_rate > 70: verdict = "HIGHLY PREDICTIVE"
                elif best_win_rate > 50: verdict = "MODERATE"
                best_stats["verdict"] = verdict
                res = best_stats
        except Exception: pass
        return res

    def backtest_volume_breakout_behavior(self):
        res = {"accuracy": "N/A", "avg_return_5d": 0, "count": 0, "behavior": "Unknown", "best_horizon": 0}
        try:
            start_idx = self._get_backtest_start_index()
            
            # Vectorized Signals
            c = self.df['Close'].values
            o = self.df['Open'].values
            v = self.df['Volume'].values
            v_ma = self.df['VOL_MA'].values
            min_liq = self.config["MIN_DAILY_VOL"]
            
            # Logic: Green candle, Close > Prev Close, Vol > 1.5x MA, Liquidity OK
            # Shifted arrays for "Prev Close"
            c_prev = np.roll(c, 1)
            c_prev[0] = c[0] # Handle first element
            
            signals = (
                (c > o) & 
                (c > c_prev) & 
                (v > 1.5 * v_ma) & 
                ((c * v) > min_liq)
            )
            
            all_indices = np.where(signals)[0]
            valid_indices = all_indices[all_indices >= start_idx]
            
            if len(valid_indices) < 3: return res

            total_len = len(c)
            best_win_rate = -1
            best_stats = None

            for h in range(1, 41):
                usable_indices = valid_indices[valid_indices < (total_len - h - 1)]
                if len(usable_indices) == 0: continue
                
                entries = c[usable_indices]
                exits = c[usable_indices + h]
                
                returns = (exits - entries) / entries
                wins = np.sum(returns > 0.02) # Target > 2%
                count = len(returns)
                
                if count > 0:
                    wr = (wins / count) * 100
                    if wr > best_win_rate:
                        best_win_rate = wr
                        avg_ret = np.mean(returns) * 100
                        best_stats = {
                            "accuracy": f"{wr:.1f}%", 
                            "avg_return_5d": f"{avg_ret:.1f}%", 
                            "count": count, 
                            "best_horizon": h
                        }

            if best_stats:
                behavior = "HONEST (Trend Follower)" if best_win_rate > 60 else "FAKEOUT (Fade the Pop)" if best_win_rate < 40 else "MIXED / CHOPPY"
                best_stats["behavior"] = behavior
                res = best_stats
        except Exception: pass
        return res

    def _detect_low_cheat_on_slice(self, df_slice):
        res = {"detected": False}
        try:
            if len(df_slice) < 20: return res
            c0 = df_slice.iloc[-1]
            vol_ma = df_slice['Volume'].rolling(20).mean().iloc[-1]
            atr = df_slice['High'].rolling(14).max() - df_slice['Low'].rolling(14).min()
            vol_dry = c0['Volume'] < vol_ma * 0.8
            spread_tight = (c0['High'] - c0['Low']) < (atr / 14)
            recent_high = df_slice['High'].iloc[-20:].max()
            below_pivot = c0['Close'] < recent_high
            if vol_dry and spread_tight and below_pivot:
                res = {"detected": True}
        except: pass
        return res

    def detect_low_cheat(self):
        res = {"detected": False, "msg": ""}
        try:
            if self.data_len < 20: return res
            if self._detect_low_cheat_on_slice(self.df)["detected"]:
                 res = {"detected": True, "msg": "Valid Low Cheat Setup (Tight + Dry Vol)"}
        except Exception: pass
        return res

    def backtest_low_cheat_performance(self):
        # Optimization: Loop logic is complex due to slicing requirements, 
        # keep standard but restrict loop range effectively
        res = {"accuracy": "N/A", "count": 0, "verdict": "Unproven"}
        try:
            start_idx = self._get_backtest_start_index()
            # Ensure we have enough data before start_idx for lookbacks (20 days)
            loop_start = max(start_idx, 20)
            
            wins = 0
            valid_count = 0
            
            # Step 5 to speed up
            for i in range(loop_start, self.data_len - 10, 5):
                # We can't easily vectorize _detect_low_cheat_on_slice because it uses rolling on slice
                # Just slice minimally
                slice_df = self.df.iloc[i-25:i] # Only need last 25 days for calculation
                
                if self._detect_low_cheat_on_slice(slice_df)["detected"]:
                    valid_count += 1
                    entry = self.df['Close'].iloc[i-1]
                    future_high = self.df['High'].iloc[i:i+10].max()
                    future_low = self.df['Low'].iloc[i:i+10].min()
                    
                    if future_high > (entry * 1.03) and future_low > (entry * 0.98):
                        wins += 1
            
            if valid_count == 0: return res
            win_rate = (wins / valid_count) * 100
            verdict = "HIGH PROBABILITY" if win_rate > 65 else "RISKY" if win_rate < 40 else "MODERATE"
            res = {"accuracy": f"{win_rate:.1f}%", "count": valid_count, "verdict": verdict}
        except Exception: pass
        return res

    def backtest_fib_bounce(self):
        res = {"accuracy": "N/A", "count": 0, "verdict": "Unproven"}
        try:
            start_idx = self._get_backtest_start_index()
            lookback = self.config["FIB_LOOKBACK_DAYS"]
            loop_start = max(start_idx, lookback + 1)
            
            wins = 0
            count = 0
            
            # Vectorize High/Low access
            h_arr = self.df['High'].values
            l_arr = self.df['Low'].values
            
            # Iterate
            for i in range(loop_start, self.data_len - 15, 10):
                # Local min/max for fib
                curr_h_win = h_arr[i-lookback:i]
                curr_l_win = l_arr[i-lookback:i]
                sh = np.max(curr_h_win)
                sl = np.min(curr_l_win)
                rng = sh - sl
                
                if rng == 0: continue
                
                fib_618 = sh - (0.618 * rng)
                current_low = l_arr[i]
                
                if abs(current_low - fib_618) / fib_618 < 0.02:
                    count += 1
                    future_max = np.max(h_arr[i:i+15])
                    if future_max > (current_low * 1.05):
                        wins += 1
                        
            if count == 0: return res
            win_rate = (wins / count) * 100
            verdict = "GOLDEN ZONE" if win_rate > 65 else "WEAK SUPPORT" if win_rate < 40 else "NEUTRAL"
            res = {"accuracy": f"{win_rate:.1f}%", "count": count, "verdict": verdict}
        except Exception: pass
        return res

    def backtest_ma_support_all(self):
        res = {"best_ma": "None", "accuracy": "N/A", "count": 0, "verdict": "Unknown", "details": {}}
        try:
            start_idx = self._get_backtest_start_index()
            best_win_rate = -1
            periods = [20, 50, 100, 200]
            
            low_np = self.df['Low'].values
            high_np = self.df['High'].values
            total_len = len(low_np)
            
            for p in periods:
                ma_col = f'EMA_{p}'
                if ma_col not in self.df.columns: continue
                
                ma_np = self.df[ma_col].values
                
                # Vectorized slope check (curr > prev-5)
                # We need to ensure we don't access negative indices
                if total_len < 6: continue
                
                ma_prev_5 = np.roll(ma_np, 5)
                slope_ok = ma_np > ma_prev_5
                
                # Touch check
                touched = (low_np <= ma_np) & (high_np >= ma_np)
                
                # Valid signals in backtest period
                signals = slope_ok & touched
                all_indices = np.where(signals)[0]
                valid_indices = all_indices[all_indices >= start_idx]
                
                if len(valid_indices) < 3: continue
                
                wins = 0
                valid_count = 0
                last_idx = -100
                
                for idx in valid_indices:
                    # Skip check if close to end
                    if idx > (total_len - 10): continue
                    # Debounce (skip if within 5 days of last check)
                    if idx - last_idx < 5: continue
                    
                    entry = ma_np[idx]
                    future_high = np.max(high_np[idx+1 : idx+10])
                    
                    if future_high > (entry * 1.03): wins += 1
                    valid_count += 1
                    last_idx = idx
                    
                if valid_count > 0:
                    wr = (wins / valid_count) * 100
                    res["details"][f"EMA{p}"] = f"{wr:.0f}% ({wins}/{valid_count})"
                    if wr > best_win_rate:
                        best_win_rate = wr
                        verdict = "STRONG SUPPORT" if wr > 65 else "WEAK SUPPORT" if wr < 40 else "MODERATE"
                        res["best_ma"] = f"EMA {p}"
                        res["accuracy"] = f"{wr:.1f}%"
                        res["count"] = valid_count
                        res["verdict"] = verdict
        except Exception: pass
        return res

    # --- NEW: STATISTICALLY VALID DYNAMIC MA (QUANT EDITION) ---
    def find_best_dynamic_ma(self):
        best_ma = {"period": 0, "score": 0, "price": 0, "win_rate": 0, "slope": 0}
        try:
            # We use the full available data for this fit, 
            # but apply heavy weighting to recent data.
            
            c_arr = self.df['Close'].values
            l_arr = self.df['Low'].values
            h_arr = self.df['High'].values
            total_len = len(c_arr)
            
            if total_len < 100: return best_ma
            
            # Recency Weights: Linear increase from 0.5 to 1.5 over the dataset
            # This gives 3x more importance to recent data than the oldest data
            weights = np.linspace(0.5, 1.5, total_len)
            
            # Search space: 20 to 200
            for p in range(20, 201, 5):
                # 1. Calc EMA using pandas (fast enough for scalar loop)
                ema = self.df['Close'].ewm(span=p, adjust=False).mean().values
                
                # 2. Calc Slope (Vectorized)
                # Slope > 0 means EMA is rising
                slope = np.zeros_like(ema)
                slope[1:] = ema[1:] - ema[:-1]
                
                # 3. Identify Touches (The Setup)
                # Zone: 1.5% around EMA (Slightly relaxed to catch wicks)
                # Condition: Low dips into zone, Close holds above EMA, EMA is rising
                touches = (l_arr <= ema * 1.015) & (l_arr >= ema * 0.985) & (c_arr > ema) & (slope > 0)
                
                touch_indices = np.where(touches)[0]
                
                if len(touch_indices) < 3: continue
                
                weighted_score = 0
                wins = 0
                valid_signals = 0
                last_sig_idx = -10
                
                # 4. Validate Bounces (The Confirmation) - Vectorized loop
                # Check if price moved +3% within next 5 days (Performance Validation)
                for idx in touch_indices:
                    if idx >= total_len - 5: continue # Skip if no future data
                    if idx < 50: continue # Skip early noise
                    
                    # DEBOUNCE: Skip if signal is too close to previous (5 days)
                    if idx - last_sig_idx < 5: continue
                    
                    future_high = np.max(h_arr[idx+1 : idx+6])
                    entry_price = ema[idx] 
                    
                    if future_high > (entry_price * 1.03): # 3% bounce target
                        wins += 1
                        weighted_score += 1 * weights[idx] # Add weighted score
                    else:
                        weighted_score -= 0.5 * weights[idx] # Penalty for failure
                        
                    valid_signals += 1
                    last_sig_idx = idx
                
                if valid_signals < 3: continue
                
                # Calculate Win Rate
                win_rate = (wins / valid_signals) * 100
                
                # Selection Criteria: High Weighted Score AND Decent Win Rate (>50%)
                if weighted_score > best_ma["score"] and win_rate > 50:
                    best_ma = {
                        "period": p,
                        "score": round(weighted_score, 1),
                        "price": ema[-1],
                        "win_rate": round(win_rate, 1),
                        "slope": slope[-1]
                    }
            
            # Fallback if nothing good found
            if best_ma["score"] == 0:
                 best_ma = {"period": 0, "score": 0, "price": 0, "win_rate": 0, "slope": 0}

        except Exception as e: 
            pass
            
        return best_ma

    # --- NEW: DYNAMIC STOCHASTIC OPTIMIZATION ---
    def find_best_stoch_settings(self):
        """
        Iterates through common Stochastic settings to find the one with the highest predictive power
        for the specific stock over the last 1 year.
        """
        best_stoch = {"k": 14, "d": 3, "win_rate": 0, "score": 0}
        
        # Candidate Settings: (K Period, D Period)
        # 14,3 = Standard
        # 5,3 = Fast/Sensitive
        # 8,3 = Balanced
        # 21,5 = Slow/Smooth
        # 10,5 = Fast but smooth (User request)
        candidates = [(14, 3), (5, 3), (8, 3), (21, 5), (10, 5)]
        
        try:
            # Limit backtest to last 250 days for speed & relevance
            hist_len = min(250, self.data_len)
            start_idx = self.data_len - hist_len
            
            close_np = self.df['Close'].values
            high_np = self.df['High'].values
            low_np = self.df['Low'].values
            
            for k_p, d_p in candidates:
                # 1. Manual Stoch Calc (Vectorized) to avoid modifying DF repeatedly
                # Rolling Min/Max
                low_min = pd.Series(low_np).rolling(window=k_p).min().values
                high_max = pd.Series(high_np).rolling(window=k_p).max().values
                
                # Fast %K
                # Handle div by zero
                denom = high_max - low_min
                denom[denom == 0] = 0.00001
                stoch_k = 100 * ((close_np - low_min) / denom)
                
                # Fast %D (SMA of K)
                stoch_d = pd.Series(stoch_k).rolling(window=d_p).mean().values
                
                # 2. Vectorized Backtest
                # Signal: K crosses above D while both < 20 (Oversold Buy)
                # Logic: K[i] > D[i] AND K[i-1] < D[i-1] AND K[i] < 20
                
                prev_k = np.roll(stoch_k, 1)
                prev_d = np.roll(stoch_d, 1)
                
                signals = (stoch_k > stoch_d) & (prev_k < prev_d) & (stoch_k < 20)
                sig_indices = np.where(signals)[0]
                
                # Filter indices
                valid_indices = sig_indices[(sig_indices >= start_idx) & (sig_indices < self.data_len - 5)]
                
                if len(valid_indices) < 3: continue
                
                wins = 0
                valid_count = 0
                
                for idx in valid_indices:
                    # Check 5 day forward return
                    entry = close_np[idx]
                    # Target: > 2% gain
                    future_high = np.max(high_np[idx+1 : idx+6])
                    
                    if future_high > entry * 1.02:
                        wins += 1
                    valid_count += 1
                
                if valid_count > 0:
                    wr = (wins / valid_count) * 100
                    # Score = Win Rate * Log(Count) to bias towards reliable signals
                    score = wr * np.log(valid_count + 1)
                    
                    if score > best_stoch["score"]:
                        best_stoch = {"k": k_p, "d": d_p, "win_rate": wr, "score": score}
                        
        except Exception as e: pass
        
        return best_stoch

    # --- NEW: DYNAMIC RSI OPTIMIZATION ---
    def find_best_rsi_settings(self):
        """
        Iterates through common RSI periods to find the one with the highest predictive power
        for Oversold reversals (RSI < 30) over the last 1 year.
        """
        best_rsi = {"period": 14, "win_rate": 0, "score": 0}
        
        # Candidate Settings: 9 (Fast), 14 (Standard), 21 (Smooth), 25 (Trend)
        candidates = [9, 14, 21, 25]
        
        try:
            hist_len = min(250, self.data_len)
            start_idx = self.data_len - hist_len
            
            close_np = self.df['Close'].values
            high_np = self.df['High'].values
            
            for p in candidates:
                # 1. Calc RSI manually (Vectorized)
                delta = pd.Series(close_np).diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=p).mean().values
                loss = (-delta.where(delta < 0, 0)).rolling(window=p).mean().values
                
                # Handle div by zero
                loss[loss == 0] = 0.00001
                rs = gain / loss
                rsi_vals = 100 - (100 / (1 + rs))
                
                # 2. Backtest Oversold Reversals (RSI < 30)
                # Signal: RSI is below 30
                signals = (rsi_vals < 30)
                sig_indices = np.where(signals)[0]
                
                # Filter indices
                valid_indices = sig_indices[(sig_indices >= start_idx) & (sig_indices < self.data_len - 5)]
                
                if len(valid_indices) < 3: continue
                
                wins = 0
                valid_count = 0
                
                # To prevent overcounting (debounce), skip if signal within last 5 days
                last_sig = -10
                
                for idx in valid_indices:
                    if idx - last_sig < 5: continue
                    
                    entry = close_np[idx]
                    future_high = np.max(high_np[idx+1 : idx+6])
                    
                    if future_high > entry * 1.02: # 2% gain target
                        wins += 1
                    valid_count += 1
                    last_sig = idx
                
                if valid_count > 0:
                    wr = (wins / valid_count) * 100
                    score = wr * np.log(valid_count + 1)
                    
                    if score > best_rsi["score"]:
                        best_rsi = {"period": p, "win_rate": wr, "score": score}
                        
        except Exception as e: pass
        
        return best_rsi

    # --- NEW: DETECT DIVERGENCE WITH DURATION (OBV & ACCUMULATION) ---
    def detect_smart_money_divergence(self):
        """
        Detects if Price is making Lower Lows while OBV is making Higher Lows (Bullish Divergence).
        Also calculates how long this divergence has been building.
        """
        res = {"detected": False, "msg": "", "duration": 0}
        try:
            if self.data_len < 60: return res
            
            # Use last 60 days
            window = 60
            close = self.df['Close'].iloc[-window:]
            obv = self.df['OBV'].iloc[-window:]
            
            # Simple Regression Slope over rolling windows to find sustained divergence
            # We look for a period where Price Slope < 0 and OBV Slope > 0
            
            divergence_days = 0
            
            # Check backwards from today
            for i in range(2, window):
                # Slice last i days
                p_slice = close.iloc[-i:]
                o_slice = obv.iloc[-i:]
                
                p_slope = self.calc_slope(p_slice, i)
                o_slope = self.calc_slope(o_slice, i)
                
                # Check for Strong Divergence: Price Falling, OBV Rising
                if p_slope < -0.1 and o_slope > 0.1:
                    divergence_days = i
                else:
                    # Break if pattern stops (we want contiguous days from now)
                    # Actually, for "how long", we want the max window that satisfies this
                    # But divergence is often a specific setup, not a continuous state for 60 days.
                    # Let's check if the *current* state is divergent.
                    pass

            # Alternative: Detect divergence over fixed windows (14, 21, 30 days) and pick the longest
            max_div_days = 0
            
            for d in [14, 21, 30, 45, 60]:
                p_chg = (close.iloc[-1] - close.iloc[-d]) / close.iloc[-d]
                o_chg = (obv.iloc[-1] - obv.iloc[-d]) / obv.iloc[-d] # Normalized change
                
                # Need to be careful with OBV magnitude, simple slope is better
                p_slope = self.calc_slope(close.iloc[-d:], d)
                o_slope = self.calc_slope(obv.iloc[-d:], d)

                if p_slope < 0 and o_slope > 0:
                    max_div_days = d
            
            if max_div_days > 0:
                 res = {
                     "detected": True, 
                     "msg": f"Bullish Divergence (Price Down, OBV Up)",
                     "duration": max_div_days
                 }
                 
        except: pass
        return res

    def get_market_context(self):
        last_price = self.df['Close'].iloc[-1]
        lookback = min(20, self.data_len)
        recent = self.df[-lookback:]
        support, resistance = recent['Low'].min(), recent['High'].max()
        dist_supp = ((last_price - support) / support) * 100
        
        fib_len = self.config["FIB_LOOKBACK_DAYS"]
        fib_win = self.df[-min(fib_len, self.data_len):]
        sh, sl = fib_win['High'].max(), fib_win['Low'].min()
        rng = sh - sl
        fibs = { "1.0 (Low)": sl, "0.618 (Golden)": sh-(0.618*rng), "0.5 (Half)": sh-(0.5*rng), "0.382": sh-(0.382*rng), "0.0 (High)": sh } if rng > 0 else {}
        
        atr = self.df['ATR'].iloc[-1] if 'ATR' in self.df.columns else 0

        change_pct = 0
        if len(self.df) >= 2:
            c0 = self.df['Close'].iloc[-1]
            c1 = self.df['Close'].iloc[-2]
            change_pct = ((c0 - c1) / c1) * 100
        
        ma_values = {
            "EMA_20": self.df['EMA_20'].iloc[-1] if 'EMA_20' in self.df.columns else 0,
            "EMA_50": self.df['EMA_50'].iloc[-1] if 'EMA_50' in self.df.columns else 0,
            "EMA_150": self.df['EMA_150'].iloc[-1] if 'EMA_150' in self.df.columns else 0,
            "EMA_200": self.df['EMA_200'].iloc[-1] if 'EMA_200' in self.df.columns else 0,
        }
        
        sm = self.analyze_smart_money_enhanced()
        weekly_trend = self.check_weekly_trend() # Sniper: Weekly Trend Check
        
        # New: Dynamic MA
        best_ma = self.find_best_dynamic_ma()

        # New: Detect Inside Bar
        inside_bar = self.detect_inside_bar()

        return {
            "price": last_price, "change_pct": change_pct, 
            "ma_values": ma_values, 
            "support": support, "resistance": resistance,
            "dist_support": dist_supp, "fib_levels": fibs, 
            "atr": atr, "smart_money": sm, "weekly_trend": weekly_trend,
            "best_ma": best_ma, # Added dynamic MA
            "inside_bar": inside_bar, # Added inside bar
            "wyckoff": self.detect_wyckoff_spring(), # --- NEW: Wyckoff Spring
            "vcp": self.detect_vcp_pattern(), 
            "candle": self.detect_candle_patterns(), # --- UPDATED: Morning Star
            "vsa": self.detect_vsa_anomalies(),
            "low_cheat": self.detect_low_cheat(),
            "squeeze": self.detect_ttm_squeeze(),
            "fundamental": self.check_fundamentals(),
            "pivots": self.calculate_pivot_points(),
            "vol_breakout": self.detect_volume_breakout(),
            "sm_predict": self.backtest_smart_money_predictivity(),
            "breakout_behavior": self.backtest_volume_breakout_behavior(),
            "lc_stats": self.backtest_low_cheat_performance(), 
            "fib_stats": self.backtest_fib_bounce(),
            "ma_stats": self.backtest_ma_support_all(),
            "bandar_ml": self.detect_smart_money_divergence(), # REPLACED: Divergence Check
            "hurst": self.calc_hurst(self.df['Close']), # --- NEW: Hurst
            "rs_score": self.calc_relative_strength_score(), # --- NEW: RS Score
            "mc_sim": self.simulate_monte_carlo(self.optimize_stock(1, 60)), # --- NEW: Monte Carlo
            "best_stoch": self.find_best_stoch_settings(), # --- NEW: Dynamic Stoch
            "best_rsi": self.find_best_rsi_settings() # --- NEW: Dynamic RSI
        }

    def generate_final_report(self):
        if not self.fetch_data(): return None
        self.prepare_indicators()
        self.analyze_news_sentiment()
        
        liq = self.check_liquidity_quality()
        best_strategy = self.optimize_stock(1, 60)
        
        trend_template = self.check_trend_template()
        ctx = self.get_market_context()
        
        action = "WAIT"
        if best_strategy['is_triggered_today'] and trend_template['score'] >= 4: action = "ACTION: BUY"
        val_score, val_verdict, val_reasons = self.validate_signal(action, ctx, trend_template)
        prob_data = self.calculate_probability(best_strategy, ctx, trend_template)

        plan = self.calculate_trade_plan_hybrid(ctx, trend_template['status'], best_strategy)

        return {
            "ticker": self.ticker, "name": self.info.get('longName', self.ticker),
            "price": ctx['price'], "change_pct": ctx['change_pct'], 
            "sentiment": self.news_analysis, "context": ctx,
            "plan": plan, 
            "validation": {"score": val_score, "verdict": val_verdict, "reasons": val_reasons},
            "probability": prob_data,
            "trend_template": trend_template, "liquidity": liq,
            "best_strategy": best_strategy,
            "is_ipo": self.data_len < 200, "days_listed": self.data_len
        }


