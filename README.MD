# Simple IHSG Stock Validator (Advanced CLI)

A sophisticated, modular **Command Line Interface (CLI)** tool designed
for the **Indonesian Stock Exchange (IDX)**.\
It combines **algorithmic backtesting**, **pattern recognition (VCP,
Triangles)**, **sentiment analysis**, and **dynamic risk management** to
validate trading ideas with real statistical evidence.

## 🚀 Key Features

### 1. 🧠 Deep Grid Search Optimization

-   Multi-strategy backtesting (RSI Reversal, MA Momentum, Stochastic)
-   Hold period optimization (1--20 days)
-   Actionable vs. historical best strategy separation

### 2. 🛡️ Smart Trade Plan & Risk Management

-   ATR-based dynamic stop loss & take profit
-   Intelligent WAIT mode with suggested limit order

### 3. 📉 Advanced Pattern Recognition

-   VCP detection
-   Triangle & Pennant pattern recognition

### 4. 🏦 Smart Money Analysis (Bandarmology)

-   OBV Divergence
-   CMF & MFI
-   Volume spike alerts

### 5. 🔢 Fibonacci & Context Analysis

-   Auto Fibonacci Retracement (120-day)
-   Adaptive trend filter (EMA 200/50/20)

### 6. 📰 Hybrid News Sentiment

-   Yahoo Finance + Google RSS fallback
-   TextBlob sentiment scoring

## 🛠️ Installation

``` bash
pip install -r requirements.txt
```

Required libraries: - yfinance\
- pandas, numpy\
- pandas_ta\
- textblob\
- beautifulsoup4, lxml

## 🖥️ Usage

### Standard Mode

``` bash
python main.py BBCA
```

### Custom Mode Examples

``` bash
python main.py GOTO --period 6mo --sl 1.5 --tp 2.0 --hold 5
python main.py TLKM --period 5y --sl 3.0 --tp 6.0 --fib 200
python main.py BREN --cmf 10 --mfi 10
```

### Flags

  Flag       Description                  Default
  ---------- ---------------------------- ---------
  --period   Backtest timeframe           2y
  --sl       Stop Loss ATR Multiplier     2.0
  --tp       Take Profit ATR Multiplier   3.0
  --hold     Max holding days             20
  --rsi      RSI Period                   14
  --fib      Fibonacci lookback           120
  --cmf      CMF period                   20
  --mfi      MFI period                   14

## ⚠️ Disclaimer

For educational purposes only.\
No guarantee of future profits.\
Do your own research.

### Additional Notice

This program was **created using Gemini 3.0**.
