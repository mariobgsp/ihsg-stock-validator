# IHSG Ultimate Scanner (Sniper Edition)

A high-precision stock analysis tool designed for the Indonesian Stock Exchange (IDX/IHSG). This program combines classical Technical Analysis (Minervini Trend Template), Smart Money/Bandarmology logic, and advanced Machine Learning (XGBoost) to identify high-probability swing trading setups.

---

## Key Features

### 1. AI & Machine Learning (XGBoost)
- Price Forecast (Bullish/Bearish directional bias)
- Bandar Accumulation AI (Detects Smart Money accumulation)
- AI Support Projection (Regresses a projected "floor" price)

### 2. Smart Money Analysis
- CMF & MFI Analysis
- NVI (Net Volume Index)
- Stealth Accumulation Detection
- Distribution Detection

### 3. Setup & Structure Detection
- VCP (Volatility Contraction Pattern)
- Low Cheat (Early entry points)
- Inside Bar (Coiling price action)
- TTM Squeeze (Explosive energy buildup)

### 4. Trend & Fundamentals Filters
- Minervini Trend Template (Stage 2 Uptrend)
- Market Regime (\^JKSE health)
- Fundamentals (Altman Z-Score, Market Cap, Profitability)

### 5. Risk Management (Sniper Logic)
- Position Sizing (Calculates lot size based on risk %)
- R:R Calculation (Projects 3R targets and Stop Losses based on ATR)

---

## Installation

### Prerequisites
- Python 3.8 or higher.

### Install Dependencies
pip install yfinance pandas numpy requests beautifulsoup4 textblob xgboost scikit-learn

---

## Usage

### Basic Scan
python main.py BBCA

### Advanced Configuration
python main.py BMRI --period 5y --risk 2.0 --balance 50000000

### Available Arguments
| Flag | Default | Description |
| :--- | :--- | :--- |
| ticker | (Required) | The stock symbol (e.g., BBCA, GOTO). |
| --balance | 100,000,000 | Account balance in IDR. |
| --risk | 1.0 | Risk per trade in percent. |
| --period | 2y | Historical data lookback period. |
| --sl | 3.0 | Stop Loss multiplier (x ATR). |
| --tp | 9.0 | Take Profit multiplier (x ATR). |
| --min_vol | 1,000,000,000 | Minimum daily volume (IDR). |

---

## Understanding the Report

1. Liquidity Check
2. Trend Health (Score >= 4 is ideal)
3. Smart Money (Buying vs. Selling Pressure)
4. AI/Machine Learning (Forecast, Confidence > 70%, Accumulation Score, AI Support)
5. Trade Plan (Entry, SL, Target, Lot Size)

---

## Disclaimer

This software is for educational and research purposes only. Predictions do not guarantee future results. Always do your own research (DYOR).
