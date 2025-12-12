import sys
import argparse
from engine import StrategyEngine

def print_separator():
    print("-" * 50)

def print_header(text):
    print(f"\n=== {text.upper()} ===")

def format_currency(val):
    return f"Rp {int(val):,}"

def main():
    parser = argparse.ArgumentParser(description="IHSG Swing Trading Quant Engine")
    parser.add_argument("ticker", type=str, help="Stock Ticker (e.g., BBRI)")
    args = parser.parse_args()

    ticker = args.ticker.upper()
    
    # Auto-append .JK for Indonesia Stock Exchange if missing
    if not ticker.endswith(".JK"):
        ticker += ".JK"

    print(f"\nInitializing Quantitative Engine for {ticker}...")
    print("Fetching 3 years of OHLCV data via yfinance...")
    print("Running Grid Search Optimization (MA, RSI, Stoch)...")
    print("Backtesting strategies with 1:3 R:R mandate...")
    
    try:
        engine = StrategyEngine(ticker)
        result = engine.run_optimization()
    except Exception as e:
        print(f"\nERROR: Could not analyze {ticker}. {str(e)}")
        print("Make sure the ticker is correct and available on Yahoo Finance.")
        sys.exit(1)

    # --- DASHBOARD OUTPUT ---

    # 1. VERDICT CARD
    print("\n" + "=" * 50)
    print(f" STOCK: {ticker.replace('.JK', '')}   |   VERDICT: {result['verdict']}")
    print("=" * 50)

    # 2. TRADE PLAN
    if result['verdict'] == "BUY":
        plan = result['plan']
        print_header("EXECUTION PLAN (OJK Compliant)")
        print(f" ENTRY PRICE : {format_currency(plan['entry'])}")
        print(f" STOP LOSS   : {format_currency(plan['sl'])} (Risk 1R)")
        print_separator()
        print(f" TARGET 1    : {format_currency(plan['tp1'])} (1R)")
        print(f" TARGET 2    : {format_currency(plan['tp2'])} (2R)")
        print(f" TARGET 3    : {format_currency(plan['tp3'])} (3R)")
    
    # 3. THE LOGIC
    print_header("THE LOGIC (WHY?)")
    print(f" {result['logic']}")
    
    # 4. SAFETY SCORE / STATS
    print_header("SAFETY SCORE")
    if result['stats']:
        wr = result['stats']['win_rate']
        count = result['stats']['trade_count']
        rating = "HIGH" if wr > 75 else "MODERATE"
        print(f" Strategy Used    : {result['stats']['strategy']}")
        print(f" Historical WR    : {wr:.2f}% (over {count} trades)")
        print(f" Confidence Lvl   : {rating}")
    else:
        print(" No strategy achieved the mandatory >65% Win Rate.")
        print(" Recommendation: Do not trade. Wait for better conditions.")

    # 5. TECHNICAL DEEP DIVE
    dd = result['deep_dive']
    print_header("TECHNICAL DEEP DIVE")
    print(f" Current Price    : {format_currency(dd['price'])}")
    print(f" Scipy Support    : {format_currency(dd['support_scipy'])}")
    print(f" RSI (14)         : {dd['rsi']:.2f}")
    print(f" Stoch K          : {dd['stoch_k']:.2f}")
    print(f" MA Squeeze?      : {'YES' if dd['ma_squeeze'] else 'NO'}")
    print_separator()
    print(" [SMART MONEY ANALYSIS]")
    print(f" OBV Slope        : {dd['obv_slope']:.4f}")
    print(f" Smart Money Start: {dd['smart_money_start']}")
    
    # Interpretation of slope
    if dd['obv_slope'] > 0:
        print(" Status           : ACCUMULATION DETECTED")
    else:
        print(" Status           : DISTRIBUTION / NEUTRAL")
    
    print("\n" + "=" * 50 + "\n")

if __name__ == "__main__":
    main()

