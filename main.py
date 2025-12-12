import argparse
import sys
from engine import StrategyEngine, IHSGTickRule, SmartMoneyAnalyzer

def print_header():
    print("="*60)
    print("      IHSG QUANT TRADING ENGINE | ARCHITECT: GEMINI")
    print("      OJK COMPLIANT | BANDARMOLOGY | VCP | OPTIMIZER")
    print("="*60)

def main():
    parser = argparse.ArgumentParser(description="IHSG Swing Trading CLI")
    parser.add_argument("ticker", type=str, help="Stock Ticker (e.g., BBCA, TLKM)")
    args = parser.parse_args()

    ticker = args.ticker.upper()
    print_header()
    print(f"[*] Initializing Engine for {ticker}...")
    print(f"[*] Fetching 3 Years OHLCV Data...")
    
    try:
        engine = StrategyEngine(ticker)
    except Exception as e:
        print(f"[!] Error: {e}")
        sys.exit(1)

    print(f"[*] Running Grid Search Optimization (MA, RSI, Stoch)...")
    print(f"[*] Filtering for >50% Win Rate Strategies...")
    
    signal = engine.optimize()
    sr_levels = engine.get_support_resistance()
    sm_status, sm_time = SmartMoneyAnalyzer.analyze_flow(engine.df)
    
    print("\n" + "-"*60)
    print(f" ANALYSIS RESULT: {ticker}")
    print("-" * 60)
    
    # 1. Action Block
    color = "\033[92m" if signal.action == "BUY" else "\033[93m"
    reset = "\033[0m"
    print(f"ACTION RECOMMENDATION : {color}{signal.action}{reset}")
    print(f"STRATEGY TYPE         : {signal.strategy_name}")
    print("-" * 60)
    
    # 2. Smart Money & Levels
    print(f"SMART MONEY FLOW      : {sm_status} (Shifted: {sm_time})")
    print(f"NEAREST SUPPORT (S1)  : Rp {sr_levels['Pivot_S1']}")
    print(f"GOLDEN RATIO FIB      : Rp {sr_levels['Fib_Golden']}")
    print("-" * 60)
    
    if signal.action == "BUY":
        print(" TRADE PLAN (OJK Adjusted):")
        print(f"   ENTRY PRICE    : Rp {IHSGTickRule.adjust(signal.entry_price)}")
        print(f"   STOP LOSS      : Rp {signal.stop_loss} (Risk: Rp {signal.entry_price - signal.stop_loss})")
        print(f"   TARGET (3R)    : Rp {signal.target_price}")
        print("\n PROBABILITIES (Based on Historical Optimization):")
        print(f"   Win Rate (>1R) : {signal.win_rate:.1%}")
        print(f"   Prob Hit 1R    : {signal.prob_1r:.1%}")
        print(f"   Prob Hit 2R    : {signal.prob_2r:.1%}")
        print(f"   Prob Hit 3R    : {signal.prob_3r:.1%}")
        
    print("-" * 60)
    print(f"REASONING:\n{signal.reasoning}")
    print("-" * 60)
    print(f"OPTIMIZED PARAMETERS USED:")
    print(str(signal.optimized_params))
    print("="*60)

if __name__ == "__main__":
    main()