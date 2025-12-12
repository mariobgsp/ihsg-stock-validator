import argparse
import sys
from engine import StrategyEngine

def print_separator():
    print("-" * 60)

def print_header(text):
    print(f"\n=== {text.upper()} ===")

def format_currency(val):
    return f"Rp {int(val):,}"

def main():
    parser = argparse.ArgumentParser(description="IHSG Swing Trading CLI")
    parser.add_argument("ticker", type=str, help="Stock Ticker (e.g., BBRI, TLKM)")
    args = parser.parse_args()

    ticker = args.ticker.upper()
    print(f"\nInitializing Quantitative Engine for {ticker}...")
    print("Fetching data (yfinance), optimizing parameters, and running Monte Carlo validation...")

    try:
        engine = StrategyEngine(ticker)
        result = engine.run_simulation()
        
        # --- UI RENDERING ---
        
        # 1. VERDICT SECTION
        print_header("STRATEGY VERDICT")
        print(f"VERDICT:  [{result['verdict']}]")
        print(f"LOGIC:    {result['reason']}")
        
        if result['verdict'] != "NO TRADE":
            print(f"VWAP:     Price is {result['data']['vwap_diff']}% relative to VWAP")
            
            print_separator()
            print("EXECUTION PLAN (OJK Compliant Ticks):")
            print(f"• ENTRY:  {format_currency(result['entry'])}")
            print(f"• STOP:   {format_currency(result['sl'])} (Strict Exit)")
            print(f"• TP 1:   {format_currency(result['targets'][0])} (Risk 1R)")
            print(f"• TP 2:   {format_currency(result['targets'][1])} (Risk 2R)")
            print(f"• TP 3:   {format_currency(result['targets'][2])} (Risk 3R)")
            
            print_separator()
            print("SAFETY SCORE (Monte Carlo Validated):")
            probs = result['probs']
            print(f"• Win Rate (>1R):          {probs['win_rate']}%")
            print(f"• Probability to hit 1R:   {probs['p1']}%")
            print(f"• Probability to hit 2R:   {probs['p2']}%")
            print(f"• Probability to hit 3R:   {probs['p3']}%")
        
        # 2. DATA SECTION
        print_header("TECHNICAL DEEP DIVE")
        data = result['data']
        
        asset_type = "IPO / NEW LISTING" if data['is_ipo'] else "MATURE ASSET"
        print(f"Asset Class:   {asset_type}")
        
        print("\n[SMART MONEY / BANDARMOLOGY]")
        print(f"Phase:         {data['sm_phase']}")
        print(f"Cycle Start:   {data['sm_start']}")
        
        print("\n[OPTIMIZED INDICATORS]")
        print(f"Params Used:   MA Fast {data['params']['ma_fast']}, Slow {data['params']['ma_slow']}, RSI {data['params']['rsi_period']}")
        print(f"Current RSI:   {data['rsi']}")
        
        print("\n[PATTERN RECOGNITION]")
        vcp = "DETECTED" if data['patterns']['vcp'] else "None"
        sqz = "ACTIVE" if data['patterns']['squeeze'] else "None"
        print(f"VCP Setup:     {vcp}")
        print(f"MA Squeeze:    {sqz}")
        
        print_separator()
        print("Disclaimer: Trade at your own risk. Past performance != Future results.")
        print("This tool is for educational quantitative analysis only.\n")

    except Exception as e:
        print(f"\n[CRITICAL ERROR] Engine Failure: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()

