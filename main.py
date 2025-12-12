import argparse
import sys
from engine import QuantEngine

def print_header():
    print("="*60)
    print("      IHSG SWING TRADER PRO - QUANT ENGINE")
    print("      Powered by: Dynamic Optimization & Bandar Flow")
    print("="*60)

def main():
    parser = argparse.ArgumentParser(description="Professional Swing Trade Analyzer for IHSG")
    parser.add_argument("ticker", type=str, help="Stock Ticker (e.g., BBRI, TLKM)")
    args = parser.parse_args()

    print_header()
    print(f"[*] Initializing Engine for {args.ticker.upper()}...")
    print("[*] Fetching 3 Years of OHLCV Data...")
    print("[*] Backtesting & Optimizing Indicators...")
    
    try:
        engine = QuantEngine(args.ticker)
        rec = engine.run_full_analysis()
        
        res = engine.analysis_results
        
        print("\n" + "-"*60)
        print(f"ANALYSIS REPORT: {rec['ticker']} @ Rp {rec['price']:,.0f}")
        print("-"*60)
        
        # 1. Technical Health
        print(f"\n[INDICATOR OPTIMIZATION]")
        print(f" > Best Strategy Found: MA {rec['optimized_settings']['MA']}, RSI {rec['optimized_settings']['RSI']}, Stoch {rec['optimized_settings']['Stoch']}")
        print(f" > Historical Win Rate: {res['backtest']['win_rate']:.1%} ({res['backtest']['total_trades']} trades)")
        
        # 2. Bandar Flow
        b_status = res['bandar']['status']
        color_code = "\033[92m" if b_status == "Accumulation" else "\033[91m"
        reset_code = "\033[0m"
        print(f"\n[BANDARMOLOGY]")
        print(f" > Status: {color_code}{b_status}{reset_code}")
        if res['bandar']['start_date'] != "N/A":
            print(f" > Detected Start: {res['bandar']['start_date']}")
        print(f" > Current VWAP: Rp {res['bandar']['vwap']:,.2f}")
        
        # 3. Structure
        print(f"\n[PRICE STRUCTURE]")
        print(f" > Strong Support (Pivot/Fibo): Rp {res['support']['pivot_strong']:,.0f}")
        print(f" > Superclose MA (Squeeze): {'YES' if res['patterns']['superclose_ma'] else 'NO'} (Spread: {res['patterns']['ma_spread_pct']:.2f}%)")
        print(f" > VCP Pattern: {'YES' if res['patterns']['vcp_detected'] else 'NO'}")
        
        # 4. Final Recommendation
        print("\n" + "="*60)
        print(f"RECOMMENDATION: {rec['action']}")
        print(f"PROBABILITY: {rec['probability']:.1%}")
        print("="*60)
        
        if rec['action'] != "WAIT / AVOID":
            print(f"PLAN (R:R 3:1 - OJK Ticks Compliant):")
            print(f" > ENTRY AREA: Rp {rec['price']:,.0f} - Rp {res['support']['pivot_strong']:,.0f}")
            print(f" > STOP LOSS : Rp {rec['stop_loss']:,.0f} (Below Support)")
            print(f" > TARGET    : Rp {rec['target_price']:,.0f} (Take Profit)")
            print("\nLOGIC:")
            for reason in rec['reasons']:
                print(f" - {reason}")
        
        print("\nDisclaimer: This tool is for educational purposes. DYOR.")

    except Exception as e:
        print(f"\n[ERROR] Failed to analyze {args.ticker}: {str(e)}")
        print("Ensure the ticker is valid and internet connection is active.")

if __name__ == "__main__":
    main()