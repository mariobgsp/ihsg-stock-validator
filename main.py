import sys
import argparse
from engine import QuantEngine

def print_header(title):
    print("\n" + "=" * 60)
    print(f"{title:^60}")
    print("=" * 60)

def print_separator():
    print("-" * 60)

def format_currency(val):
    return f"Rp {int(val):,}"

def main():
    parser = argparse.ArgumentParser(description="IDX Swing Trading Quant CLI")
    parser.add_argument('ticker', type=str, help='Stock Ticker (e.g., BBCA, TLKM)')
    args = parser.parse_args()

    ticker = args.ticker.upper()
    print_header(f"ANALYZING {ticker} - PLEASE WAIT...")
    
    try:
        engine = QuantEngine(ticker)
        data = engine.run_strategy()
    except Exception as e:
        print(f"\n[ERROR] Failed to analyze {ticker}: {str(e)}")
        print("Tip: Check internet connection or if ticker exists on Yahoo Finance.")
        return

    # --- VERDICT SECTION ---
    print_header("STRATEGY VERDICT")
    
    v_color = " [!]" 
    if data['verdict'] == "BUY": v_color = " [>>>]"
    
    print(f"\nVERDICT: {v_color} {data['verdict']} {v_color}")
    
    if data['verdict'] != "NO TRADE":
        print("\n--- TRADE PLAN (OJK Compliant) ---")
        entry = data['plan']['entry']
        sl = data['plan']['sl']
        risk_pct = ((entry - sl) / entry) * 100
        
        print(f"ENTRY PRICE : {format_currency(entry)}")
        print(f"STOP LOSS   : {format_currency(sl)} (-{risk_pct:.2f}%)")
        print_separator()
        print(f"TARGET 1 (1R): {format_currency(data['plan']['tp1'])}")
        print(f"TARGET 2 (2R): {format_currency(data['plan']['tp2'])}")
        print(f"TARGET 3 (3R): {format_currency(data['plan']['tp3'])}")
    
    print("\n--- THE LOGIC ---")
    for reason in data['reasons']:
        print(f"* {reason}")
    
    vwap_status = "ABOVE" if data['smart_money']['vwap_diff'] > 0 else "BELOW"
    print(f"* Price is {abs(data['smart_money']['vwap_diff']):.2f}% {vwap_status} Smart Money VWAP")

    print("\n--- SAFETY SCORE ---")
    print(f"Probability to hit 1R: {data['probabilities']['1R']:.1f}%")
    print(f"Probability to hit 2R: {data['probabilities']['2R']:.1f}%")
    print(f"Probability to hit 3R: {data['probabilities']['3R']:.1f}%")

    # --- DATA SECTION ---
    print_header("DEEP DIVE DATA")
    
    # Asset Status
    status_type = "IPO / NEW LISTING" if data['is_ipo'] else "MATURE STOCK"
    print(f"ASSET CLASS   : {status_type}")
    
    # Smart Money
    sm = data['smart_money']
    print(f"BANDARMOLOGY  : {sm['status']}")
    print(f"FLOW STRENGTH : {sm['strength']:.4f} (Slope)")
    print(f"PHASE START   : {sm['start_date']}")
    
    # Key Levels
    print_separator()
    print("KEY LEVELS & BOUNCE ZONES:")
    if data['patterns']['bounce_levels']:
        levels = [format_currency(x) for x in data['patterns']['bounce_levels']]
        print(f"Hist. Support : {', '.join(levels)}")
    else:
        print("Hist. Support : Calculating...")

    # Pattern Flags
    print_separator()
    print("PATTERN RECOGNITION:")
    print(f"VCP Pattern   : {'DETECTED' if data['patterns']['is_vcp'] else 'None'}")
    print(f"MA Squeeze    : {'YES' if data['patterns']['is_squeeze'] else 'No'} ({data['patterns']['squeeze_pct']:.2f}% width)")

    # Indicator Details
    print_separator()
    params = data['params']
    print("OPTIMIZED SETTINGS (Grid Search):")
    print(f"MA Fast/Slow  : {params['ma_fast']} / {params['ma_slow']}")
    print(f"RSI Period    : {params['rsi_period']}")
    
    print("\n" + "="*60 + "\n")

if __name__ == "__main__":
    main()

