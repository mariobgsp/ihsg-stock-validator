import argparse
import sys
from engine import TradingEngine

def print_header(title):
    print("\n" + "="*50)
    print(f" {title.upper()}")
    print("="*50)

def print_separator():
    print("-" * 50)

def main():
    parser = argparse.ArgumentParser(description="Indonesian Stock Exchange Swing Trading CLI")
    parser.add_argument("ticker", type=str, help="Stock Ticker (e.g., BBCA, ANTM)")
    args = parser.parse_args()

    print(f"\nInitializing Advanced Technical Analysis for {args.ticker}...")
    
    engine = TradingEngine(args.ticker)
    result = engine.analyze()
    
    if not result:
        print("Analysis Failed. Check ticker or internet connection.")
        sys.exit(1)

    d = result['data']
    setup = result['setup']
    probs = result['probs']

    # --- VERDICT SECTION ---
    print_header(f"VERDICT: {result['verdict']}")
    
    if result['verdict'] == "BUY":
        print(f"Strategy Triggered: {result['strategy']}")
        print_separator()
        print("TRADE PLAN (Strict 1:3 RR):")
        print(f"[*] ENTRY PRICE : {setup[0]:,.0f}")
        print(f"[!] STOP LOSS   : {setup[1]:,.0f} (RISK: {setup[0]-setup[1]:,.0f})")
        print(f"[$] TARGET 1 (1R): {setup[2]:,.0f} (Prob: {probs[1]:.1f}%)")
        print(f"[$] TARGET 2 (2R): {setup[3]:,.0f} (Prob: {probs[2]:.1f}%)")
        print(f"[$] TARGET 3 (3R): {setup[4]:,.0f} (Prob: {probs[3]:.1f}%)")
        
        print_separator()
        print("THE LOGIC:")
        print(f"Triggered by {result['strategy']}.")
        print(f"Price is {d['vwap_diff']:.2f}% relative to VWAP.")
        if d['vcp']: print("Bonus: Valid VCP Pattern detected (Supply drying up).")
        if d['squeeze']: print("Bonus: MA Squeeze detected (Energy building).")
        print(f"Smart Money is in {d['sm_status']} phase.")

        print_separator()
        print(f"SAFETY SCORE / HISTORICAL WIN RATE: {result['win_rate']:.1f}%")
        if result['win_rate'] < 60:
            print("(WARNING: Win rate is low. Strict position sizing recommended.)")
    
    else:
        print("No high-probability setup found (>60% WR).")
        print("Recommendation: Sit on hands. Cash is a position.")

    # --- DATA SECTION ---
    print("\n")
    print("="*50)
    print(" DEEP DATA ANALYTICS")
    print("="*50)
    
    print(f"Asset Status   : {d['is_ipo']}")
    print(f"Current Price  : {d['price']:,.0f}")
    print_separator()
    
    print(f"SMART MONEY    : {d['sm_status']}")
    print(f"Phase Start    : {d['sm_date']}")
    print(f"VWAP Context   : {d['vwap_diff']:+.2f}% vs Price")
    print_separator()
    
    print("KEY LEVELS:")
    print(f"Pivot Point    : {d['pivot']:,.0f}")
    print(f"Bounce Support : {d['last_bounce']:,.0f}")
    print(f"Fib 0.618      : {d['fib_618']:,.0f}")
    print_separator()
    
    print("INDICATOR OPTIMIZATION:")
    print(f"Used Params    : RSI {d['params']['rsi']}, MA Fast {d['params']['ma_short']}, MA Slow {d['params']['ma_long']}")
    print(f"RSI Value      : {d['rsi']:.2f}")
    print(f"Stochastic K   : {d['stoch']:.2f}")
    print_separator()
    
    print("PATTERN RECOGNITION:")
    print(f"VCP Pattern    : {'DETECTED' if d['vcp'] else 'None'}")
    print(f"Super Squeeze  : {'ACTIVE' if d['squeeze'] else 'Inactive'}")
    print("="*50 + "\n")

if __name__ == "__main__":
    main()

