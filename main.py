import argparse
import sys
import time
from engine import AlphaEngine

def print_header():
    print("\n" + "="*60)
    print("      INDONESIA STOCK EXCHANGE (IDX) - QUANT TRADING CLI      ")
    print("         Engine: Python 3 | Compliance: OJK Rules             ")
    print("="*60 + "\n")

def loading_animation(text):
    chars = "/—\|" 
    for i in range(10):
        time.sleep(0.1)
        sys.stdout.write(f'\r{text} {chars[i % len(chars)]}')
        sys.stdout.flush()
    sys.stdout.write('\r' + ' '*len(text) + '  \r')

def format_currency(value):
    return f"Rp {int(value):,}".replace(",", ".")

def main():
    parser = argparse.ArgumentParser(description="IDX Swing Trading Alpha Engine")
    parser.add_argument("ticker", type=str, help="Stock Ticker (e.g., BBRI, TLKM)")
    args = parser.parse_args()
    
    ticker = args.ticker.upper()
    print_header()
    
    try:
        loading_animation(f"[*] Connecting to yfinance for {ticker}...")
        engine = AlphaEngine(ticker)
        
        loading_animation(f"[*] Running Grid Search Optimization (Strategies, MAs, RSI)...")
        result = engine.optimize_and_run()
        
        # --- DASHBOARD OUTPUT ---
        
        # 1. Executive Summary
        print(f"REPORT FOR: {ticker}")
        print(f"{'-'*60}")
        
        color = "\033[92m" if result.action == "BUY" else "\033[91m" if result.action == "NO TRADE" else "\033[93m"
        reset = "\033[0m"
        
        print(f"RECOMMENDATION : {color}{result.action}{reset}")
        print(f"STRATEGY       : {result.backtest_stats.strategy_name} (Optimized)")
        print(f"REASON         : {result.reason}")
        print(f"ENTRY PRICE    : {format_currency(result.entry_price)}")
        print(f"STOP LOSS      : {format_currency(result.stop_loss)} (OJK Tick Aligned)")
        print(f"TARGET (3R)    : {format_currency(result.target_price)}")
        print(f"\n")
        
        # 2. Probability Analysis
        print(f"[PROBABILITY MATRIX]")
        print(f"Backtested Win Rate : {result.backtest_stats.win_rate:.1f}%")
        print(f"Prob to Hit 1R      : {result.probabilities['1R']:.1f}%")
        print(f"Prob to Hit 2R      : {result.probabilities['2R']:.1f}%")
        print(f"Prob to Hit 3R      : {result.probabilities['3R']:.1f}%")
        print(f"\n")
        
        # 3. Smart Money / Bandarmology
        sm = result.technical_data['Smart_Money']
        sm_color = "\033[92m" if sm['status'] == "Accumulation" else "\033[91m"
        print(f"[SMART MONEY FLOW]")
        print(f"Status              : {sm_color}{sm['status']}{reset}")
        print(f"Phase Started On    : {sm['start_date']}")
        print(f"Flow Strength       : {sm['strength']}")
        print(f"\n")
        
        # 4. Technical Deep Dive
        tech = result.technical_data
        print(f"[TECHNICAL METRICS]")
        print(f"Price vs VWAP       : {'Above' if tech['Current_Price'] > tech['VWAP'] else 'Below'} ({format_currency(tech['VWAP'])})")
        print(f"Optimized Ind.      : RSI-{tech['Indicators']['rsi']} | MA-Fast {tech['Indicators']['ma_fast']}")
        print(f"Current RSI         : {tech['RSI_Value']}")
        print(f"VCP Detected        : {tech['Patterns']['VCP']}")
        print(f"MA Squeeze          : {tech['Patterns']['MA_Squeeze']}")
        print(f"\n")
        
        # 5. Key Levels
        levels = tech['Levels']
        print(f"[CRITICAL LEVELS]")
        print(f"Pivot Support       : {format_currency(levels['Pivot_Support'])}")
        print(f"Last Bounce Zone    : {format_currency(levels['Bounce_Zone'])}")
        print(f"Golden Fib (61.8%)  : {format_currency(levels['Fib_Golden_Zone'])}")
        print(f"\n")
        
        # 6. Backtest Validation
        stats = result.backtest_stats
        print(f"[BACKTEST VALIDATION (3 Years)]")
        print(f"Total Trades        : {stats.total_trades}")
        print(f"Max Drawdown        : {stats.max_drawdown:.2f}%")
        print(f"Avg Hold Time       : {stats.avg_candles_held} Days")
        
        if not stats.is_valid:
            print(f"\n\033[91mWARNING: Best strategy Win Rate is below 65% threshold. Trade with caution.\033[0m")
            
        print("="*60)

    except Exception as e:
        print(f"\n\033[91mERROR: {str(e)}\033[0m")

if __name__ == "__main__":
    main()