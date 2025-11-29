import os
import sys
import argparse
from engine import StockAnalyzer, DEFAULT_CONFIG

def clear_screen():
    if sys.stdout.isatty():
        os.system('cls' if os.name == 'nt' else 'clear')

def print_header():
    print("="*65)
    print("      IHSG ULTIMATE SCANNER (V3.1 - Mega Features)      ")
    print("      Trend + VCP + Rectangles + Smart Money + Fundamentals")
    print("="*65)

def print_report(data, balance):
    if not data:
        print("\n[!] Error: Could not fetch data or invalid ticker.")
        return

    print(f"\nSTOCK: {data['name']} ({data['ticker']})")
    print(f"PRICE: Rp {data['price']:,.0f}")

    # 1. LIQUIDITY
    liq = data['liquidity']
    symbol = "✅" if liq['status'] == "PASS" else "⚠️ "
    print(f"\n{symbol} LIQUIDITY CHECK")
    print(f"   {liq['msg']}")

    # 2. FUNDAMENTALS
    fund = data['context'].get('fundamental', {})
    print(f"\n📊 FUNDAMENTALS")
    print(f"   Market Cap: Rp {fund.get('market_cap', 0):,.0f}")
    if fund.get('warning'): print(f"   ⚠️ {fund['warning']}")

    # 3. TREND HEALTH
    tt = data['trend_template']
    symbol = "✅" if "UPTREND" in tt['status'] else "⚠️ "
    print(f"\n{symbol} TREND HEALTH")
    print(f"   Status: {tt['status']} (Score: {tt['score']}/6)")
    for det in tt['details']: print(f"   - {det}")

    # 4. SMART MONEY
    sm = data['context']['smart_money']
    symbol = "✅" if "BULLISH" in sm['status'] else "⚠️ " if "BEARISH" in sm['status'] else "🔹"
    print(f"\n{symbol} SMART MONEY (Bandarmology Proxy)")
    print(f"   Status: {sm['status']}")
    for s in sm['signals']: print(f"   - {s}")

    # 5. PATTERNS
    print(f"\n💎 PATTERN RECOGNITION")
    
    # Rectangle
    rect = data['rectangle']
    if rect['detected']:
        print(f"   [RECTANGLE] {rect['status']} (Range: {rect['bottom']:,.0f}-{rect['top']:,.0f})")
    
    # VCP
    vcp = data['context']['vcp']
    if vcp['detected']: print(f"   [VCP] {vcp['msg']}")
    
    # Geo
    geo = data['context']['geo']
    if geo['pattern'] != "None": print(f"   [GEO] {geo['pattern']} {geo['msg']}")
    
    # Squeeze
    sqz = data['context']['squeeze']
    if sqz.get('detected'): print(f"   [TTM SQUEEZE] {sqz['msg']}")

    # 6. TRADE PLAN
    plan = data['plan']
    print(f"\n🚀 TRADE PLAN (3R)")
    
    if "WAIT" in plan['status']:
        print(f"   Action: {plan['status']}")
        print(f"   Reason: {plan['reason']}")
    else:
        print(f"   ACTION:      BUY ({plan['status']})")
        print(f"   WHY?:        {plan['reason']}")
        print(f"   ENTRY:       Rp {plan['entry']:,.0f}")
        print(f"   STOP LOSS:   Rp {plan['stop_loss']:,.0f}")
        print(f"   TARGET (3R): Rp {plan['take_profit']:,.0f}")
        
        print(f"\n   --- POSITION SIZING (Bal: {balance/1e6:.0f} Jt) ---")
        if plan.get('lots', 0) > 0:
            print(f"   🛒 BUY:      {plan['lots']} LOTS")
            print(f"   💰 Capital:  Rp {plan['lots'] * 100 * plan['entry']:,.0f}")
            print(f"   🔥 Risk:     Rp {plan['risk_amt']:,.0f}")
        else:
            print("   [!] Stop Loss too tight or risk too high.")

    # 7. FIBONACCI
    print(f"\n📐 KEY LEVELS (Fibs)")
    fibs = data['context']['fib_levels']
    if fibs:
        print(f"   Golden (0.618): Rp {fibs.get('0.618 (Golden)', 0):,.0f}")
        print(f"   Low (Support):  Rp {fibs.get('1.0 (Low)', 0):,.0f}")

    # 8. NEWS
    print(f"\n📰 SENTIMENT")
    s = data['sentiment']
    print(f"   Score: {s['score']} ({s['sentiment']})")
    for hl in s['headlines']: print(f"   - {hl}")

    print("\n" + "="*65)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('ticker', nargs='?')
    
    # Sizing Args
    parser.add_argument('--balance', type=int, default=100_000_000, help="Account Balance (IDR)")
    parser.add_argument('--risk', type=float, default=1.0, help="Risk per trade (%)")
    
    # Technical Args (RESTORED)
    parser.add_argument('--period', default=DEFAULT_CONFIG['BACKTEST_PERIOD'], help="Data period (e.g. 1y, 2y)")
    parser.add_argument('--sl', type=float, default=DEFAULT_CONFIG['SL_MULTIPLIER'], help="Stop Loss ATR Multiplier")
    parser.add_argument('--tp', type=float, default=DEFAULT_CONFIG['TP_MULTIPLIER'], help="Take Profit ATR Multiplier")
    parser.add_argument('--fib', type=int, default=DEFAULT_CONFIG['FIB_LOOKBACK_DAYS'], help="Fibonacci Lookback Days")
    
    args = parser.parse_args()

    if not args.ticker:
        clear_screen()
        print_header()
        args.ticker = input("\nEnter Ticker (e.g. BBCA): ").strip()

    # Map args to config dictionary
    user_config = {
        "ACCOUNT_BALANCE": args.balance,
        "RISK_PER_TRADE_PCT": args.risk,
        "BACKTEST_PERIOD": args.period,
        "SL_MULTIPLIER": args.sl,
        "TP_MULTIPLIER": args.tp,
        "FIB_LOOKBACK_DAYS": args.fib
    }

    print(f"\nRunning Mega Analysis on {args.ticker}...")
    analyzer = StockAnalyzer(args.ticker, user_config)
    print_report(analyzer.generate_final_report(), args.balance)

if __name__ == "__main__":
    main()
