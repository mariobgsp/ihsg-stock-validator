import os
import sys
import argparse
from engine import StockAnalyzer, DEFAULT_CONFIG

def clear_screen():
    if sys.stdout.isatty():
        os.system('cls' if os.name == 'nt' else 'clear')

def print_report(data, balance):
    if not data:
        print("Error: Could not fetch data.")
        return

    print("\n" + "="*60)
    print(f" REPORT: {data['name']} ({data['ticker']})")
    print(f" PRICE:  Rp {data['price']:,.0f}")
    print("="*60)

    # 1. LIQUIDITY
    liq = data['liquidity']
    symbol = "✅" if liq['status'] == "PASS" else "⚠️ "
    print(f"\n{symbol} LIQUIDITY")
    print(f"   {liq['msg']}")

    # 2. TREND & SMART MONEY
    tt = data['trend']
    sm = data['smart_money']
    print(f"\n✅ CONTEXT")
    print(f"   Trend: {tt['status']} (Score: {tt['score']}/6)")
    print(f"   Flow:  {sm['status']}")
    [attachment_0](attachment)

    # 3. PATTERNS
    rect = data['rectangle']
    if rect['detected']:
        print(f"\n💎 PATTERN: RECTANGLE")
        print(f"   State:  {rect['status']}")
        print(f"   Range:  Rp {rect['bottom']:,.0f} - Rp {rect['top']:,.0f}")
        [attachment_1](attachment)
    else:
        print(f"\n🔹 PATTERNS")
        print("   No distinct box detected.")

    # 4. HISTORICAL BACKTEST (New Section)
    bt = data['backtest']
    print(f"\n📜 HISTORICAL PERFORMANCE (2 Years)")
    print(f"   Strategy: Box Breakout + Trend + Smart Money")
    if bt['count'] > 0:
        win_symbol = "🔥" if bt['win_rate'] > 50 else "❄️"
        print(f"   Trades:   {bt['count']}")
        print(f"   Win Rate: {bt['win_rate']:.1f}% {win_symbol}")
        print(f"   Net PnL:  {bt['total_return']:.1f}% (Without compounding)")
    else:
        print(f"   No historical trades found for this strategy.")

    # 5. TRADE PLAN
    plan = data['plan']
    print(f"\n🚀 TRADE PLAN")
    
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
        if plan['lots'] > 0:
            print(f"   🛒 Buy:      {plan['lots']} LOTS")
            print(f"   💰 Capital:  Rp {plan['lots'] * 100 * plan['entry']:,.0f}")
        else:
            print("   [!] Risk too high or Stop too close.")

    # 6. SENTIMENT
    print(f"\n📰 SENTIMENT")
    s = data['sentiment']
    print(f"   Score: {s['score']} ({s['sentiment']})")
    
    print("\n" + "="*60 + "\n")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('ticker', nargs='?')
    parser.add_argument('--balance', type=int, default=100_000_000)
    parser.add_argument('--risk', type=float, default=1.0)
    args = parser.parse_args()

    if not args.ticker:
        clear_screen()
        args.ticker = input("Enter Ticker (e.g., ADRO): ").strip()

    config = DEFAULT_CONFIG.copy()
    config["ACCOUNT_BALANCE"] = args.balance
    config["RISK_PER_TRADE_PCT"] = args.risk

    analyzer = StockAnalyzer(args.ticker, config)
    print(f"\nAnalyzing {args.ticker}...")
    print_report(analyzer.generate_report(), args.balance)

if __name__ == "__main__":
    main()
