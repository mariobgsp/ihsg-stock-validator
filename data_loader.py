import pandas as pd
import pandas_ta as ta
import yfinance as yf

def _calculate_indicators(df):
    """
    Shared logic to calculate all indicators for both CSV and YFinance data.
    """
    try:
        # 1. Basic Indicators
        df.ta.rsi(close=df['Close'], length=14, append=True)
        df.ta.macd(close=df['Close'], fast=12, slow=26, signal=9, append=True)
        
        # --- RESTORED STOCHASTIC ---
        df.ta.stoch(high=df['High'], low=df['Low'], close=df['Close'], k=10, d=3, smooth_k=3, append=True)
        
        # Moving Averages
        df.ta.sma(close=df['Close'], length=50, append=True)
        df.ta.sma(close=df['Close'], length=200, append=True)
        df['volume_ma20'] = df['Volume'].rolling(window=20).mean()

        # 2. Bollinger Bands (Explicit Rename)
        bb = df.ta.bbands(close=df['Close'], length=20, std=2)
        if bb is not None:
            lower_col = [c for c in bb.columns if c.startswith("BBL")][0]
            upper_col = [c for c in bb.columns if c.startswith("BBU")][0]
            df['bb_lower'] = bb[lower_col]
            df['bb_upper'] = bb[upper_col]

        # 3. Supertrend (Explicit Rename)
        st = df.ta.supertrend(high=df['High'], low=df['Low'], close=df['Close'], length=10, multiplier=3)
        if st is not None:
            line_col = [c for c in st.columns if c.startswith("SUPERT_")][0]
            df['supertrend'] = st[line_col]

        # 4. 5-Dimensional Components
        df['atrr_14'] = df.ta.atr(high=df['High'], low=df['Low'], close=df['Close'], length=14)
        
        adx = df.ta.adx(high=df['High'], low=df['Low'], close=df['Close'], length=14)
        if not adx.empty: df['adx_14'] = adx.iloc[:, 0]

        df['rvol'] = df['Volume'] / df['volume_ma20']
        
        # 5. Candle Patterns
        df.ta.cdl_pattern(open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], 
                          name=["hammer", "engulfing"], append=True)

        # Final Cleanup
        df.columns = df.columns.str.lower()
        df = df.loc[:, ~df.columns.duplicated()]
        
        return df
        
    except Exception as e:
        print(f"Error calculating indicators: {e}")
        return None

def get_stock_data_from_csv(filepath):
    """ Load from CSV file. """
    print(f"Membaca data dari {filepath}...")
    try:
        df = pd.read_csv(filepath, header=0, skiprows=[1, 2])
    except FileNotFoundError:
        return None
    except Exception as e:
        print(f"Error CSV: {e}")
        return None

    if df.empty: return None

    if 'Price' in df.columns: df.rename(columns={'Price': 'Date'}, inplace=True)
    
    date_col = None
    if 'Date' in df.columns: date_col = 'Date'
    elif 'Tanggal' in df.columns: date_col = 'Tanggal'
    
    if not date_col: return None
        
    try:
        df[date_col] = pd.to_datetime(df[date_col], format='%Y-%m-%d')
        df.set_index(date_col, inplace=True)
    except: return None

    required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    df.columns = [c.title() if c.lower() in [x.lower() for x in required_cols] else c for c in df.columns]
    
    for col in required_cols:
        if col not in df.columns: return None
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    df.dropna(subset=required_cols, inplace=True)
    
    return _calculate_indicators(df)

def get_stock_data_from_yfinance(ticker, period="2y", interval="1d"):
    """
    Load data directly from Yahoo Finance.
    """
    print(f"Fetching {ticker} from YFinance...")
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(period=period, interval=interval)
        
        if df.empty:
            print(f"No data found for {ticker}")
            return None
            
        # YFinance returns columns: Open, High, Low, Close, Volume, Dividends, Stock Splits
        # We need to ensure standard format
        df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
        
        # Ensure index is datetime (it usually is timezone aware, let's remove tz for simplicity)
        df.index = df.index.tz_localize(None)
        df.index.name = 'Date'
        
        return _calculate_indicators(df)
        
    except Exception as e:
        print(f"YFinance Error: {e}")
        return None