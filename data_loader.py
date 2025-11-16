import pandas as pd
import pandas_ta as ta

def get_stock_data_from_csv(filepath):
    """
    Membaca data saham dari file CSV dan menghitung indikator.
    (VERSI DIPERBAIKI: Mengganti nama 'Price' -> 'Date' secara eksplisit)
    """
    print(f"Membaca data dari {filepath}...")
    try:
        # PENTING: Disesuaikan untuk file spesifik Anda
        # header=0 -> Baris pertama adalah header
        # skiprows=[1, 2] -> Lewati baris ke-2 dan ke-3 (indeks 1 dan 2)
        df = pd.read_csv(filepath, header=0, skiprows=[1, 2])
    except FileNotFoundError:
        print(f"Error: File tidak ditemukan di {filepath}")
        return None
    except Exception as e:
        print(f"Error saat membaca CSV: {e}")
        return None

    if df.empty:
        print("Error: File CSV kosong.")
        return None

    # --- Standarisasi Data (LOGIKA BARU YANG LEBIH BERSIH) ---
    
    # 1. Ganti nama 'Price' menjadi 'Date' (Sesuai permintaan Anda)
    #    Ini untuk menangani format CSV Anda yang unik.
    if 'Price' in df.columns:
        df.rename(columns={'Price': 'Date'}, inplace=True)
        print("Catatan: Kolom 'Price' telah diganti namanya menjadi 'Date'.")

    # 2. Cari kolom Tanggal (Date) dan jadikan index
    date_col = None
    if 'Date' in df.columns:
        date_col = 'Date'
    elif 'Tanggal' in df.columns:
        date_col = 'Tanggal'
    
    # Jika tidak ada 'Date' atau 'Tanggal' (setelah di-rename), baru error
    if not date_col:
        print(f"Error: CSV harus punya kolom 'Date' atau 'Tanggal'. Kolom ditemukan: {df.columns.values}")
        return None
        
    try:
        # Konversi ke datetime dan set sebagai index
        df[date_col] = pd.to_datetime(df[date_col], format='%Y-%m-%d')
        df.set_index(date_col, inplace=True)
        # Logika 'drop' yang membingungkan sudah tidak perlu lagi
    except Exception as e:
        print(f"Error saat memproses kolom tanggal '{date_col}': {e}")
        return None

    # --- Sisa file ini sama persis ---

    # 3. Cek kolom wajib (Open, High, Low, Close, Volume)
    required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    if not all(col in df.columns for col in required_cols):
        print(f"Error: CSV harus memiliki kolom 'Open', 'High', 'Low', 'Close', 'Volume'.")
        print(f"Kolom yang ditemukan: {df.columns.values}")
        return None
    
    # 4. Konversi kolom OHLCV ke numerik (PENTING untuk CSV)
    for col in required_cols:
         df[col] = pd.to_numeric(df[col], errors='coerce')

    df.dropna(subset=required_cols, inplace=True)
    
    # --- Perhitungan Indikator ---
    try:
        # Gunakan 'close=df['Close']' secara eksplisit karena nama kolom
        df.ta.rsi(close=df['Close'], length=14, append=True)
        df.ta.macd(close=df['Close'], fast=12, slow=26, signal=9, append=True)
        df.ta.bbands(close=df['Close'], length=20, std=2, append=True)
        df.ta.stoch(high=df['High'], low=df['Low'], close=df['Close'], k=10, d=3, smooth_k=3, append=True)
        
        # Moving Averages
        df.ta.sma(close=df['Close'], length=10, append=True)
        df.ta.sma(close=df['Close'], length=20, append=True)
        df.ta.sma(close=df['Close'], length=50, append=True)
        df.ta.sma(close=df['Close'], length=100, append=True)
        df.ta.sma(close=df['Close'], length=200, append=True)
        
        df['volume_ma20'] = df['Volume'].rolling(window=20).mean()

        # Ubah semua NAMA KOLOM jadi lowercase (Termasuk 'SMA_10' -> 'sma_10')
        df.columns = df.columns.str.lower()
        
        print("Data dan indikator berhasil dihitung.")
        
    except Exception as e:
        print(f"Error saat menghitung indikator: {e}. Periksa nama kolom.")
        return None

    # Hapus baris di awal yang punya NaN (karena MA, dll)
    # df = df.dropna()
    return df