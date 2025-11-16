import pandas as pd
import numpy as np
from scipy.signal import find_peaks

def cluster_levels(levels, cluster_percent=0.02):
    """
    Mengelompokkan level harga yang berdekatan.
    """
    if levels.empty:
        return pd.DataFrame(columns=['Level', 'Hits'])
    
    sorted_levels = levels.sort_values().values
    clusters = []
    
    i = 0
    while i < len(sorted_levels):
        current_cluster_levels = [sorted_levels[i]]
        cluster_base_price = sorted_levels[i]
        
        j = i + 1
        while j < len(sorted_levels):
            level = sorted_levels[j]
            if (level - cluster_base_price) / cluster_base_price <= cluster_percent:
                current_cluster_levels.append(level)
                j += 1
            else:
                break 
        
        cluster_mean = np.mean(current_cluster_levels)
        cluster_hits = len(current_cluster_levels)
        clusters.append({'Level': cluster_mean, 'Hits': cluster_hits})
        i = j 
        
    df_clusters = pd.DataFrame(clusters).sort_values(by='Hits', ascending=False)
    df_clusters['Level'] = df_clusters['Level'].round(0)
    return df_clusters

def find_support_resistance(df):
    """
    Mendeteksi level S/R mentah dan yang sudah di-cluster.
    (Menggunakan kolom lowercase 'high' dan 'low')
    """
    # 1. Cari Puncak (Resistance) dan Lembah (Support)
    #    Tingkatkan 'prominence' ke 1.0 untuk S/R yang lebih akurat (lebih ketat)
    #    Gunakan 'prominence' 0.8 sebagai "jalan tengah" (lebih ketat dari 0.5, tapi tidak seketat 1.0)
    peaks_indices, _ = find_peaks(df['high'], distance=10, prominence=df['high'].std() * 0.8)
    valleys_indices, _ = find_peaks(-df['low'], distance=10, prominence=df['low'].std() * 0.8)

    raw_supports = df['low'].iloc[valleys_indices]
    raw_resistances = df['high'].iloc[peaks_indices]
    
    clustered_supports = cluster_levels(raw_supports, cluster_percent=0.02)
    clustered_resistances = cluster_levels(raw_resistances, cluster_percent=0.02)
    
    return clustered_supports, clustered_resistances, raw_supports, raw_resistances

def detect_market_structure(raw_supports, raw_resistances):
    """
    Menganalisis 2 puncak (High) dan 2 lembah (Low) terakhir untuk menentukan
    market structure (Uptrend, Downtrend, Sideways)
    (VERSI BARU: Lebih simpel dan robust)
    """
    
    # 1. Cek apakah kita punya cukup data mentah
    #    Kita butuh setidaknya 2 puncak dan 2 lembah DARI KESELURUHAN DATA
    if len(raw_resistances) < 2 or len(raw_supports) < 2:
        return "UNKNOWN" # Tidak cukup data untuk perbandingan

    # 2. Ambil 2 puncak (Highs) terakhir
    last_peaks = raw_resistances.iloc[-2:]
    
    # 3. Ambil 2 lembah (Lows) terakhir
    last_valleys = raw_supports.iloc[-2:]

    # 4. Cek Tren (Logika HH/HL dan LL/LH)
    
    # Cek UPTREND (Higher Highs AND Higher Lows)
    is_hh = last_peaks.iloc[1] > last_peaks.iloc[0]
    is_hl = last_valleys.iloc[1] > last_valleys.iloc[0]
    
    if is_hh and is_hl:
        return "UPTREND"
        
    # Cek DOWNTREND (Lower Highs AND Lower Lows)
    is_lh = last_peaks.iloc[1] < last_peaks.iloc[0]
    is_ll = last_valleys.iloc[1] < last_valleys.iloc[0]
    
    if is_lh and is_ll:
        return "DOWNTREND"
    
    # 5. Jika tidak keduanya (misal: HH dan LL, atau LH dan HL), berarti SIDEWAYS/Konsolidasi
    return "SIDEWAYS"