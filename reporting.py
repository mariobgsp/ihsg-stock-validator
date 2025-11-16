import pandas as pd

def scan_for_signals(df, clustered_supports, clustered_resistances):
    """
    Memindai seluruh riwayat data (timeframe) untuk menemukan 
    dan melabeli sinyal trading spesifik (TERMASUK Reversal & Buy The Dip).
    """
    print("\n--- (DETECTOR) Memulai Pemindaian Sinyal Historis ---")
    
    signals_found = []
    support_levels = clustered_supports['Level'].values
    resistance_levels = clustered_resistances['Level'].values
    proximity = 0.02 # Toleransi 2%
    
    if len(df) < 201:
        print("Data tidak cukup untuk pemindaian MA 200.")
        return pd.DataFrame() 

    for i in range(200, len(df)):
        bar = df.iloc[i]
        prev_bar = df.iloc[i-1]
        
        # --- KUMPULKAN STATUS INDIKATOR (BULLISH) ---
        rsi_val = bar['rsi_14']
        rsi_bullish_cross = prev_bar['rsi_14'] < 30 and rsi_val > 30
        
        macd_val = bar['macd_12_26_9']
        macds_val = bar['macds_12_26_9']
        macd_bullish_cross = prev_bar['macd_12_26_9'] < prev_bar['macds_12_26_9'] and macd_val > macds_val

        stochk_val = bar['stochk_10_3_3']
        stochd_val = bar['stochd_10_3_3']
        stoch_bullish_cross = prev_bar['stochk_10_3_3'] < prev_bar['stochd_10_3_3'] and stochk_val > stochd_val
        stoch_leaving_oversold = stoch_bullish_cross and prev_bar['stochk_10_3_3'] < 30 
        
        # --- KUMPULKAN STATUS INDIKATOR (BEARISH) ---
        rsi_bearish_cross = prev_bar['rsi_14'] > 70 and rsi_val < 70
        stoch_bearish_cross = prev_bar['stochk_10_3_3'] > prev_bar['stochd_10_3_3'] and stochk_val < stochd_val
        stoch_leaving_overbought = stoch_bearish_cross and prev_bar['stochk_10_3_3'] > 70
        
        # --- SISA INDIKATOR ---
        volume_breakout = bar['volume'] > (bar['volume_ma20'] * 1.5)
        ma50_val = bar['sma_50']
        ma200_val = bar['sma_200']
        ma50_prev = prev_bar['sma_50']
        ma200_prev = prev_bar['sma_200']
        
        is_golden_cross = ma50_prev < ma200_prev and ma50_val > ma200_val
        is_death_cross = ma50_prev > ma200_prev and ma50_val < ma200_val
        
        # --- (LOGIKA BARU) FILTER TREN UTAMA ---
        is_main_uptrend = bar['close'] > ma200_val
        
        # --- (LOGIKA BARU) Pola Reversal Candlestick ---
        # Pola Bearish Engulfing (Reversal Jual)
        is_bearish_engulfing = (prev_bar['close'] > prev_bar['open']) and \
                               (bar['close'] < bar['open']) and \
                               (bar['close'] < prev_bar['open']) and \
                               (bar['open'] > prev_bar['close'])
        
        # Pola Bullish Engulfing (Reversal Beli)
        is_bullish_engulfing = (prev_bar['close'] < prev_bar['open']) and \
                               (bar['close'] > bar['open']) and \
                               (bar['close'] > prev_bar['open']) and \
                               (bar['open'] < prev_bar['close'])
        
        # --- LOGIKA PENCARIAN & PELABELAN "SPOT" ---
        
        # 1. SINYAL TREN (Golden/Death Cross) - Selalu dicari
        if is_golden_cross:
            signals_found.append({
                "Tanggal": bar.name, "Sinyal (Label)": "TREND - GOLDEN CROSS",
                "Harga": bar['close'], "Detail": "MA 50 memotong ke atas MA 200"
            })
            continue 
        
        if is_death_cross:
            signals_found.append({
                "Tanggal": bar.name, "Sinyal (Label)": "TREND - DEATH CROSS",
                "Harga": bar['close'], "Detail": "MA 50 memotong ke bawah MA 200"
            })
            continue 
            
        # 2. (BARU) SINYAL REVERSAL (Engulfing) - Prioritas tertinggi setelah Cross
        # Cek Bullish Engulfing di Support
        if is_bullish_engulfing:
            for sup_level in support_levels:
                is_near_support = abs(bar['low'] - sup_level) / sup_level < proximity
                if is_near_support:
                    signals_found.append({
                        "Tanggal": bar.name, "Sinyal (Label)": "REVERSAL - BULLISH ENGULFING",
                        "Harga": bar['close'], "Detail": f"Pola Engulfing Bullish di S {sup_level:.0f}"
                    })
                    break # Hentikan loop sup_level
            if signals_found and signals_found[-1]["Tanggal"] == bar.name: continue # Jika sinyal ditemukan, lanjut ke hari berikutnya
            
        # Cek Bearish Engulfing di Resistance
        if is_bearish_engulfing:
            for res_level in resistance_levels:
                is_near_resistance = abs(bar['high'] - res_level) / res_level < proximity
                if is_near_resistance:
                    signals_found.append({
                        "Tanggal": bar.name, "Sinyal (Label)": "REVERSAL - BEARISH ENGULFING",
                        "Harga": bar['close'], "Detail": f"Pola Engulfing Bearish di R {res_level:.0f}"
                    })
                    break # Hentikan loop res_level
            if signals_found and signals_found[-1]["Tanggal"] == bar.name: continue # Jika sinyal ditemukan, lanjut ke hari berikutnya

        
        # 3. SINYAL FILTER TREN (Hanya searah tren)
        if is_main_uptrend:
            
            # (BARU) SINYAL "BUY THE DIP" (Pantulan MA 50)
            is_dip_buy = (bar['low'] < ma50_val) and (bar['close'] > ma50_val) and (bar['close'] > bar['open'])
            if is_dip_buy:
                signals_found.append({
                    "Tanggal": bar.name, "Sinyal (Label)": "BUY THE DIP (MA 50)",
                    "Harga": bar['close'], "Detail": f"Harga memantul dari SMA 50"
                })
                continue
                
            # SINYAL "ALL VALID CROSSOVER" (BUY)
            if rsi_bullish_cross and macd_bullish_cross and stoch_leaving_oversold and volume_breakout:
                signals_found.append({
                    "Tanggal": bar.name, "Sinyal (Label)": "BUY - ALL VALID CROSSOVER",
                    "Harga": bar['close'], "Detail": "RSI keluar OS, MACD Cross, Stoch Cross, Vol Spike"
                })
                continue 
            
            # SINYAL "BREAKOUT RESISTANCE" (BUY)
            for res_level in resistance_levels:
                is_breakout = prev_bar['close'] < res_level and bar['close'] > res_level
                
                # --- FILTER TAMBAHAN (PENTING) ---
                is_not_rsi_overbought = rsi_val < 70 
                is_not_stoch_overbought = stochk_val < 80 # Filter baru!

                if is_breakout and volume_breakout and is_not_rsi_overbought and is_not_stoch_overbought:
                    signals_found.append({
                        "Tanggal": bar.name, "Sinyal (Label)": "BREAKOUT - RESISTANCE",
                        "Harga": bar['close'], 
                        "Detail": f"Tembus Res {res_level:.0f} (Vol Kuat, RSI < 70, Stoch < 80)"
                    })
                    break
            
            # SINYAL "BOUNCE / RbS FLIP" (BUY)
            for sup_level in support_levels:
                is_near_support = abs(bar['low'] - sup_level) / sup_level < proximity
                is_bounce = bar['close'] > bar['open'] 
                is_confirmed_buy = rsi_bullish_cross or stoch_leaving_oversold
                
                if is_near_support and is_bounce and is_confirmed_buy:
                    is_rbs_flip = (abs(clustered_resistances['Level'] - sup_level) / sup_level < proximity).any()
                    signal_label = "BOUNCE - RbS FLIP" if is_rbs_flip else "BOUNCE - SUPPORT"
                    detail = f"Pantulan dari Zona Flip {sup_level:.0f}" if is_rbs_flip else f"Pantulan dari Sup {sup_level:.0f}"
                    
                    signals_found.append({
                        "Tanggal": bar.name, "Sinyal (Label)": signal_label, "Harga": bar['close'], "Detail": detail
                    })
                    break 

        else: # (if not is_main_uptrend) -> Tren Turun
            
            # SINYAL "REJECTION / SbR FLIP" (SELL)
            for res_level in resistance_levels:
                is_near_resistance = abs(bar['high'] - res_level) / res_level < proximity
                is_rejection = bar['close'] < bar['open'] 
                is_confirmed_sell = rsi_bearish_cross or stoch_leaving_overbought
                
                if is_near_resistance and is_rejection and is_confirmed_sell:
                    is_sbr_flip = (abs(clustered_supports['Level'] - res_level) / res_level < proximity).any()
                    signal_label = "REJECTION - SbR FLIP" if is_sbr_flip else "REJECTION - RESISTANCE"
                    detail = f"Ditolak dari Zona Flip {res_level:.0f}" if is_sbr_flip else f"Ditolak dari Res {res_level:.0f}"

                    signals_found.append({
                        "Tanggal": bar.name, "Sinyal (Label)": signal_label, "Harga": bar['close'], "Detail": detail
                    })
                    break

    # --- Cetak Hasil Pemindaian ---
    if not signals_found:
        print("Tidak ditemukan sinyal historis signifikan yang sesuai kriteria.")
        return pd.DataFrame()
    else:
        df_signals = pd.DataFrame(signals_found).set_index('Tanggal')
        print(f"Ditemukan total {len(df_signals)} 'spot' historis berlabel (sudah difilter tren):")
        print(df_signals.to_string())
        
        return df_signals

def analyze_behavior(df, clustered_supports, clustered_resistances, raw_supports, raw_resistances, market_structure):
    """
    Menganalisis perilaku HARI TERAKHIR dan MENGEMBALIKAN rangkuman (dictionary).
    (Fungsi ini tidak berubah)
    """
    print("\n\n--- (ANALISIS HARI INI) Menganalisis Perilaku Saham ---")
    
    summary_findings = {}
    proximity = 0.02 # Toleransi 2%

    # Laporan S/R cluster
    print("Resistance Terkuat (berdasarkan jumlah tes dalam 2% rentang harga):")
    if clustered_resistances.empty:
        print("  - Tidak ada cluster resistance terdeteksi.")
    else:
        print(clustered_resistances.head(3).to_string(index=False))

    print("\nSupport Terkuat (berdasarkan jumlah tes dalam 2% rentang harga):")
    if clustered_supports.empty:
        print("  - Tidak ada cluster support terdeteksi.")
    else:
        print(clustered_supports.head(3).to_string(index=False))

    last_bar = df.iloc[-1]
    
    recent_resistances = raw_resistances[raw_resistances.index < last_bar.name]
    if recent_resistances.empty:
        print("\nKesimpulan: Tidak ada data resistance historis yang cukup.")
        return None 

    last_resistance_level = recent_resistances.iloc[-1]
    
    is_sideways = (market_structure == "SIDEWAYS")
    is_uptrend = (market_structure == "UPTREND")
    is_downtrend = (market_structure == "DOWNTREND")
    
    is_breakout_resistance = last_bar['close'] > last_resistance_level
    is_volume_strong = last_bar['volume'] > last_bar['volume_ma20']
    is_rsi_bullish = last_bar['rsi_14'] > 50
    is_macd_bullish = last_bar['macd_12_26_9'] > last_bar['macds_12_26_9']
    is_breakout_bb_upper = last_bar['close'] > last_bar['bbu_20_2.0_2.0']
    riding_the_band = (df['close'].iloc[-3:] > df['bbu_20_2.0_2.0'].iloc[-3:]).sum() >= 2
    
    kesimpulan = "Perilaku tidak terdeteksi."
    
    if is_sideways and is_breakout_resistance and is_volume_strong and is_breakout_bb_upper:
        kesimpulan = "**Pola Terdeteksi: Sideways (HH/LL) lalu Breakout (Tervalidasi Kuat oleh Volume & BB)**"
    elif is_uptrend and is_breakout_resistance and is_volume_strong:
        kesimpulan = "**Pola Terdeteksi: Melanjutkan Uptrend (HH/HL), Breakout Resistance (Tervalidasi Volume)**"
    elif is_uptrend and riding_the_band:
        kesimpulan = "**Pola Terdeteksi: Uptrend Kuat (HH/HL) dan 'Riding the Band'**"
    elif is_downtrend and is_breakout_resistance:
        kesimpulan = "**Peringatan: Downtrend (LH/LL). Breakout saat ini berisiko 'Bull Trap'.**"
    elif is_downtrend:
        kesimpulan = "**Pola Terdeteksi: Masih dalam Fase Downtrend (LH/LL).**"
    elif is_sideways and not is_breakout_resistance:
        kesimpulan = "**Pola Terdeteksi: Masih dalam Fase Sideways (HH/LL mixed)**"
    elif is_uptrend and not is_breakout_resistance:
         kesimpulan = f"**Pola Terdetaksi: Uptrend (HH/HL), bergerak menuju Res {last_resistance_level:.0f}**"
    elif is_breakout_resistance and not is_volume_strong:
        kesimpulan = "**Peringatan: Potensi False Breakout (Volume Lemah)**"

    
    summary_findings["Kesimpulan Utama"] = kesimpulan
    summary_findings["Market Structure (HH/LL)"] = market_structure
    summary_findings["Harga Saat Ini"] = f"{last_bar['close']:.0f}"
    summary_findings["Resistance Historis Terakhir"] = f"{last_resistance_level:.0f} (pada {recent_resistances.index[-1].date()})"
    
    volatility_check = last_bar['bbb_20_2.0_2.0'] < (df['bbb_20_2.0_2.0'].mean() * 0.7)
    summary_findings["Volatilitas (BB Width)"] = 'Rendah (Squeeze)' if volatility_check else 'Normal/Tinggi'
    
    summary_findings["Breakout Resistance"] = 'YA' if is_breakout_resistance else 'TIDAK'
    summary_findings["Breakout Upper BB"] = 'YA' if is_breakout_bb_upper else 'TIDAK'
    summary_findings["Validasi Bullish (Vol/RSI/MACD)"] = 'YA' if (is_volume_strong and is_rsi_bullish and is_macd_bullish) else 'TIDAK'
    summary_findings["Posisi RSI (14) Terakhir"] = f"{last_bar['rsi_14']:.2f}"
    summary_findings["Posisi MACD (12,26,9)"] = f"MACD: {last_bar['macd_12_26_9']:.2f} | Signal: {last_bar['macds_12_26_9']:.2f}"

    if not clustered_supports.empty:
        strongest_sup = clustered_supports.iloc[0]['Level']
        sup_is_flipped = (abs(clustered_resistances['Level'] - strongest_sup) / strongest_sup < proximity).any()
        if sup_is_flipped:
            summary_findings["Catatan Support"] = f"Zona support terkuat ({strongest_sup:.0f}) adalah zona flip (bekas R)."
            
    if not clustered_resistances.empty:
        strongest_res = clustered_resistances.iloc[0]['Level']
        res_is_flipped = (abs(clustered_supports['Level'] - strongest_res) / strongest_res < proximity).any()
        if res_is_flipped:
            summary_findings["Catatan Resistance"] = f"Zona resistance terkuat ({strongest_res:.0f}) adalah zona flip (bekas S)."

    ma_list = [10, 20, 50, 100, 200]
    ma_detail_list = []
    all_ma_bullish = True
    ma_summary_text = ""
    
    for ma_len in ma_list:
        ma_col = f'sma_{ma_len}'
        if ma_col in last_bar:
            ma_val = last_bar[ma_col]
            status = "di ATAS" if last_bar['close'] > ma_val else "di BAWAH"
            if last_bar['close'] < ma_val:
                all_ma_bullish = False
            ma_detail_list.append(f"MA {ma_len}: {ma_val:.0f} (Harga {status} MA)")
        else:
            ma_detail_list.append(f"MA {ma_len}: Tidak terhitung (data kurang)")
            all_ma_bullish = False
            
    if all_ma_bullish:
        ma_summary_text = "TREN SANGAT KUAT (Harga di atas semua MA)"
    elif last_bar['close'] > last_bar['sma_50']:
        ma_summary_text = "TREN JANGKA MENENGAH NAIK (Harga di atas MA 50)"
    else:
        ma_summary_text = "TREN JANGKA MENENGAH TURUN (Harga di bawah MA 50)"

    summary_findings["Status MA (Ringkasan)"] = ma_summary_text
    summary_findings["Status MA (Detail)"] = ma_detail_list 

    return summary_findings


# === [ FUNGSI 'recommend_trade' DIPERBARUI DI SINI ] ===

def recommend_trade(df, clustered_supports, clustered_resistances, raw_supports, raw_resistances, market_structure, min_rr_ratio=1.5):
    """
    Menganalisis HARI TERAKHIR untuk rekomendasi trade.
    (VERSI UPDATE: Mempertimbangkan S/R Bounce ATAU MA50 Bounce)
    """
    print("\n--- (REKOMENDASI TRADE) Menganalisis Sinyal Beli Hari Ini ---")
    
    # 1. Setup Awal
    last_bar = df.iloc[-1]
    current_price = last_bar['close']
    
    recommendation = {
        "Rekomendasi": "JANGAN BELI (Tahan/Tunggu)",
        "Alasan": "Tidak ada sinyal beli yang jelas atau R/R tidak memenuhi.",
        "Harga Saat Ini": f"{current_price:.0f}",
        "Area Beli (Entry)": "N/A",
        "Stop Loss (SL)": "N/A",
        "Take Profit (TP)": "N/A",
        "Risk/Reward (R/R)": "N/A"
    }

    # --- 2. Filter Kondisi WAJIB (Kondisi 'Jangan Beli') ---
    if market_structure == "DOWNTREND":
        recommendation["Alasan"] = "Kondisi pasar DOWNTREND, risiko terlalu tinggi."
        return recommendation

    if last_bar['rsi_14'] > 70 or last_bar['stochk_10_3_3'] > 80:
        recommendation["Alasan"] = "Indikator Overbought (RSI > 70 atau Stoch > 80)."
        return recommendation
        
    # === [FILTER INI DIHAPUS] ===
    # if current_price < last_bar['sma_50']:
    #    ... (Tadinya ada filter ini, sekarang dihapus agar MA bounce terdeteksi)

    # --- 3. Tentukan Level SL, TP, dan Entry (Berbasis S/R) ---

    # Level SL WAJIB berdasarkan STRONG Support (Cluster) terdekat
    valid_strong_supports = clustered_supports[clustered_supports['Level'] < current_price]
    if valid_strong_supports.empty:
        recommendation["Alasan"] = "Tidak ada STRONG support (cluster) di bawah harga saat ini untuk basis SL."
        return recommendation
    strong_support_level = valid_strong_supports.iloc[0]['Level']
    
    # Level TP WAJIB berdasarkan STRONG Resistance (Cluster) terdekat
    valid_strong_resistances = clustered_resistances[clustered_resistances['Level'] > current_price]
    if valid_strong_resistances.empty:
        recommendation["Alasan"] = "Tidak ada STRONG resistance (cluster) di atas harga saat ini untuk basis TP."
        return recommendation
    strong_resistance_level = valid_strong_resistances.iloc[0]['Level']

    # Level ENTRY berdasarkan SOFT Support (Raw Pivot) terdekat
    s_raw_recent = raw_supports.iloc[-3:]
    valid_soft_supports = s_raw_recent[s_raw_recent < current_price]
    
    entry_support_level = strong_support_level # Default ke strong support
    entry_support_type = "Strong (Cluster)"
    
    if not valid_soft_supports.empty:
        nearest_soft_support = valid_soft_supports.max() # Ambil yg paling dekat (tertinggi)
        if nearest_soft_support > strong_support_level:
            entry_support_level = nearest_soft_support
            entry_support_type = "Soft (Raw Pivot)"

    # --- 4. Hitung Risk/Reward (R/R) ---
    
    # SL 2% di bawah STRONG support
    sl_price = strong_support_level * 0.98 
    # TP 1% di bawah STRONG resistance
    tp_price = strong_resistance_level * 0.99 

    risk_per_share = current_price - sl_price
    reward_per_share = tp_price - current_price
    
    if risk_per_share <= 0 or reward_per_share <= 0:
        recommendation["Alasan"] = f"Logika S/R tidak valid (Harga: {current_price:.0f}, Strong S: {strong_support_level:.0f}, Strong R: {strong_resistance_level:.0f})."
        return recommendation

    rr_ratio = reward_per_share / risk_per_share

    # --- 5. (LOGIKA BARU) Buat Keputusan Akhir ---
    
    # Cek Skenario A: Apakah harga dekat S/R Entry? (maks 3% di atasnya)
    is_near_entry_support = (current_price - entry_support_level) / entry_support_level < 0.03
    
    # Cek Skenario B: Apakah harga memantul dari MA 50?
    sma_50_val = last_bar['sma_50']
    is_ma50_bounce = (last_bar['low'] < sma_50_val) and (current_price > sma_50_val) and (last_bar['close'] > last_bar['open'])

    # Cek Kondisi Wajib
    is_rr_good = rr_ratio >= min_rr_ratio
    is_confirmed = (last_bar['close'] > last_bar['open']) or (last_bar['macd_12_26_9'] > last_bar['macds_12_26_9'])

    # Update detail di rekomendasi (ini di-set untuk semua, even if no buy)
    recommendation["Area Beli (Entry)"] = f"~{entry_support_level:.0f} (S/R {entry_support_type}) ATAU ~{sma_50_val:.0f} (MA 50)"
    recommendation["Stop Loss (SL)"] = f"~{sl_price:.0f} (Basis: STRONG S {strong_support_level:.0f})"
    recommendation["Take Profit (TP)"] = f"~{tp_price:.0f} (Basis: STRONG R {strong_resistance_level:.0f})"
    recommendation["Risk/Reward (R/R)"] = f"1 : {rr_ratio:.2f}"

    # --- (LOGIKA BARU) Tentukan Rekomendasi ---
    
    if not is_rr_good:
        recommendation["Alasan"] = f"Risk/Reward tidak menarik (Hanya 1:{rr_ratio:.2f})."
        return recommendation # Gagal R/R

    if not is_confirmed:
        recommendation["Alasan"] = "Tidak ada konfirmasi sinyal (candle merah/MACD bearish)."
        return recommendation # Gagal Konfirmasi

    # JIKA Lolos R/R dan Konfirmasi, cek skenario Entry:
    if is_near_entry_support:
        recommendation["Rekomendasi"] = "REKOMENDASI BELI"
        recommendation["Alasan"] = f"Harga dekat Area Beli S/R ({entry_support_level:.0f}), R/R bagus, Sinyal terkonfirmasi."
    elif is_ma50_bounce:
        recommendation["Rekomendasi"] = "REKOMENDASI BELI"
        recommendation["Alasan"] = f"Harga memantul (Bounce) dari SMA 50, R/R bagus, Sinyal terkonfirmasi."
    else:
        # Jika lolos R/R dan Konfirmasi, TAPI tidak dekat S/R atau MA
        recommendation["Alasan"] = f"Harga 'nanggung', jauh dari Area Beli S/R ({entry_support_level:.0f}) dan tidak memantul dari MA50."
         
    return recommendation