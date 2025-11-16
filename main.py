import argparse

# --- Impor fungsi dari file .py Anda yang lain ---
from data_loader import get_stock_data_from_csv
from core_analysis import (
    find_support_resistance, 
    detect_market_structure
)
from reporting import (
    scan_for_signals, 
    analyze_behavior,
    recommend_trade  # <-- Fungsi ini sudah diimpor
)
from charting import create_chart

def main():
    # 1. Setup Argumen Command Line
    parser = argparse.ArgumentParser(description="Backtesting Behavior Saham dari CSV.")
    parser.add_argument('--file', type=str, required=True, 
                        help="Path ke file CSV (Contoh: BBCA.JK_history.csv)")
    args = parser.parse_args()

    # 2. Ambil dan Hitung Data dari CSV
    data = get_stock_data_from_csv(args.file)
    
    if data is not None:
        # 3. Cari Level S/R
        clustered_s, clustered_r, raw_s, raw_r = find_support_resistance(data)
        
        # 4. Deteksi Market Structure (HH/LL)
        market_structure = detect_market_structure(raw_s, raw_r)
        
        # 5. Jalankan Pemindai Sinyal Historis
        df_signals = scan_for_signals(data, clustered_s, clustered_r)
        
        # 6. Ambil Rangkuman Laporan Hari Terakhir
        summary_report = analyze_behavior(data, clustered_s, clustered_r, raw_s, raw_r, market_structure)
        
        # 7. Cetak Rangkuman Laporan
        print("\n--- RANGKUMAN ANALISIS HARI TERAKHIR ---")
        if summary_report:
            for key, value in summary_report.items():
                if isinstance(value, list):
                    # Untuk 'Status MA (Detail)'
                    print(f"  - {key}:")
                    for item in value:
                        print(f"    -> {item}")
                else:
                    # Untuk semua item lain
                    print(f"  - {key}: {value}")
        
        
        # === [PERUBAHAN DI SINI] ===
        # 8. Dapatkan Rekomendasi Trade
        #    Kita tambahkan raw_s dan raw_r ke dalam panggilan fungsi
        trade_recommendation = recommend_trade(data, clustered_s, clustered_r, raw_s, raw_r, market_structure, min_rr_ratio=1.5)
        # === [AKHIR PERUBAHAN] ===
        
        # 9. Cetak Rekomendasi Trade
        print("\n--- REKOMENDASI TRADE HARI INI ---")
        if trade_recommendation:
            for key, value in trade_recommendation.items():
                print(f"  - {key}: {value}")

        
        # 10. Buat dan simpan visualisasi
        # output_filename = args.file.replace('.csv', '_analysis_chart.png').replace('.CSV', '_analysis_chart.png')
        # create_chart(data, clustered_s, clustered_r, raw_s, raw_r, df_signals, market_structure, filename=output_filename)

# Ini adalah entry point program
if __name__ == "__main__":
    main()