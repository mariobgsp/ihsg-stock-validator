import matplotlib.pyplot as plt
import mplfinance as mpf

# Definisi fungsi sekarang menerima raw_s dan raw_r
def create_chart(df, clustered_s, clustered_r, raw_s, raw_r, df_signals, market_structure, filename="stock_analysis_chart.png"):
    """
    Membuat dan menyimpan chart visualisasi dari data analisis.
    (VERSI UPDATE: Menambahkan subplot ke-4 untuk Volume)
    """
    print(f"\n--- Membuat Visualisasi Chart ---")
    
    # Tentukan data yang akan di-plot (misal: 1 tahun terakhir)
    plot_data = df.iloc[-252:]
    
    # --- (BERUBAH) Ubah 3, 1 menjadi 4, 1 ---
    fig, axes = plt.subplots(4, 1, figsize=(15, 14), # Tinggikan figsize (misal: 12 -> 14)
                             sharex=True, 
                             # Tambahkan 1 rasio lagi untuk volume
                             gridspec_kw={'height_ratios': [3, 1, 1, 1]}) 
    
    # Atur margin kanan agar ada ruang untuk teks harga
    plt.subplots_adjust(right=0.95)
    
    # --- 1. AXES[0]: CHART HARGA (Tidak berubah) ---
    ax_price = axes[0]
    ax_price.set_title(f"Analisis Teknikal (Tren: {market_structure})")
    ax_price.plot(plot_data.index, plot_data['close'], label='Close', color='blue', lw=1.5, zorder=2)
    ax_price.plot(plot_data.index, plot_data['sma_50'], label='SMA 50', color='orange', lw=0.8, zorder=1)
    ax_price.plot(plot_data.index, plot_data['sma_200'], label='SMA 200', color='red', lw=0.8, zorder=1)
    ax_price.plot(plot_data.index, plot_data['bbu_20_2.0_2.0'], label='BB Upper', color='gray', lw=0.5, linestyle='--', zorder=1)
    ax_price.plot(plot_data.index, plot_data['bbl_20_2.0_2.0'], label='BB Lower', color='gray', lw=0.5, linestyle='--', zorder=1)
    ax_price.fill_between(plot_data.index, plot_data['bbu_20_2.0_2.0'], plot_data['bbl_20_2.0_2.0'], color='gray', alpha=0.1, zorder=0)

    # Plot Keterangan (Titik) Pivot S/R
    supports_to_plot = raw_s[raw_s.index.isin(plot_data.index)]
    resistances_to_plot = raw_r[raw_r.index.isin(plot_data.index)]
    if not supports_to_plot.empty:
        ax_price.scatter(supports_to_plot.index, supports_to_plot.values, 
                         marker='.', color='green', s=30, label='Pivot Support (Hit Low)', zorder=3, alpha=0.6)
    if not resistances_to_plot.empty:
        ax_price.scatter(resistances_to_plot.index, resistances_to_plot.values, 
                         marker='.', color='red', s=30, label='Pivot Resistance (Hit High)', zorder=3, alpha=0.6)

    # Plot Support & Resistance Zones (Top 5)
    zone_percent = 0.01 
    half_zone = zone_percent / 2.0
    proximity = 0.02 
    
    # Plot Support Zones (Top 5)
    support_levels = clustered_s.head(5)
    if not support_levels.empty:
        max_hits = support_levels['Hits'].max()
        min_hits = support_levels['Hits'].min()
        min_alpha, max_alpha = 0.1, 0.4
        
        for i, row in support_levels.iterrows():
            level = row['Level']
            hits = row['Hits']
            if max_hits == min_hits: alpha = max_alpha
            else: alpha = min_alpha + ((hits - min_hits) / (max_hits - min_hits)) * (max_alpha - min_alpha)
            ymin = level * (1 - half_zone); ymax = level * (1 + half_zone)
            is_rbs_flip = (abs(clustered_r['Level'] - level) / level < proximity).any()
            box_color = 'cyan' if is_rbs_flip else 'green'
            plot_label = None
            if i == support_levels.index[0]:
                plot_label = f'Strongest Support (Hits: {hits})'
                if is_rbs_flip: plot_label += ' (RbS Flip Zone)'
            ax_price.axhspan(ymin, ymax, color=box_color, alpha=alpha, label=plot_label, zorder=0)
            ax_price.text(plot_data.index[-1], ymax, f' {ymax:.0f}', ha='left', va='bottom', fontsize=7, color=box_color)
            ax_price.text(plot_data.index[-1], ymin, f' {ymin:.0f}', ha='left', va='top', fontsize=7, color=box_color)

    # Plot Resistance Zones (Top 5)
    resistance_levels = clustered_r.head(5)
    if not resistance_levels.empty:
        max_hits = resistance_levels['Hits'].max()
        min_hits = resistance_levels['Hits'].min()
        min_alpha, max_alpha = 0.1, 0.4
        
        for i, row in resistance_levels.iterrows():
            level = row['Level']
            hits = row['Hits']
            if max_hits == min_hits: alpha = max_alpha
            else: alpha = min_alpha + ((hits - min_hits) / (max_hits - min_hits)) * (max_alpha - min_alpha)
            ymin = level * (1 - half_zone); ymax = level * (1 + half_zone)
            is_sbr_flip = (abs(clustered_s['Level'] - level) / level < proximity).any()
            box_color = 'magenta' if is_sbr_flip else 'red'
            plot_label = None
            if i == resistance_levels.index[0]:
                plot_label = f'Strongest Resistance (Hits: {hits})'
                if is_sbr_flip: plot_label += ' (SbR Flip Zone)'
            ax_price.axhspan(ymin, ymax, color=box_color, alpha=alpha, label=plot_label, zorder=0)
            ax_price.text(plot_data.index[-1], ymax, f' {ymax:.0f}', ha='left', va='bottom', fontsize=7, color=box_color)
            ax_price.text(plot_data.index[-1], ymin, f' {ymin:.0f}', ha='left', va='top', fontsize=7, color=box_color)

    
    # Plot Sinyal (Spots)
    if not df_signals.empty:
        signals_to_plot = df_signals[df_signals.index.isin(plot_data.index)]
        if not signals_to_plot.empty:
            
            # === [PERUBAHAN DI SINI] ===
            # 'BREAKOUT' dipindah ke 'buy_signals'
            buy_signals = signals_to_plot[signals_to_plot['Sinyal (Label)'].str.contains('BUY|BOUNCE|GOLDEN|REVERSAL - BULLISH|BREAKOUT')]
            sell_signals = signals_to_plot[signals_to_plot['Sinyal (Label)'].str.contains('DEATH|REJECTION|REVERSAL - BEARISH')]
            # === [AKHIR PERUBAHAN] ===

            if not buy_signals.empty:
                ax_price.scatter(buy_signals.index, buy_signals['Harga'], marker='^', color='green', s=100, label='Sinyal Beli/Positif', zorder=5)
            if not sell_signals.empty:
                ax_price.scatter(sell_signals.index, sell_signals['Harga'], marker='v', color='red', s=100, label='Sinyal Jual/Negatif', zorder=5)

    ax_price.set_ylabel("Harga")
    ax_price.legend()
    ax_price.grid(True, linestyle='--', alpha=0.5)
    
    
    # --- 2. (BARU) AXES[1]: CHART VOLUME ---
    ax_vol = axes[1]
    # Buat list warna: Hijau jika close > open, Merah jika close < open
    colors = (plot_data['close'] > plot_data['open']).map({True: 'green', False: 'red'})
    ax_vol.bar(plot_data.index, plot_data['volume'], color=colors, alpha=0.5, label='Volume')
    # Plot Volume MA 20
    ax_vol.plot(plot_data.index, plot_data['volume_ma20'], color='blue', lw=0.8, label='Volume MA(20)')
    ax_vol.set_ylabel("Volume")
    ax_vol.legend()
    ax_vol.grid(True, linestyle='--', alpha=0.5)
    
    
    # --- 3. (BERUBAH) AXES[2]: CHART RSI ---
    ax_rsi = axes[2] # <-- Diubah dari [1] menjadi [2]
    ax_rsi.plot(plot_data.index, plot_data['rsi_14'], label='RSI 14', color='purple', lw=1)
    ax_rsi.axhline(70, color='red', linestyle='--', lw=0.5)
    ax_rsi.axhline(30, color='green', linestyle='--', lw=0.5)
    ax_rsi.set_ylabel("RSI")
    ax_rsi.set_ylim(0, 100)
    ax_rsi.legend()
    ax_rsi.grid(True, linestyle='--', alpha=0.5)

    # --- 4. (BERUBAH) AXES[3]: CHART MACD ---
    ax_macd = axes[3] # <-- Diubah dari [2] menjadi [3]
    ax_macd.plot(plot_data.index, plot_data['macd_12_26_9'], label='MACD', color='blue', lw=1)
    ax_macd.plot(plot_data.index, plot_data['macds_12_26_9'], label='Signal', color='orange', lw=1)
    ax_macd.bar(plot_data.index, plot_data['macdh_12_26_9'], label='Histogram', color=(plot_data['macdh_12_26_9'] > 0).map({True: 'g', False: 'r'}), alpha=0.3)
    ax_macd.set_ylabel("MACD")
    ax_macd.legend()
    ax_macd.grid(True, linestyle='--', alpha=0.5)
    
    # --- Finalisasi ---
    plt.xlabel("Tanggal (Timeframe)")
    plt.tight_layout() 
    plt.subplots_adjust(right=0.92)
    plt.savefig(filename) 
    
    print(f"Chart berhasil disimpan ke: {filename}")