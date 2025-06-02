# =======================
# 📦 IMPORT LIBRARY
# =======================
import streamlit as st
import pandas as pd
import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly
import plotly.graph_objs as go
from datetime import datetime

# Konfigurasi halaman dashboard
st.set_page_config(page_title="Crypto Forecast Dashboard", layout="wide")

# =======================
# 📥 FUNGSI AMBIL DATA HISTORIS
# =======================
@st.cache_data  # Cache agar tidak perlu unduh ulang jika input sama
def load_data(symbol, start_date):
    df = yf.download(symbol, start=start_date)  # Ambil data dari Yahoo Finance
    if isinstance(df.columns, pd.MultiIndex):  # Tangani multi-index jika ada
        df.columns = df.columns.get_level_values(0)
    df = df[['Close']].dropna()  # Ambil kolom Close saja
    df.reset_index(inplace=True)  # Reset index agar kolom Date jadi kolom biasa
    return df

# =======================
# 🤖 FUNGSI FORECAST MENGGUNAKAN PROPHET
# =======================
def forecast_with_prophet(df, period_days):
    df_train = df.rename(columns={"Date": "ds", "Close": "y"})  # Format kolom untuk Prophet
    model = Prophet()  # Inisialisasi model Prophet
    model.fit(df_train)  # Latih model
    future = model.make_future_dataframe(periods=period_days)  # Buat tanggal ke depan
    forecast = model.predict(future)  # Prediksi harga
    return forecast, model  # Kembalikan hasil prediksi dan modelnya

# =======================
# 🎛️ SIDEBAR: PILIH KOIN & TANGGAL
# =======================
st.sidebar.title("Crypto Forecaster")
coin_symbol = st.sidebar.selectbox("Pilih Coin", ["BTC-USD", "ETH-USD", "ADA-USD", "DOGE-USD", "SOL-USD"])
start_date = st.sidebar.date_input("Start Date", datetime(2015, 1, 1))

# =======================
# 🔄 LOAD DATA HISTORIS
# =======================
data_load_state = st.text("Mengambil data harga...")
df = load_data(coin_symbol, start_date)
data_load_state.text("✅ Data harga berhasil diambil.")

# =======================
# 📊 TAMPILKAN GRAFIK HISTORIS
# =======================
st.subheader(f"📈 Harga Historis: {coin_symbol}")
st.line_chart(df.set_index("Date")["Close"])  # Grafik garis harga Close

# =======================
# 🔮 PREDIKSI 30 HARI KE DEPAN
# =======================
st.subheader("🔮 Prediksi Harian (30 Hari ke Depan)")
forecast_30, model_30 = forecast_with_prophet(df, 30)

fig_30 = plot_plotly(model_30, forecast_30)  # Buat plot interaktif
fig_30.update_layout(title=f"Prediksi 30 Hari - {coin_symbol}", xaxis_title="Tanggal", yaxis_title="Harga")
st.plotly_chart(fig_30, use_container_width=True)

# =======================
# 🔭 PREDIKSI 1 TAHUN KE DEPAN (365 HARI)
# =======================
st.subheader("🔭 Prediksi Jangka Panjang (1 Tahun ke Depan)")
forecast_365, model_365 = forecast_with_prophet(df, 365)

fig_365 = plot_plotly(model_365, forecast_365)
fig_365.update_layout(title=f"Prediksi 1 Tahun - {coin_symbol}", xaxis_title="Tanggal", yaxis_title="Harga")
st.plotly_chart(fig_365, use_container_width=True)

# =======================
# 📑 TABEL PREDIKSI GABUNGAN (30D & 365D)
# =======================
with st.expander("📊 Tabel Prediksi Lengkap"):
    merged = pd.merge(
        forecast_30[['ds', 'yhat']],
        forecast_365[['ds', 'yhat']],
        on='ds',
        how='outer',
        suffixes=('_30d', '_365d')
    )
    merged = merged.rename(columns={"ds": "Tanggal"})
    merged = merged.sort_values("Tanggal")
    st.dataframe(merged.set_index("Tanggal"), use_container_width=True, height=500)

# =======================
# 📥 TOMBOL DOWNLOAD EXCEL
# =======================
with st.sidebar:
    st.markdown("---")
    if st.button("📥 Download Prediksi"):
        merged.to_excel("prediksi_crypto.xlsx", index=False)
        st.success("File prediksi_crypto.xlsx berhasil disiapkan.")
