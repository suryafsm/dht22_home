# 🌡️ Dashboard Monitoring DHT22

Dashboard real-time untuk memantau suhu & kelembaban dari sensor **DHT22**
yang terpasang di **Tegalrejo, Argomulyo, Salatiga**. Sensor mengirim data
setiap **20 detik** melalui ESP32 dan disimpan di InfluxDB, lalu ditampilkan
di sini lengkap dengan analisis dan prediksi.

## ✨ Fitur

- 📡 **Live Monitor** — gauge suhu & kelembaban real-time dengan auto-refresh, status koneksi sensor, dan alert saat nilai keluar dari ambang normal
- 📈 **Data Historis** — grafik interaktif dengan rentang waktu fleksibel dan autoscale otomatis
- 🔮 **Prediksi** — 4 metode: AI (LLM), Regresi Linear, Moving Average, dan ARIMA dengan interval kepercayaan
- 🔬 **Uji Diagnostik** — analisis stasioneritas, tren, dan pola musiman dari data sensor

## 🛠️ Dibangun dengan

Streamlit · Plotly · InfluxDB · statsmodels · scikit-learn

https://dht22home-suryafsm.streamlit.app/


*Proyek monitoring IoT — data sensor lingkungan Tegalrejo, Argomulyo, Salatiga.*
