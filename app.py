"""
Dashboard Monitoring DHT22 dengan Streamlit — v3 (Revisi UI/UX + ARIMA)
========================================================================
- Data dari InfluxDB (measurement: tas_ai_2026)
- 4 Metode Prediksi: AI (Groq), Statistik (Linear Regression), Moving
  Average, dan ARIMA (time series) dengan interval kepercayaan
- Uji diagnostik time series: stasioneritas (ADF test), dekomposisi
  tren/musiman/residual
- Tampilan: tab layout, kartu KPI, badge status koneksi, tema modern
- Auto-refresh mulus pakai st.fragment (tidak lagi blocking time.sleep loop)
- Fitur lain: alert ambang batas (threshold), grafik dual-axis, sparkline
  mini, ringkasan cepat, export prediksi ke CSV

Catatan instalasi:
    pip install streamlit>=1.33 pandas numpy plotly influxdb3-python groq scikit-learn statsmodels
(st.fragment butuh Streamlit >= 1.33. Jika versi lebih lama, auto-refresh
otomatis nonaktif dan tombol refresh manual akan muncul sebagai gantinya.)
"""

import time
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
import streamlit as st
from influxdb_client_3 import InfluxDBClient3
from groq import Groq
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller

# ============================================================================
# ZONA WAKTU — WIB (Asia/Jakarta, UTC+7)
# ============================================================================
# InfluxDB menyimpan & mengembalikan waktu dalam UTC (format RFC3339, akhiran
# "Z"). Semua timestamp yang diambil dari InfluxDB dikonversi ke WIB di sini
# supaya seluruh dashboard (gauge, grafik, tabel, log) konsisten menampilkan
# waktu lokal Indonesia, bukan UTC.
WIB = ZoneInfo("Asia/Jakarta")


def to_wib(value):
    """Konversi Timestamp/Series pandas (asumsi UTC bila belum ber-tz) ke WIB"""
    if isinstance(value, pd.Series):
        if value.dt.tz is None:
            value = value.dt.tz_localize("UTC")
        return value.dt.tz_convert(WIB)
    if value.tzinfo is None:
        value = value.tz_localize("UTC")
    return value.tz_convert(WIB)


def now_wib():
    return datetime.now(WIB)

# ============================================================================
# KONFIGURASI HALAMAN
# ============================================================================
st.set_page_config(
    page_title="DHT22 Monitoring Dashboard",
    page_icon="🌡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================================
# CSS KUSTOM — TEMA MODERN
# ============================================================================
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"]  { font-family: 'Inter', sans-serif; }

    /* Sembunyikan padding atas bawaan Streamlit agar header custom pas */
    .block-container { padding-top: 1.5rem; padding-bottom: 2rem; }

    /* ---------- Header banner ---------- */
    .dht-header {
        background: linear-gradient(120deg, #0f172a 0%, #0ea5a5 55%, #14b8a6 100%);
        border-radius: 20px;
        padding: 28px 32px;
        margin-bottom: 22px;
        color: white;
        box-shadow: 0 10px 30px rgba(14,165,165,0.25);
    }
    .dht-header h1 { margin: 0; font-size: 1.7rem; font-weight: 800; }
    .dht-header p { margin: 4px 0 0 0; opacity: 0.85; font-size: 0.92rem; }

    /* ---------- Status pill ---------- */
    .pill {
        display: inline-flex; align-items: center; gap: 6px;
        padding: 5px 14px; border-radius: 999px; font-size: 0.8rem;
        font-weight: 600; letter-spacing: 0.2px;
    }
    .pill-ok      { background: rgba(34,197,94,0.15); color: #16a34a; border: 1px solid rgba(34,197,94,0.35);}
    .pill-warn    { background: rgba(245,158,11,0.15); color: #b45309; border: 1px solid rgba(245,158,11,0.35);}
    .pill-danger  { background: rgba(239,68,68,0.15); color: #dc2626; border: 1px solid rgba(239,68,68,0.35);}
    .pill-offline { background: rgba(100,116,139,0.15); color: #475569; border: 1px solid rgba(100,116,139,0.35);}
    .dot { width: 8px; height: 8px; border-radius: 50%; background: currentColor; display: inline-block; }

    /* ---------- Kartu ---------- */
    .dht-card {
        background: var(--background-color, #ffffff);
        border: 1px solid rgba(120,120,120,0.15);
        border-radius: 16px;
        padding: 18px 20px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.04);
    }

    /* ---------- Metric override ---------- */
    div[data-testid="stMetric"] {
        background: rgba(120,120,120,0.06);
        border-radius: 14px;
        padding: 12px 16px 6px 16px;
        border: 1px solid rgba(120,120,120,0.1);
    }
    div[data-testid="stMetricLabel"] { font-weight: 600; opacity: 0.75; }

    /* ---------- Tabs ---------- */
    button[data-baseweb="tab"] { font-weight: 600; font-size: 0.95rem; }

    /* ---------- Section title ---------- */
    .section-title { font-weight: 700; font-size: 1.05rem; margin: 6px 0 10px 0; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ============================================================================
# INISIALISASI SESSION STATE
# ============================================================================
_defaults = {
    "historical_df": pd.DataFrame(),
    "chart_updated": None,
    "forecast_result": None,
    "forecast_method": "",
    "question_text": "",
    "y_min_temp": 0, "y_max_temp": 50,
    "y_min_hum": 0, "y_max_hum": 100,
    "autoscale_y": True,
    "auto_refresh": True,
    "refresh_interval": 10,
    "temp_threshold": (18, 32),
    "hum_threshold": (30, 80),
    "alert_log": [],
    "prev_temp": None,
    "prev_hum": None,
    "chart_style": "Terpisah",
    "fetch_mode": "Rentang Waktu",
    "time_value": 5,
    "time_unit": "jam",
    "n_data": 100,
    "interval_label": "5 menit",
    "time_label": "5 jam (rata-rata 5 menit)",
    "visitor_count": None,
    "_visit_counted": False,
    "_visitor_debug": "",
}
for k, v in _defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ============================================================================
# COUNTER PENGUNJUNG (CounterAPI v2 — perlu API token, disimpan di secrets)
# ============================================================================
# Setiap sesi browser baru (tab/reload) akan menambah counter tepat SEKALI,
# ditandai lewat session_state agar tidak ikut bertambah tiap kali ada
# interaksi widget yang men-trigger rerun Streamlit.
_COUNTERAPI_WORKSPACE = "dht22-suryafsm"
_COUNTERAPI_NAME = "first-counter-4967"


def _extract_counter_value(data):
    """Parsing tangguh (recursive) — cari nilai counter di JSON apapun
    bentuknya/berapa pun kedalaman nesting-nya, dengan mengecek nama field
    yang paling umum dipakai CounterAPI ('value', 'count', dst)."""
    keys_priority = ("value", "count", "current", "total", "counter_value")

    def search(node):
        if isinstance(node, dict):
            for k in keys_priority:
                if k in node and isinstance(node[k], (int, float)) and not isinstance(node[k], bool):
                    return int(node[k])
            for v in node.values():
                res = search(v)
                if res is not None:
                    return res
        elif isinstance(node, list):
            for v in node:
                res = search(v)
                if res is not None:
                    return res
        return None

    return search(data)


def _bump_visitor_counter():
    """Naikkan counter (endpoint /up) lalu baca nilai TERKINI lewat endpoint
    baca terpisah (tanpa /up) — CounterAPI v2 pakai cache buffering, jadi
    respons /up sendiri belum tentu langsung membawa nilai terbaru."""
    try:
        token = st.secrets.get("COUNTERAPI_TOKEN")
    except Exception:
        token = None
    if not token:
        st.session_state["_visitor_debug"] = "COUNTERAPI_TOKEN belum diisi di secrets"
        return None

    headers = {"Authorization": f"Bearer {token}"}
    base_url = f"https://api.counterapi.dev/v2/{_COUNTERAPI_WORKSPACE}/{_COUNTERAPI_NAME}"

    r_up = None
    try:
        r_up = requests.get(f"{base_url}/up", headers=headers, timeout=5)
        up_dbg = f"UP {r_up.status_code}: {r_up.text[:200]}"
    except Exception as e:
        up_dbg = f"UP error: {e}"

    r_get = None
    try:
        r_get = requests.get(base_url, headers=headers, timeout=5)
        get_dbg = f"GET {r_get.status_code}: {r_get.text[:200]}"
    except Exception as e:
        get_dbg = f"GET error: {e}"

    st.session_state["_visitor_debug"] = f"{up_dbg} | {get_dbg}"

    val = _extract_counter_value(r_get.json()) if (r_get is not None and r_get.status_code == 200) else None
    if val is None and r_up is not None and r_up.status_code == 200:
        val = _extract_counter_value(r_up.json())  # fallback kalau /up ternyata sudah bawa nilainya juga
    if val is None:
        st.session_state["_visitor_debug"] += " — gagal parsing nilai dari kedua respons di atas"
    return val


if not st.session_state._visit_counted:
    st.session_state.visitor_count = _bump_visitor_counter()
    st.session_state._visit_counted = True

# ============================================================================
# KONEKSI KE DATABASE
# ============================================================================
@st.cache_resource
def init_clients():
    """Inisialisasi koneksi ke InfluxDB dan Groq"""
    try:
        influx_client = InfluxDBClient3(
            host=st.secrets["INFLUXDB_HOST"],
            token=st.secrets["INFLUXDB_TOKEN"],
            database=st.secrets["INFLUXDB_BUCKET"],
        )
        groq_client = Groq(api_key=st.secrets["GROQ_API_KEY"])
        return influx_client, groq_client
    except Exception as e:
        st.error(f"❌ Gagal koneksi: {e}")
        st.stop()


try:
    influx_client, groq_client = init_clients()
except Exception:
    st.error("❌ Error: Periksa secrets.toml")
    st.stop()

# ============================================================================
# FUNGSI QUERY DATA
# ============================================================================
@st.cache_data(ttl=10)
def get_latest_data():
    """Ambil 1 data terbaru untuk gauge"""
    query = """
    SELECT suhu, kelembaban, time
    FROM "tas_ai_2026"
    ORDER BY time DESC
    LIMIT 1
    """
    try:
        df = influx_client.query_dataframe(query, language="sql")
        if not df.empty:
            return {
                "temperature": float(df["suhu"].iloc[0]),
                "humidity": float(df["kelembaban"].iloc[0]),
                "time": to_wib(pd.to_datetime(df["time"].iloc[0])),
            }
    except Exception as e:
        print(f"Error get latest data: {e}")
    return None


@st.cache_data(ttl=15)
def get_sparkline_data(minutes=30):
    """Ambil data ringan untuk sparkline mini di kartu KPI"""
    query = f"""
    SELECT suhu, kelembaban, time
    FROM "tas_ai_2026"
    WHERE time >= now() - interval '{minutes} minute'
    ORDER BY time
    """
    try:
        df = influx_client.query_dataframe(query, language="sql")
        if not df.empty:
            df["time"] = to_wib(pd.to_datetime(df["time"]))
            df = df.rename(columns={"suhu": "temperature", "kelembaban": "humidity"})
            return df.dropna(subset=["temperature", "humidity"])
    except Exception as e:
        print(f"Error get sparkline data: {e}")
    return pd.DataFrame()


@st.cache_data(ttl=60)
def query_historical_raw(span_seconds):
    """Ambil data historis MENTAH (tanpa averaging) untuk rentang waktu
    tertentu dalam detik. Proses averaging dilakukan terpisah di
    `apply_interval_averaging` sesuai Interval yang dipilih user."""
    span_seconds = max(1, int(span_seconds))
    query = f"""
    SELECT suhu, kelembaban, time
    FROM "tas_ai_2026"
    WHERE time >= now() - interval '{span_seconds} second'
    ORDER BY time
    """
    try:
        df = influx_client.query_dataframe(query, language="sql")
        if not df.empty and "time" in df.columns:
            df["time"] = to_wib(pd.to_datetime(df["time"]))
            df = df.dropna(subset=["suhu", "kelembaban"])
            df = df.rename(columns={"suhu": "temperature", "kelembaban": "humidity"})
            return df.sort_values("time").reset_index(drop=True)
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Error query: {e}")
        return pd.DataFrame()


def apply_interval_averaging(df_raw, interval_sec, n_data=None):
    """Ratakan (averaging) data mentah sesuai Interval yang dipilih user.
    Jika n_data diisi, ambil N titik TERAKHIR setelah averaging (mode
    'N Data')."""
    if df_raw.empty:
        return df_raw
    freq = _seconds_to_freq(interval_sec)
    df = df_raw.set_index("time")[["temperature", "humidity"]]
    df = df.resample(freq).mean().dropna().reset_index()
    if n_data is not None:
        df = df.tail(int(n_data)).reset_index(drop=True)
    return df


# Pilihan Interval averaging yang bisa dipilih user di sidebar
INTERVAL_OPTIONS = {
    "10 detik": 10, "30 detik": 30, "1 menit": 60, "5 menit": 300,
    "15 menit": 900, "30 menit": 1800, "1 jam": 3600, "3 jam": 10800,
    "6 jam": 21600, "12 jam": 43200, "1 hari": 86400,
}
_UNIT_TO_SECONDS = {"menit": 60, "jam": 3600, "hari": 86400, "minggu": 604800}


# ============================================================================
# HELPER: STATUS KONEKSI & ALERT
# ============================================================================
def get_connection_status(latest_time):
    """Tentukan status koneksi berdasarkan usia data terbaru"""
    if latest_time is None:
        return "offline", "Tidak Ada Data"
    try:
        now = pd.Timestamp.now(tz=latest_time.tz) if latest_time.tzinfo else pd.Timestamp.now()
        age_sec = (now - latest_time).total_seconds()
    except Exception:
        age_sec = 0
    if age_sec < 30:
        return "ok", "Online"
    elif age_sec < 300:
        return "warn", f"Delay {int(age_sec)}s"
    else:
        return "offline", "Offline"


def status_pill(kind, label):
    return f'<span class="pill pill-{kind}"><span class="dot"></span>{label}</span>'


def check_thresholds(temp, hum):
    """Cek ambang batas & catat ke alert log (debounced)"""
    t_lo, t_hi = st.session_state.temp_threshold
    h_lo, h_hi = st.session_state.hum_threshold
    now_str = now_wib().strftime("%H:%M:%S")

    temp_alert = temp is not None and (temp < t_lo or temp > t_hi)
    hum_alert = hum is not None and (hum < h_lo or hum > h_hi)

    if temp_alert and not st.session_state.get("_temp_alert_active", False):
        st.toast(f"🌡️ Suhu di luar batas: {temp:.1f}°C", icon="⚠️")
        st.session_state.alert_log.insert(0, f"[{now_str}] Suhu {temp:.1f}°C di luar rentang {t_lo}-{t_hi}°C")
    if hum_alert and not st.session_state.get("_hum_alert_active", False):
        st.toast(f"💧 Kelembaban di luar batas: {hum:.1f}%", icon="⚠️")
        st.session_state.alert_log.insert(0, f"[{now_str}] Kelembaban {hum:.1f}% di luar rentang {h_lo}-{h_hi}%")

    st.session_state["_temp_alert_active"] = temp_alert
    st.session_state["_hum_alert_active"] = hum_alert
    st.session_state.alert_log = st.session_state.alert_log[:20]
    return temp_alert, hum_alert


# ============================================================================
# FUNGSI GAUGE & SPARKLINE
# ============================================================================
def create_gauge(value, title, min_val=0, max_val=50, unit=""):
    if title == "Suhu":
        if value < 20:
            gauge_color = "#38BDF8"
        elif value < 28:
            gauge_color = "#22C55E"
        elif value < 35:
            gauge_color = "#F59E0B"
        else:
            gauge_color = "#EF4444"
    else:
        if value < 30:
            gauge_color = "#F59E0B"
        elif value < 60:
            gauge_color = "#22C55E"
        elif value < 80:
            gauge_color = "#38BDF8"
        else:
            gauge_color = "#EF4444"

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=value,
            number={"suffix": unit, "font": {"size": 34, "color": gauge_color}},
            title={"text": title, "font": {"size": 16}},
            gauge={
                "axis": {"range": [min_val, max_val], "tickwidth": 1, "tickcolor": "rgba(120,120,120,0.4)"},
                "bar": {"color": gauge_color, "thickness": 0.28},
                "bgcolor": "rgba(0,0,0,0)",
                "borderwidth": 0,
                "steps": [
                    {"range": [min_val, max_val * 0.4], "color": "rgba(148,163,184,0.12)"},
                    {"range": [max_val * 0.4, max_val * 0.7], "color": "rgba(148,163,184,0.08)"},
                    {"range": [max_val * 0.7, max_val], "color": "rgba(148,163,184,0.12)"},
                ],
            },
        )
    )
    fig.update_layout(
        height=220,
        margin=dict(l=15, r=15, t=45, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def mini_sparkline(df, col, color):
    """Grafik mini tanpa sumbu untuk tren singkat di kartu KPI"""
    fig = go.Figure()
    if not df.empty:
        fig.add_trace(
            go.Scatter(
                x=df["time"], y=df[col], mode="lines",
                line=dict(color=color, width=2),
                fill="tozeroy", fillcolor=color.replace(")", ",0.12)").replace("rgb", "rgba"),
            )
        )
    fig.update_layout(
        height=60,
        margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        showlegend=False,
    )
    return fig


# ============================================================================
# FUNGSI PLOT — DATA HISTORIS
# ============================================================================
def _auto_range(*series_list, pad_ratio=0.12, min_span=1.0):
    """Hitung rentang Y dari NILAI DATA ASLI (bukan dari bounding box shape
    Plotly, yang ikut menghitung area fill/tozeroy sehingga salah memaksa
    mulai dari 0). Diberi sedikit padding di atas & bawah agar tidak mepet."""
    vals = pd.concat([s.dropna() for s in series_list if s is not None and not s.empty])
    if vals.empty:
        return None
    lo, hi = float(vals.min()), float(vals.max())
    span = hi - lo
    if span < min_span:
        pad = min_span / 2
        mid = (hi + lo) / 2
        lo, hi = mid - pad, mid + pad
    else:
        pad = span * pad_ratio
        lo, hi = lo - pad, hi + pad
    return [lo, hi]


def _y_axis_kwargs(rng):
    """rng: None → autorange bawaan Plotly (fallback saja).
    [lo, hi] → rentang eksplisit, baik hasil hitung otomatis (_auto_range)
    maupun manual dari pengguna."""
    if rng is None or rng[0] is None or rng[1] is None:
        return {"autorange": True}
    return {"range": rng}


def plot_separate_charts(df, title_prefix="", temp_range=None, hum_range=None):
    if temp_range is None:
        temp_range = _auto_range(df["temperature"])
    if hum_range is None:
        hum_range = _auto_range(df["humidity"])

    fig_temp = go.Figure()
    fig_temp.add_trace(go.Scatter(
        x=df["time"], y=df["temperature"], name="Suhu",
        line=dict(color="#EF4444", width=2.2),
        fill="tozeroy", fillcolor="rgba(239,68,68,0.08)",
    ))
    fig_temp.update_layout(
        title=f"{title_prefix} · Suhu (°C)", xaxis_title=None,
        hovermode="x unified", height=320, template="plotly_white",
        margin=dict(l=10, r=10, t=45, b=10),
        yaxis=dict(title="°C", **_y_axis_kwargs(temp_range)),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
    )

    fig_hum = go.Figure()
    fig_hum.add_trace(go.Scatter(
        x=df["time"], y=df["humidity"], name="Kelembaban",
        line=dict(color="#38BDF8", width=2.2),
        fill="tozeroy", fillcolor="rgba(56,189,248,0.08)",
    ))
    fig_hum.update_layout(
        title=f"{title_prefix} · Kelembaban (%)", xaxis_title=None,
        hovermode="x unified", height=320, template="plotly_white",
        margin=dict(l=10, r=10, t=45, b=10),
        yaxis=dict(title="%", **_y_axis_kwargs(hum_range)),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig_temp, fig_hum


def plot_combined_dual_axis(df, title_prefix=""):
    """Fitur baru: satu grafik dengan dua sumbu-Y (suhu & kelembaban bersamaan)"""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["time"], y=df["temperature"], name="Suhu (°C)",
        line=dict(color="#EF4444", width=2.2), yaxis="y1",
    ))
    fig.add_trace(go.Scatter(
        x=df["time"], y=df["humidity"], name="Kelembaban (%)",
        line=dict(color="#38BDF8", width=2.2), yaxis="y2",
    ))
    fig.update_layout(
        title=f"{title_prefix} · Suhu & Kelembaban (Dual-Axis)",
        height=420, template="plotly_white", hovermode="x unified",
        margin=dict(l=10, r=10, t=45, b=10),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        yaxis=dict(title=dict(text="Suhu (°C)", font=dict(color="#EF4444")), tickfont=dict(color="#EF4444")),
        yaxis2=dict(title=dict(text="Kelembaban (%)", font=dict(color="#38BDF8")), tickfont=dict(color="#38BDF8"),
                    overlaying="y", side="right", showgrid=False),
    )
    return fig


# ============================================================================
# FUNGSI PREDIKSI
# ============================================================================
def predict_ai(df, question=""):
    if df.empty:
        return "❌ Tidak ada data untuk dianalisis."

    stats = df[["temperature", "humidity"]].describe()
    latest = df.tail(10)[["time", "temperature", "humidity"]].to_string()
    is_prediction = any(w in question.lower() for w in ["prediksi", "predict", "forecast", "24 jam"])

    if is_prediction:
        system_prompt = "Anda adalah ahli prediksi data sensor IoT."
        prompt = f"""
        Anda adalah analis data IoT. Berdasarkan data sensor berikut:

        STATISTIK:
        {stats}

        10 DATA TERAKHIR:
        {latest}

        Pertanyaan: {question}

        Berikan jawaban dengan format:
        1. Prediksi Suhu (nilai, trend, rentang)
        2. Prediksi Kelembaban (nilai, trend, rentang)
        3. Saran atau rekomendasi
        """
    else:
        system_prompt = "Anda adalah ahli analisis data IoT."
        prompt = f"""
        Anda adalah analis data IoT. Berdasarkan data sensor berikut:

        STATISTIK:
        {stats}

        10 DATA TERAKHIR:
        {latest}

        Pertanyaan: {question}

        Berikan jawaban dengan format:
        1. Ringkasan Singkat
        2. Analisis Tren
        3. Anomali (jika ada)
        4. Rekomendasi
        """

    try:
        response = groq_client.chat.completions.create(
            model="openai/gpt-oss-20b",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,
            max_tokens=800,
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"❌ Error AI: {e}"


def predict_statistical(df, hours_ahead=24):
    if len(df) < 10:
        return None, None, "Data terlalu sedikit (minimal 10 titik)"
    try:
        df = df.copy()
        df["hours"] = (df["time"] - df["time"].min()).dt.total_seconds() / 3600
        X = df["hours"].values.reshape(-1, 1)

        model_temp = LinearRegression().fit(X, df["temperature"].values)
        model_hum = LinearRegression().fit(X, df["humidity"].values)

        last_hour = df["hours"].max()
        future_hours = np.array([last_hour + i for i in range(1, hours_ahead + 1)]).reshape(-1, 1)

        pred_temp = model_temp.predict(future_hours)
        pred_hum = model_hum.predict(future_hours)

        future_times = pd.date_range(start=df["time"].max(), periods=hours_ahead + 1, freq="h")[1:]
        result_df = pd.DataFrame({"time": future_times, "temperature": pred_temp, "humidity": pred_hum})

        r2_temp = model_temp.score(X, df["temperature"].values)
        r2_hum = model_hum.score(X, df["humidity"].values)
        info = f"📊 **Akurasi Model:**\n- Suhu: R² = {r2_temp:.3f}\n- Kelembaban: R² = {r2_hum:.3f}"
        return result_df, info, None
    except Exception as e:
        return None, None, f"❌ Error: {e}"


def predict_moving_average(df, window=5, hours_ahead=24):
    if len(df) < window:
        return None, None, f"Data terlalu sedikit (minimal {window} titik)"
    try:
        last_temp_ma = df["temperature"].tail(window).mean()
        last_hum_ma = df["humidity"].tail(window).mean()
        temp_diff = df["temperature"].diff().tail(window).mean()
        hum_diff = df["humidity"].diff().tail(window).mean()

        future_times = pd.date_range(start=df["time"].max(), periods=hours_ahead + 1, freq="h")[1:]

        pred_temp, pred_hum = [], []
        current_temp, current_hum = last_temp_ma, last_hum_ma
        for _ in range(hours_ahead):
            current_temp += temp_diff
            current_hum += hum_diff
            pred_temp.append(current_temp)
            pred_hum.append(current_hum)

        result_df = pd.DataFrame({"time": future_times, "temperature": pred_temp, "humidity": pred_hum})
        info = (f"📊 **Parameter Moving Average:**\n- Window: {window} data point\n"
                f"- Trend Suhu: {temp_diff:.3f}°C per jam\n- Trend Kelembaban: {hum_diff:.3f}% per jam")
        return result_df, info, None
    except Exception as e:
        return None, None, f"❌ Error: {e}"


def _seconds_to_freq(sec):
    """Ubah durasi detik jadi string frekuensi pandas yang valid & 'rapi'."""
    sec = int(sec)
    if sec < 60:
        return f"{sec}s"
    if sec < 3600:
        return f"{max(1, sec // 60)}min"
    if sec < 86400:
        return f"{max(1, sec // 3600)}h"
    return f"{max(1, sec // 86400)}D"


# Kandidat resolusi (detik): dari 10 detik sampai 1 hari
_FREQ_CANDIDATES_SEC = [10, 15, 30, 60, 120, 300, 600, 900, 1800, 3600, 7200, 10800, 21600, 43200, 86400]


def _pick_adaptive_rule(df, min_points=20, target_points=90):
    """Pilih resolusi resampling secara OTOMATIS berdasarkan rentang waktu &
    kepadatan data yang tersedia — bukan dipatok per jam. Data 1 jam dengan
    sampling tiap 20 detik akan diratakan per menit (bukan per jam), supaya
    tetap dapat cukup titik untuk ARIMA/diagnostik. Data berminggu-minggu
    otomatis diratakan lebih kasar (jam/hari) supaya model tetap ringan."""
    t = df["time"].sort_values()
    if len(t) < 5:
        return None, None
    span = (t.iloc[-1] - t.iloc[0]).total_seconds()
    if span <= 0:
        return None, None

    raw_interval = t.diff().median().total_seconds()
    raw_interval = raw_interval if raw_interval and raw_interval > 0 else 1

    # Kandidat yang tidak "mengada-ada" (tidak lebih halus dari data mentah)
    # dan tetap menghasilkan >= min_points titik pada rentang data ini
    valid = [c for c in _FREQ_CANDIDATES_SEC if c >= raw_interval * 0.8 and span / c >= min_points]

    if valid:
        chosen = min(valid, key=lambda c: abs(span / c - target_points))
    else:
        # Rentang terlalu pendek untuk target ideal — pakai resolusi sehalus
        # mungkin (mendekati interval sampling asli) asalkan masih dapat >=5 titik
        fallback = [c for c in _FREQ_CANDIDATES_SEC if span / c >= 5]
        if not fallback:
            return None, None
        chosen = min(fallback)

    return chosen, raw_interval


def _resample_adaptive(df, col, rule_sec):
    """Ratakan data pada resolusi yang dipilih otomatis, isi celah kecil."""
    freq = _seconds_to_freq(rule_sec)
    # batas interpolasi menyesuaikan resolusi: makin halus, makin longgar batas isian celahnya
    limit = max(3, int(600 / rule_sec)) if rule_sec < 600 else 3
    s = df.set_index("time")[col].resample(freq).mean().interpolate(limit=limit)
    return s.dropna(), freq


def predict_arima(df, hours_ahead=24, auto_order=True, manual_order=(2, 1, 2), ci_alpha=0.2):
    """4. PREDIKSI ARIMA — ringan, dengan resolusi resampling ADAPTIF
    (otomatis menyesuaikan rentang & kepadatan data), pemilihan orde
    otomatis via AIC, dan interval kepercayaan (default 80%)."""
    if len(df) < 15:
        return None, None, "Data terlalu sedikit (minimal 15 titik mentah) untuk ARIMA. Perbesar rentang waktu atau tunggu data lebih banyak."

    try:
        rule_sec, raw_interval = _pick_adaptive_rule(df)
        if rule_sec is None:
            return None, None, "Rentang waktu data terlalu pendek/datanya belum bervariasi untuk ARIMA. Coba perbesar rentang waktu di sidebar."

        temp_s, freq = _resample_adaptive(df, "temperature", rule_sec)
        hum_s, _ = _resample_adaptive(df, "humidity", rule_sec)

        if len(temp_s) < 12 or len(hum_s) < 12:
            return None, None, (
                f"Data setelah diratakan tiap {freq} masih terlalu sedikit "
                f"({min(len(temp_s), len(hum_s))} titik, minimal 12). Perbesar rentang waktu di sidebar."
            )

        def fit_best(series):
            if not auto_order:
                model = ARIMA(series, order=manual_order).fit()
                return model, manual_order, model.aic
            candidates = [(1, 1, 1), (2, 1, 2), (1, 1, 0), (0, 1, 1), (2, 1, 0), (1, 0, 0), (3, 1, 1)]
            best_model, best_order, best_aic = None, manual_order, np.inf
            for cand in candidates:
                try:
                    m = ARIMA(series, order=cand).fit()
                    if m.aic < best_aic:
                        best_model, best_order, best_aic = m, cand, m.aic
                except Exception:
                    continue
            if best_model is None:
                best_model = ARIMA(series, order=(1, 1, 1)).fit()
                best_order, best_aic = (1, 1, 1), best_model.aic
            return best_model, best_order, best_aic

        model_temp, order_temp, aic_temp = fit_best(temp_s)
        model_hum, order_hum, aic_hum = fit_best(hum_s)

        # Konversi "jam ke depan" yang diminta user jadi jumlah langkah pada
        # resolusi yang dipilih otomatis, dengan batas wajar agar tetap ringan
        steps = int(round(hours_ahead * 3600 / rule_sec))
        steps = max(3, min(steps, 500))
        capped = steps < round(hours_ahead * 3600 / rule_sec) - 0.5

        fc_temp = model_temp.get_forecast(steps=steps)
        fc_hum = model_hum.get_forecast(steps=steps)

        ci_temp = fc_temp.conf_int(alpha=ci_alpha)
        ci_hum = fc_hum.conf_int(alpha=ci_alpha)

        future_times = pd.date_range(start=temp_s.index.max(), periods=steps + 1, freq=freq)[1:]

        result_df = pd.DataFrame({
            "time": future_times,
            "temperature": fc_temp.predicted_mean.values,
            "humidity": fc_hum.predicted_mean.values,
            "temp_lower": ci_temp.iloc[:, 0].values,
            "temp_upper": ci_temp.iloc[:, 1].values,
            "hum_lower": ci_hum.iloc[:, 0].values,
            "hum_upper": ci_hum.iloc[:, 1].values,
        })

        actual_coverage_h = steps * rule_sec / 3600
        conf_pct = int((1 - ci_alpha) * 100)
        info = (
            f"📊 **Model ARIMA:**\n"
            f"- Suhu: ARIMA{order_temp}, AIC = {aic_temp:.1f}\n"
            f"- Kelembaban: ARIMA{order_hum}, AIC = {aic_hum:.1f}\n"
            f"- Interval kepercayaan: {conf_pct}%\n"
            f"- Resolusi otomatis: tiap **{freq}** ({len(temp_s)} titik latih, dipilih otomatis dari rentang & kepadatan data)"
        )
        if capped:
            info += f"\n- ⚠️ Cakupan prediksi dibatasi ke ~{actual_coverage_h:.1f} jam (dari {hours_ahead} jam diminta) agar model tetap ringan di resolusi ini"
        return result_df, info, None
    except Exception as e:
        return None, None, f"❌ Error ARIMA: {e}"


def run_diagnostics(df):
    """Uji diagnostik time series: stasioneritas (ADF) + dekomposisi
    tren/musiman/residual, untuk suhu dan kelembaban. Resolusi resampling
    dipilih otomatis (adaptif), bukan dipatok per jam."""
    if len(df) < 15:
        return None, "Data terlalu sedikit untuk uji diagnostik (minimal 15 titik)."

    try:
        rule_sec, _ = _pick_adaptive_rule(df, min_points=15, target_points=90)
        if rule_sec is None:
            return None, "Rentang waktu data terlalu pendek/datanya belum bervariasi untuk uji diagnostik."

        results = {}
        for name, col in [("Suhu", "temperature"), ("Kelembaban", "humidity")]:
            series, freq = _resample_adaptive(df, col, rule_sec)
            if len(series) < 12:
                results[name] = {"insufficient": True, "n": len(series), "freq": freq}
                continue

            adf_stat, adf_p = adfuller(series.dropna())[:2]

            # Perkiraan periode musiman harian pada resolusi yang dipakai,
            # dibatasi agar tidak melebihi separuh panjang data
            period_full_day = max(2, round(86400 / rule_sec))
            period = min(period_full_day, max(2, len(series) // 2))
            decomposition = None
            if len(series) >= period * 2:
                try:
                    decomposition = seasonal_decompose(
                        series, model="additive", period=period, extrapolate_trend="freq"
                    )
                except Exception:
                    decomposition = None

            # Estimasi kekuatan tren & musiman (0-1) ala Hyndman
            trend_strength = seasonal_strength = None
            if decomposition is not None:
                try:
                    resid = decomposition.resid.dropna()
                    detrended = (decomposition.trend + decomposition.resid).dropna()
                    deseason = (series - decomposition.seasonal).dropna()
                    var_resid = np.var(resid)
                    if len(detrended) > 1 and np.var(detrended) > 0:
                        trend_strength = max(0, 1 - var_resid / np.var(detrended))
                    if len(deseason) > 1 and np.var(deseason) > 0:
                        seasonal_strength = max(0, 1 - var_resid / np.var(deseason))
                except Exception:
                    pass

            results[name] = {
                "insufficient": False,
                "adf_stat": adf_stat,
                "adf_p": adf_p,
                "stationary": adf_p < 0.05,
                "decomposition": decomposition,
                "period": period,
                "freq": freq,
                "n": len(series),
                "trend_strength": trend_strength,
                "seasonal_strength": seasonal_strength,
            }
        return results, None
    except Exception as e:
        return None, f"❌ Error diagnostik: {e}"


def plot_decomposition(decomposition, series, label, color):
    """Grafik dekomposisi: observed, trend, seasonal, residual"""
    fig = make_subplots(
        rows=4, cols=1, shared_xaxes=True,
        subplot_titles=("Data Asli", "Tren", "Musiman", "Residual"),
        vertical_spacing=0.06,
    )
    fig.add_trace(go.Scatter(x=series.index, y=series.values, line=dict(color=color, width=1.6)), row=1, col=1)
    fig.add_trace(go.Scatter(x=decomposition.trend.index, y=decomposition.trend.values,
                              line=dict(color="#0EA5A5", width=1.8)), row=2, col=1)
    fig.add_trace(go.Scatter(x=decomposition.seasonal.index, y=decomposition.seasonal.values,
                              line=dict(color="#F59E0B", width=1.4)), row=3, col=1)
    fig.add_trace(go.Scatter(x=decomposition.resid.index, y=decomposition.resid.values,
                              mode="markers", marker=dict(color="#94A3B8", size=4)), row=4, col=1)
    fig.update_layout(
        height=560, showlegend=False, template="plotly_white",
        margin=dict(l=10, r=10, t=40, b=10),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        title=f"Dekomposisi {label}",
    )
    return fig


def plot_arima_forecast(hist_df, forecast_df, temp_range=None, hum_range=None, title_prefix=""):
    """Grafik prediksi ARIMA dengan pita interval kepercayaan"""
    if temp_range is None:
        temp_range = _auto_range(hist_df["temperature"], forecast_df["temperature"],
                                  forecast_df["temp_lower"], forecast_df["temp_upper"])
    if hum_range is None:
        hum_range = _auto_range(hist_df["humidity"], forecast_df["humidity"],
                                 forecast_df["hum_lower"], forecast_df["hum_upper"])

    def band(fig, x, lower, upper, color):
        fig.add_trace(go.Scatter(
            x=list(x) + list(x)[::-1],
            y=list(upper) + list(lower)[::-1],
            fill="toself", fillcolor=color, line=dict(color="rgba(0,0,0,0)"),
            name="Interval kepercayaan", hoverinfo="skip", showlegend=True,
        ))

    fig_temp = go.Figure()
    band(fig_temp, forecast_df["time"], forecast_df["temp_lower"], forecast_df["temp_upper"], "rgba(249,115,22,0.18)")
    fig_temp.add_trace(go.Scatter(x=hist_df["time"], y=hist_df["temperature"], name="Historis",
                                   line=dict(color="#EF4444", width=2)))
    fig_temp.add_trace(go.Scatter(x=forecast_df["time"], y=forecast_df["temperature"], name="Prediksi ARIMA",
                                   line=dict(color="#F97316", width=2, dash="dash")))
    fig_temp.update_layout(
        title=f"{title_prefix} · Suhu (°C)", height=340, template="plotly_white", hovermode="x unified",
        margin=dict(l=10, r=10, t=45, b=10), yaxis=dict(title="°C", **_y_axis_kwargs(temp_range)),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
    )

    fig_hum = go.Figure()
    band(fig_hum, forecast_df["time"], forecast_df["hum_lower"], forecast_df["hum_upper"], "rgba(56,189,248,0.18)")
    fig_hum.add_trace(go.Scatter(x=hist_df["time"], y=hist_df["humidity"], name="Historis",
                                  line=dict(color="#0EA5A5", width=2)))
    fig_hum.add_trace(go.Scatter(x=forecast_df["time"], y=forecast_df["humidity"], name="Prediksi ARIMA",
                                  line=dict(color="#38BDF8", width=2, dash="dash")))
    fig_hum.update_layout(
        title=f"{title_prefix} · Kelembaban (%)", height=340, template="plotly_white", hovermode="x unified",
        margin=dict(l=10, r=10, t=45, b=10), yaxis=dict(title="%", **_y_axis_kwargs(hum_range)),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
    )

    split_time = hist_df["time"].max()
    for f in (fig_temp, fig_hum):
        f.add_vline(x=split_time, line_dash="dash", line_color="gray")
        f.add_annotation(x=split_time, y=1, yref="paper", text="← Historis | Prediksi →", showarrow=False)
    return fig_temp, fig_hum


def get_y_ranges():
    """Ambil rentang Y-axis aktif: None (autoscale) atau [min, max] manual,
    sesuai toggle Autoscale di sidebar."""
    if st.session_state.autoscale_y:
        return None, None
    return (
        [st.session_state.y_min_temp, st.session_state.y_max_temp],
        [st.session_state.y_min_hum, st.session_state.y_max_hum],
    )


# ============================================================================
# FRAGMENT: PANEL LIVE (AUTO-REFRESH MULUS, TANPA BLOCKING)
# ============================================================================
try:
    _fragment = st.fragment
except AttributeError:
    def _fragment(*args, **kwargs):
        def _decorator(func):
            return func
        return _decorator


def _live_panel_impl():
    latest = get_latest_data()
    spark_df = get_sparkline_data(30)

    top_l, top_r = st.columns([3, 1])
    with top_l:
        st.markdown('<div class="section-title">📡 Kondisi Saat Ini</div>', unsafe_allow_html=True)
    with top_r:
        kind, label = get_connection_status(latest["time"] if latest else None)
        st.markdown(f'<div style="text-align:right">{status_pill(kind, label)}</div>', unsafe_allow_html=True)

    if latest:
        check_thresholds(latest["temperature"], latest["humidity"])

        d_temp = None if st.session_state.prev_temp is None else latest["temperature"] - st.session_state.prev_temp
        d_hum = None if st.session_state.prev_hum is None else latest["humidity"] - st.session_state.prev_hum

        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(create_gauge(latest["temperature"], "Suhu", 0, 50, "°C"),
                             use_container_width=True, key=f"gauge_temp_{time.time()}")
            st.metric("Suhu", f"{latest['temperature']:.1f}°C",
                      delta=None if d_temp is None else f"{d_temp:+.1f}°C")
            if not spark_df.empty:
                st.plotly_chart(mini_sparkline(spark_df, "temperature", "rgb(239,68,68)"),
                                 use_container_width=True, config={"displayModeBar": False},
                                 key=f"spark_temp_{time.time()}")
            st.caption(f"🕐 Update: {latest['time'].strftime('%H:%M:%S')} WIB")

        with col2:
            st.plotly_chart(create_gauge(latest["humidity"], "Kelembaban", 0, 100, "%"),
                             use_container_width=True, key=f"gauge_hum_{time.time()}")
            st.metric("Kelembaban", f"{latest['humidity']:.1f}%",
                      delta=None if d_hum is None else f"{d_hum:+.1f}%")
            if not spark_df.empty:
                st.plotly_chart(mini_sparkline(spark_df, "humidity", "rgb(56,189,248)"),
                                 use_container_width=True, config={"displayModeBar": False},
                                 key=f"spark_hum_{time.time()}")
            st.caption(f"💧 Rentang normal: {st.session_state.hum_threshold[0]}–{st.session_state.hum_threshold[1]}%")

        st.session_state.prev_temp = latest["temperature"]
        st.session_state.prev_hum = latest["humidity"]

        with st.expander("🕓 10 Pembacaan Terakhir"):
            if not spark_df.empty:
                st.dataframe(
                    spark_df.tail(10).sort_values("time", ascending=False).rename(
                        columns={"time": "Waktu (WIB)", "temperature": "Suhu (°C)", "humidity": "Kelembaban (%)"}
                    ),
                    use_container_width=True, hide_index=True,
                )
            else:
                st.caption("Belum ada data pada 30 menit terakhir.")
    else:
        st.warning("⚠️ Tidak ada data terbaru dari sensor. Menunggu data dari ESP32...")

    st.caption(f"🔄 Panel ini otomatis diperbarui — terakhir dicek {now_wib().strftime('%H:%M:%S')} WIB")


refresh_seconds = st.session_state.refresh_interval if st.session_state.auto_refresh else None

@_fragment(run_every=refresh_seconds)
def live_panel():
    _live_panel_impl()


# ============================================================================
# SIDEBAR
# ============================================================================
with st.sidebar:
    st.header("⚙️ Kontrol Dashboard")

    st.session_state.auto_refresh = st.checkbox(
        "🔄 Auto Refresh Panel Live", value=st.session_state.auto_refresh,
        help="Panel live diperbarui otomatis tanpa reload seluruh halaman.",
    )
    if st.session_state.auto_refresh:
        st.session_state.refresh_interval = st.select_slider(
            "Interval refresh", options=[5, 10, 15, 30, 60],
            value=st.session_state.refresh_interval, format_func=lambda v: f"{v} detik",
        )
    if not hasattr(st, "fragment"):
        st.caption("ℹ️ Auto-refresh mulus butuh Streamlit ≥ 1.33. Update paket Streamlit Anda.")

    st.divider()
    st.subheader("🚨 Ambang Batas Alert")
    st.session_state.temp_threshold = st.slider("Suhu normal (°C)", 0, 60, st.session_state.temp_threshold)
    st.session_state.hum_threshold = st.slider("Kelembaban normal (%)", 0, 100, st.session_state.hum_threshold)

    if st.session_state.alert_log:
        with st.expander(f"📋 Riwayat Alert ({len(st.session_state.alert_log)})"):
            for entry in st.session_state.alert_log:
                st.caption(entry)
            if st.button("🗑️ Bersihkan Riwayat", use_container_width=True):
                st.session_state.alert_log = []
                st.rerun()

    st.divider()
    st.subheader("📊 Kontrol Chart Historis")

    st.session_state.fetch_mode = st.radio(
        "Cara ambil data",
        ["Rentang Waktu", "Jumlah Data (N)"],
        index=0 if st.session_state.fetch_mode == "Rentang Waktu" else 1,
        horizontal=True,
        help="Rentang Waktu: tentukan Nilai + Unit (mis. 5 jam terakhir). "
             "Jumlah Data: tentukan berapa titik data yang mau ditampilkan SETELAH averaging.",
    )

    if st.session_state.fetch_mode == "Rentang Waktu":
        col1, col2 = st.columns(2)
        with col1:
            time_value = st.number_input("Nilai", min_value=1, max_value=1000, value=st.session_state.time_value, step=1)
        with col2:
            time_unit = st.selectbox("Unit", ["menit", "jam", "hari", "minggu"],
                                      index=["menit", "jam", "hari", "minggu"].index(st.session_state.time_unit))
        st.session_state.time_value, st.session_state.time_unit = time_value, time_unit

        interval_label = st.selectbox(
            "Interval (averaging)", list(INTERVAL_OPTIONS.keys()),
            index=list(INTERVAL_OPTIONS.keys()).index(st.session_state.interval_label),
        )
        st.session_state.interval_label = interval_label
        st.caption(f"📅 {time_value} {time_unit} terakhir · rata-rata tiap {interval_label}")
    else:
        col1, col2 = st.columns(2)
        with col1:
            n_data = st.number_input("N Data", min_value=2, max_value=5000, value=st.session_state.n_data, step=1)
        with col2:
            interval_label = st.selectbox(
                "Interval (averaging)", list(INTERVAL_OPTIONS.keys()),
                index=list(INTERVAL_OPTIONS.keys()).index(st.session_state.interval_label),
            )
        st.session_state.n_data, st.session_state.interval_label = n_data, interval_label
        st.caption(f"📅 {n_data} titik data · rata-rata tiap {interval_label}")

    st.session_state.chart_style = st.radio(
        "Gaya grafik", ["Terpisah", "Dual-Axis"], index=0 if st.session_state.chart_style == "Terpisah" else 1,
        horizontal=True,
    )

    update_chart = st.button("📈 Update Chart", use_container_width=True, type="primary")
    if update_chart:
        interval_sec = INTERVAL_OPTIONS[st.session_state.interval_label]

        if st.session_state.fetch_mode == "Rentang Waktu":
            span_sec = st.session_state.time_value * _UNIT_TO_SECONDS[st.session_state.time_unit]
            time_label = f"{st.session_state.time_value} {st.session_state.time_unit}"
            n_data_target = None
        else:
            # Ambil sedikit lebih lebar (+20% buffer) supaya setelah averaging
            # tetap dapat >= N titik walau ada celah data
            span_sec = int(st.session_state.n_data * interval_sec * 1.2) + interval_sec
            time_label = f"{st.session_state.n_data} data"
            n_data_target = st.session_state.n_data

        with st.spinner(f"Mengambil & meratakan data (tiap {st.session_state.interval_label})..."):
            df_raw = query_historical_raw(span_sec)
            df = apply_interval_averaging(df_raw, interval_sec, n_data=n_data_target)

            if not df.empty:
                st.session_state.historical_df = df
                st.session_state.chart_updated = now_wib()
                st.session_state.time_label = f"{time_label} (rata-rata {st.session_state.interval_label})"
                if n_data_target and len(df) < n_data_target:
                    st.warning(f"⚠️ Cuma dapat {len(df)} dari {n_data_target} titik diminta — data mentah di rentang ini belum cukup.")
                else:
                    st.success(f"✅ Data diupdate! {len(df)} titik data (setelah averaging)")
            else:
                st.warning("⚠️ Tidak ada data untuk rentang/parameter ini")

    with st.expander("🎛️ Rentang Y-Axis", expanded=False):
        st.session_state.autoscale_y = st.checkbox(
            "🔎 Autoscale (cari min/max otomatis)", value=st.session_state.autoscale_y,
            help="Saat aktif, sumbu-Y grafik menyesuaikan sendiri ke rentang data yang tampil. "
                 "Matikan untuk mengunci rentang manual (mis. supaya beberapa grafik sebanding).",
        )
        if not st.session_state.autoscale_y:
            c1, c2 = st.columns(2)
            with c1:
                st.session_state.y_min_temp = st.number_input("Min Suhu (°C)", value=st.session_state.y_min_temp, step=1)
                st.session_state.y_min_hum = st.number_input("Min Kelembaban (%)", value=st.session_state.y_min_hum, step=1)
            with c2:
                st.session_state.y_max_temp = st.number_input("Max Suhu (°C)", value=st.session_state.y_max_temp, step=1)
                st.session_state.y_max_hum = st.number_input("Max Kelembaban (%)", value=st.session_state.y_max_hum, step=1)
        else:
            st.caption("Rentang manual disembunyikan selagi autoscale aktif.")

    st.divider()
    if not st.session_state.historical_df.empty:
        csv = st.session_state.historical_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "📥 Download Data Historis (CSV)", data=csv,
            file_name=f"dht22_data_{now_wib().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv", use_container_width=True,
        )

    st.divider()
    st.caption(f"🕐 {now_wib().strftime('%H:%M:%S')} WIB")
    if st.session_state.visitor_count is not None:
        st.caption(f"cnt={st.session_state.visitor_count}")
    else:
        st.caption("cnt=?")
        if st.session_state._visitor_debug:
            st.caption(f"⚠️ {st.session_state._visitor_debug}")
    st.caption("Suryasatriya ©2026")


# ============================================================================
# HEADER
# ============================================================================
st.markdown(
    """
    <div class="dht-header">
        <h1>🌡️ Dashboard Monitoring DHT22</h1>
        <p>Live sensor · Prediksi AI · Statistik · Moving Average — data dari InfluxDB Cloud</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ============================================================================
# TABS UTAMA
# ============================================================================
tab_live, tab_hist, tab_pred = st.tabs(["📡 Live Monitor", "📈 Data Historis", "🔮 Prediksi & Analisis"])

# ---------------------------------------------------------------- TAB LIVE
with tab_live:
    live_panel()

# ---------------------------------------------------------------- TAB HISTORIS
with tab_hist:
    if st.session_state.chart_updated:
        st.caption(
            f"🔄 Terakhir update: {st.session_state.chart_updated.strftime('%Y-%m-%d %H:%M:%S')} WIB · "
            f"📅 {st.session_state.time_label}"
        )

    if not st.session_state.historical_df.empty:
        df = st.session_state.historical_df

        kpi1, kpi2, kpi3, kpi4 = st.columns(4)
        kpi1.metric("Rata-rata Suhu", f"{df['temperature'].mean():.1f}°C")
        kpi2.metric("Suhu Maks / Min", f"{df['temperature'].max():.1f} / {df['temperature'].min():.1f}°C")
        kpi3.metric("Rata-rata Kelembaban", f"{df['humidity'].mean():.1f}%")
        kpi4.metric("Kelembaban Maks / Min", f"{df['humidity'].max():.1f} / {df['humidity'].min():.1f}%")

        st.markdown("<br>", unsafe_allow_html=True)

        if st.session_state.chart_style == "Terpisah":
            t_range, h_range = get_y_ranges()
            fig_temp, fig_hum = plot_separate_charts(
                df, title_prefix=f"Data {st.session_state.time_label}", temp_range=t_range, hum_range=h_range,
            )
            st.plotly_chart(fig_temp, use_container_width=True)
            st.plotly_chart(fig_hum, use_container_width=True)
        else:
            st.plotly_chart(plot_combined_dual_axis(df, title_prefix=f"Data {st.session_state.time_label}"),
                             use_container_width=True)

        with st.expander("📊 Statistik Deskriptif"):
            c1, c2 = st.columns(2)
            with c1:
                st.write("**🌡️ Suhu**")
                st.dataframe(df["temperature"].describe().round(2), use_container_width=True)
            with c2:
                st.write("**💧 Kelembaban**")
                st.dataframe(df["humidity"].describe().round(2), use_container_width=True)
    else:
        st.info("📌 Belum ada data historis. Klik 'Update Chart' di sidebar untuk mengambil data.")

# ---------------------------------------------------------------- TAB PREDIKSI
with tab_pred:
    with st.expander("🔬 Uji Musiman, Tren & Stasioneritas (ADF Test)"):
        st.caption(
            "Data diratakan otomatis (resolusi menyesuaikan rentang & kepadatan data — "
            "bisa per detik/menit/jam) lalu diuji: **ADF test** untuk mengecek apakah data "
            "stasioner (tidak ada tren jangka panjang), dan **dekomposisi** untuk memisahkan "
            "komponen tren, musiman (siklus harian), dan residual (noise)."
        )
        diag_btn = st.button("▶️ Jalankan Uji Diagnostik", use_container_width=False)
        if diag_btn:
            if st.session_state.historical_df.empty:
                st.warning("⚠️ Tidak ada data. Update chart terlebih dahulu di sidebar.")
            else:
                with st.spinner("Menghitung ADF test & dekomposisi..."):
                    diag_results, diag_err = run_diagnostics(st.session_state.historical_df)
                if diag_err:
                    st.error(diag_err)
                elif diag_results:
                    for label, res in diag_results.items():
                        st.markdown(f"#### {'🌡️' if label == 'Suhu' else '💧'} {label}")
                        if res.get("insufficient"):
                            st.caption(f"Data terlalu sedikit ({res['n']} titik pada resolusi {res.get('freq', '?')}) untuk uji ini.")
                            continue

                        st.caption(f"📏 Resolusi otomatis: setiap **{res['freq']}** · {res['n']} titik dianalisis")
                        c1, c2, c3, c4 = st.columns(4)
                        c1.metric("ADF statistic", f"{res['adf_stat']:.3f}")
                        c2.metric("p-value", f"{res['adf_p']:.4f}")
                        stat_kind = "ok" if res["stationary"] else "warn"
                        stat_label = "Stasioner ✅" if res["stationary"] else "Tidak stasioner (ada tren)"
                        c3.markdown(status_pill(stat_kind, stat_label), unsafe_allow_html=True)
                        c4.metric("Periode musiman", f"{res['period']} × {res['freq']}")

                        if res["trend_strength"] is not None or res["seasonal_strength"] is not None:
                            c5, c6 = st.columns(2)
                            if res["trend_strength"] is not None:
                                c5.metric("Kekuatan Tren", f"{res['trend_strength']*100:.0f}%")
                            if res["seasonal_strength"] is not None:
                                c6.metric("Kekuatan Musiman", f"{res['seasonal_strength']*100:.0f}%")

                        if res["decomposition"] is not None:
                            rule_sec, _ = _pick_adaptive_rule(st.session_state.historical_df, min_points=15, target_points=90)
                            series_for_plot, _ = _resample_adaptive(
                                st.session_state.historical_df,
                                "temperature" if label == "Suhu" else "humidity",
                                rule_sec,
                            )
                            color = "#EF4444" if label == "Suhu" else "#38BDF8"
                            st.plotly_chart(
                                plot_decomposition(res["decomposition"], series_for_plot, label, color),
                                use_container_width=True,
                            )
                        else:
                            st.caption("Data belum cukup untuk dekomposisi musiman (butuh ≥ 2x periode).")
                        st.divider()

    method = st.radio(
        "Pilih Metode Prediksi:",
        ["AI (Groq)", "Statistik (Linear Regression)", "Moving Average", "ARIMA (Time Series)"],
        index=0, horizontal=True,
    )
    st.markdown("<br>", unsafe_allow_html=True)

    # ---------------- METODE 1: AI ----------------
    if method == "AI (Groq)":
        col1, col2 = st.columns([3, 1])
        with col1:
            question = st.text_input(
                "💬 Tanyakan tentang data sensor:",
                value=st.session_state.question_text,
                placeholder="Contoh: Bagaimana tren suhu 3 hari terakhir?",
                key="ai_question_input",
            )
        with col2:
            st.write("")
            predict_btn = st.button("🔮 Prediksi dengan AI", use_container_width=True, type="primary")

        col3, _, _ = st.columns([1, 1, 1])
        with col3:
            analyze_btn = st.button("💬 Analisis Data", use_container_width=True, help="Analisis tanpa prediksi")

        st.markdown("**Pertanyaan Cepat:**")
        qcols = st.columns(4)
        quick = [
            ("📈 Tren", "Analisis tren suhu dan kelembaban."),
            ("⚠️ Anomali", "Deteksi anomali dalam data ini."),
            ("🔮 Prediksi 24 Jam", "Prediksi tren suhu dan kelembaban 24 jam ke depan."),
            ("📊 Statistik", "Berikan analisis statistik lengkap."),
        ]
        for c, (label, q) in zip(qcols, quick):
            with c:
                if st.button(label, use_container_width=True):
                    st.session_state.question_text = q
                    st.rerun()

        if predict_btn or analyze_btn:
            if not st.session_state.historical_df.empty:
                default_q = "Prediksi tren 24 jam ke depan." if predict_btn else "Berikan analisis komprehensif data sensor ini."
                label = "Prediksi" if predict_btn else "Analisis"
                with st.spinner(f"🧠 AI sedang mem-{label.lower()}..."):
                    result = predict_ai(st.session_state.historical_df, question if question else default_q)
                    st.session_state.forecast_result = result
                    st.session_state.forecast_method = f"AI (Groq) - {label}"
                st.success(f"✅ {label} selesai!")
                st.markdown(f"### 📋 Hasil {label} AI")
                st.markdown(result)
                st.caption("🤖 Model: openai/gpt-oss-20b")
            else:
                st.warning("⚠️ Tidak ada data. Update chart terlebih dahulu di sidebar.")

    # ---------------- METODE 2: STATISTIK ----------------
    elif method == "Statistik (Linear Regression)":
        col1, col2 = st.columns([2, 1])
        with col1:
            hours = st.slider("Jam Prediksi ke Depan:", 1, 72, 24, key="stat_hours")
        with col2:
            st.write("")
            predict_btn = st.button("📊 Prediksi Statistik", use_container_width=True, type="primary")

        if predict_btn:
            if not st.session_state.historical_df.empty:
                with st.spinner("Menghitung prediksi..."):
                    forecast_df, info, error = predict_statistical(st.session_state.historical_df, hours)
                if error:
                    st.error(error)
                else:
                    st.session_state.forecast_result = forecast_df
                    st.session_state.forecast_method = "Statistik (Linear Regression)"
                    st.success("✅ Prediksi selesai!")
                    st.markdown(info)

                    combined_df = pd.concat([st.session_state.historical_df, forecast_df]).reset_index(drop=True)
                    t_range, h_range = get_y_ranges()
                    fig_temp, fig_hum = plot_separate_charts(
                        combined_df, title_prefix=f"Prediksi {hours} Jam · Linear Regression",
                        temp_range=t_range, hum_range=h_range,
                    )
                    split_time = st.session_state.historical_df["time"].max()
                    for f in (fig_temp, fig_hum):
                        f.add_vline(x=split_time, line_dash="dash", line_color="gray")
                        f.add_annotation(x=split_time, y=1, yref="paper", text="← Historis | Prediksi →", showarrow=False)
                    st.plotly_chart(fig_temp, use_container_width=True)
                    st.plotly_chart(fig_hum, use_container_width=True)

                    with st.expander("📋 Tabel Prediksi"):
                        st.dataframe(forecast_df, use_container_width=True)
                        st.download_button(
                            "📥 Download Prediksi (CSV)",
                            data=forecast_df.to_csv(index=False).encode("utf-8"),
                            file_name="prediksi_linear_regression.csv", mime="text/csv",
                        )
            else:
                st.warning("⚠️ Tidak ada data. Update chart terlebih dahulu di sidebar.")

    # ---------------- METODE 3: MOVING AVERAGE ----------------
    elif method == "Moving Average":
        col1, col2, col3 = st.columns(3)
        with col1:
            window = st.slider("Window (data point):", 3, 20, 5, key="ma_window")
        with col2:
            hours = st.slider("Jam Prediksi:", 1, 72, 24, key="ma_hours")
        with col3:
            st.write("")
            predict_btn = st.button("📈 Prediksi MA", use_container_width=True, type="primary")

        if predict_btn:
            if not st.session_state.historical_df.empty:
                with st.spinner("Menghitung Moving Average..."):
                    forecast_df, info, error = predict_moving_average(st.session_state.historical_df, window, hours)
                if error:
                    st.error(error)
                else:
                    st.session_state.forecast_result = forecast_df
                    st.session_state.forecast_method = "Moving Average"
                    st.success("✅ Prediksi selesai!")
                    st.markdown(info)

                    combined_df = pd.concat([st.session_state.historical_df, forecast_df]).reset_index(drop=True)
                    t_range, h_range = get_y_ranges()
                    fig_temp, fig_hum = plot_separate_charts(
                        combined_df, title_prefix=f"Prediksi {hours} Jam · MA (window={window})",
                        temp_range=t_range, hum_range=h_range,
                    )
                    split_time = st.session_state.historical_df["time"].max()
                    for f in (fig_temp, fig_hum):
                        f.add_vline(x=split_time, line_dash="dash", line_color="gray")
                        f.add_annotation(x=split_time, y=1, yref="paper", text="← Historis | Prediksi →", showarrow=False)
                    st.plotly_chart(fig_temp, use_container_width=True)
                    st.plotly_chart(fig_hum, use_container_width=True)

                    with st.expander("📋 Tabel Prediksi"):
                        st.dataframe(forecast_df, use_container_width=True)
                        st.download_button(
                            "📥 Download Prediksi (CSV)",
                            data=forecast_df.to_csv(index=False).encode("utf-8"),
                            file_name="prediksi_moving_average.csv", mime="text/csv",
                        )
            else:
                st.warning("⚠️ Tidak ada data. Update chart terlebih dahulu di sidebar.")

    # ---------------- METODE 4: ARIMA ----------------
    else:
        st.caption(
            "ARIMA cocok untuk data dengan tren yang jelas dan memberikan **interval "
            "kepercayaan**, bukan cuma satu angka prediksi. Resolusi data (detik/menit/jam) "
            "dipilih otomatis mengikuti rentang & kepadatan data yang Anda punya."
        )
        col1, col2, col3 = st.columns([1.3, 1, 1])
        with col1:
            hours = st.slider("Jam Prediksi ke Depan:", 1, 72, 24, key="arima_hours")
        with col2:
            auto_order = st.toggle("Orde otomatis (AIC)", value=True, key="arima_auto")
        with col3:
            conf_level = st.selectbox("Interval kepercayaan", ["80%", "90%", "95%"], index=0, key="arima_conf")
        ci_alpha = {"80%": 0.2, "90%": 0.1, "95%": 0.05}[conf_level]

        manual_order = (2, 1, 2)
        if not auto_order:
            m1, m2, m3 = st.columns(3)
            with m1:
                p = st.number_input("p (AR)", 0, 5, 2, key="arima_p")
            with m2:
                d = st.number_input("d (differencing)", 0, 2, 1, key="arima_d")
            with m3:
                q = st.number_input("q (MA)", 0, 5, 2, key="arima_q")
            manual_order = (p, d, q)

        predict_btn = st.button("📈 Prediksi ARIMA", use_container_width=True, type="primary")

        if predict_btn:
            if not st.session_state.historical_df.empty:
                with st.spinner("Melatih model ARIMA..."):
                    forecast_df, info, error = predict_arima(
                        st.session_state.historical_df, hours,
                        auto_order=auto_order, manual_order=manual_order, ci_alpha=ci_alpha,
                    )
                if error:
                    st.error(error)
                else:
                    st.session_state.forecast_result = forecast_df
                    st.session_state.forecast_method = "ARIMA (Time Series)"
                    st.success("✅ Prediksi selesai!")
                    st.markdown(info)

                    t_range, h_range = get_y_ranges()
                    fig_temp, fig_hum = plot_arima_forecast(
                        st.session_state.historical_df, forecast_df,
                        temp_range=t_range, hum_range=h_range,
                        title_prefix=f"Prediksi {hours} Jam · ARIMA",
                    )
                    st.plotly_chart(fig_temp, use_container_width=True)
                    st.plotly_chart(fig_hum, use_container_width=True)

                    with st.expander("📋 Tabel Prediksi"):
                        st.dataframe(forecast_df, use_container_width=True)
                        st.download_button(
                            "📥 Download Prediksi (CSV)",
                            data=forecast_df.to_csv(index=False).encode("utf-8"),
                            file_name="prediksi_arima.csv", mime="text/csv",
                        )
            else:
                st.warning("⚠️ Tidak ada data. Update chart terlebih dahulu di sidebar.")

    # ---------------- HASIL PREDIKSI SEBELUMNYA ----------------
    if st.session_state.forecast_result is not None:
        st.divider()
        if isinstance(st.session_state.forecast_result, pd.DataFrame):
            st.info(f"📌 Hasil prediksi terakhir: {st.session_state.forecast_method}")
            if not st.session_state.historical_df.empty:
                combined_df = pd.concat(
                    [st.session_state.historical_df, st.session_state.forecast_result]
                ).reset_index(drop=True)
                t_range, h_range = get_y_ranges()
                fig_temp, fig_hum = plot_separate_charts(
                    combined_df, title_prefix=f"Hasil Prediksi · {st.session_state.forecast_method}",
                    temp_range=t_range, hum_range=h_range,
                )
                split_time = st.session_state.historical_df["time"].max()
                for f in (fig_temp, fig_hum):
                    f.add_vline(x=split_time, line_dash="dash", line_color="gray")
                    f.add_annotation(x=split_time, y=1, yref="paper", text="← Historis | Prediksi →", showarrow=False)
                st.plotly_chart(fig_temp, use_container_width=True)
                st.plotly_chart(fig_hum, use_container_width=True)
        elif isinstance(st.session_state.forecast_result, str):
            st.info(f"📌 Hasil analisis terakhir: {st.session_state.forecast_method}")
            st.markdown(st.session_state.forecast_result)

# ============================================================================
# FOOTER
# ============================================================================
st.markdown("---")
f1, f2, f3 = st.columns(3)
f1.caption("📊 Measurement: Tegalrejo, Argomulyo, Salatiga")
f2.caption("⚡ Salam dari Pepe, Leo, Milo, Oksi, Tom-tom, Tim-tim, Bobby")
f3.caption(f"🔄 {now_wib().strftime('%Y-%m-%d %H:%M:%S')} WIB")