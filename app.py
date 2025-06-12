import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from pyngrok import ngrok
from sklearn.preprocessing import StandardScaler, LabelEncoder



THRESHOLDS = {
    'pm25': 35,
    'pm10': 50,
    'so2': 20,
    'co': 9,
    'o3': 100,
    'no2': 40,
}

COLUMN_LABELS = {
    'pm25': "PM2.5 (µg/m³)",
    'pm10': "PM10 (µg/m³)",
    'so2': "SO₂ (ppb)",
    'co': "CO (ppm)",
    'o3': "O₃ (ppb)",
    'no2': "NO₂ (ppb)"
}

# HEADER
st.set_page_config(page_title="Air Quality Predictor", layout="wide")
st.markdown(
    "<h1 style='text-align: center;'>🌍 Air Quality Prediction</h1>",
    unsafe_allow_html=True
)
st.markdown(
    """
    <div style='text-align: center; font-size: 24px;'>
        Memprediksi kategori kualitas udara (
        <span style='color: green; font-weight: bold;'>BAIK</span> /
        <span style='color: orange; font-weight: bold;'>SEDANG</span> /
        <span style='color: red; font-weight: bold;'>TIDAK SEHAT</span>)
        berdasarkan tingkat polutan.
    </div>
    """,
    unsafe_allow_html=True
)



st.markdown("\n")
st.markdown("\n")
st.markdown("---")


# LOAD MODEL AND SIDEBAR
@st.cache_resource
def load_model_and_scalers():
    model = tf.keras.models.load_model('air_quality_model.h5')
    scaler = joblib.load('scaler.pkl')
    le = joblib.load('label_encoder.pkl')
    return model, scaler, le

model, scaler, le = load_model_and_scalers()


st.sidebar.header("🔧 Mengatur Tingkat Polutan")
def get_user_input():
    pm25 = st.sidebar.slider('PM2.5 (µg/m³)', 0.0, 150.0, 35.0)
    st.sidebar.caption("**PM2.5 (Partikel Halus):** Partikel kecil di udara yang berukuran kurang dari 2,5 mikrometer; dapat menembus jauh ke dalam paru-paru.")

    pm10 = st.sidebar.slider('PM10 (µg/m³)', 0.0, 200.0, 60.0)
    st.sidebar.caption("**PM10 (Partikel Kasar):** Partikel yang lebih besar yang dapat mengiritasi mata, hidung, dan tenggorokan.")

    so2 = st.sidebar.slider('SO2 (ppb)', 0.0, 50.0, 5.0)
    st.sidebar.caption("**SO₂ (Sulfur Dioksida):** Gas yang dihasilkan dari pembakaran bahan bakar fosil; dapat memperparah asma dan berkontribusi terhadap hujan asam.")

    co = st.sidebar.slider('CO (ppm)', 0.0, 100.0, 50.0)
    st.sidebar.caption("**CO (Karbon Monoksida):** Gas yang tidak berwarna dan tidak berbau yang dapat mengganggu transportasi oksigen dalam darah.")

    o3 = st.sidebar.slider('O3 (ppb)', 0.0, 150.0, 30.0)
    st.sidebar.caption("**O₃ (Ozon):** Gas reaktif yang dapat menyebabkan nyeri dada, batuk, dan radang saluran napas.")

    no2 = st.sidebar.slider('NO2 (ppb)', 0.0, 50.0, 15.0)
    st.sidebar.caption("**NO₂ (Nitrogen Dioksida):** Gas yang dipancarkan dari kendaraan dan pembangkit listrik; dapat mengiritasi saluran udara dan mengurangi fungsi paru-paru.")

    return {
        'pm25': pm25,
        'pm10': pm10,
        'so2': so2,
        'co': co,
        'o3': o3,
        'no2': no2
    }

user_input = get_user_input()

# INPUT TABLE
def highlight_critical(val, threshold):
    if val > threshold:
        return 'background-color: red; color: white; font-weight:bold;'
    elif val > 0.8 * threshold:
        return 'background-color: orange; color: black;'
    else:
        return ''

original_names = list(THRESHOLDS.keys())
pretty_names = [COLUMN_LABELS[k] for k in original_names]

def highlight_table(row):
    return [
        highlight_critical(
            row[COLUMN_LABELS[orig]], THRESHOLDS[orig]
        ) for orig in original_names
    ]

input_df = pd.DataFrame([user_input])
input_df_display = input_df.rename(columns=COLUMN_LABELS)
styled_df = input_df_display.style.apply(highlight_table, axis=1)

table_style = """
<style>
    table {
        font-size: 20px !important;
        padding: 10px !important;
    }
    th, td {
        padding: 12px 20px !important;
    }
</style>
"""

st.subheader("📊 Nilai Input")
st.write("""
<div style="font-size:24px; margin-bottom:10px;">
    <span style="background-color:red;color:white;font-weight:bold;">Tanda</span> melebihi ambang batas aman.
    <span style="background-color:orange;color:black;">Tanda</span> mendekati ambang batas.
</div>
""", unsafe_allow_html=True)

st.write(table_style + styled_df.to_html(), unsafe_allow_html=True)



# PREDICTION RESULT
def predict_quality(input_dict):
    X = np.array([[input_dict['pm25'],
                  input_dict['pm10'],
                  input_dict['so2'],
                  input_dict['co'],
                  input_dict['o3'],
                  input_dict['no2']]])
    X_scaled = scaler.transform(X)
    probabilities = model.predict(X_scaled, verbose=0)[0]
    predicted_class = le.inverse_transform([np.argmax(probabilities)])[0]
    return predicted_class, probabilities

pred_class, probs = predict_quality(user_input)


def get_aq_color(category):
    if category == "BAIK":
        return "green"
    elif category == "SEDANG":
        return "orange"
    else:
        return "red"
    
st.markdown("\n")    
st.markdown("\n")    
st.subheader("🔮 Hasil Prediksi")
aq_color = get_aq_color(pred_class)
st.markdown(
    f"<div style='background-color:{aq_color};color:white;padding:1em;border-radius:10px;font-size:2em;font-weight:bold;text-align:center;'>{pred_class}</div>",
    unsafe_allow_html=True,
)
st.markdown(
    f"<div style='background-color:#222;color:white;padding:0.5em;border-radius:7px;font-size:1.5em;font-weight:bold;text-align:center;'>Skor Confidence: {max(probs)*100:.1f}%</div>",
    unsafe_allow_html=True,
)

def get_health_recommendation(category):
    if category == "BAIK":
        return ("Kualitas udara baik. Tidak perlu tindakan pencegahan kesehatan yang diperlukan.", "green")
    elif category == "SEDANG":
        return ("Kualitas udara dapat diterima. Kelompok yang sensitif harus mempertimbangkan untuk mengurangi aktivitas di luar ruangan dalam waktu tertentu.", "orange")
    else:
        return ("Membatasi aktivitas di luar ruangan dalam waktu lama. Anak-anak, orang tua, dan kelompok yang memiliki masalah pernapasan harus tetap berada di dalam ruangan.", "red")

recommendation_text, color = get_health_recommendation(pred_class)

st.markdown(
    f"""
    <div style="text-align: center; color: {color}; font-size: 24px; font-weight: bold;">
        {recommendation_text}
    </div>
    """,
    unsafe_allow_html=True
)


st.markdown("\n")
st.markdown("---")
st.markdown("\n")
st.subheader("🧮 Distribusi Probabilitas")



# PROBABILITY PIE CHART
fig, ax = plt.subplots(figsize=(4, 5))
wedges, texts, autotexts = ax.pie(
    probs,
    labels=None,
    autopct='%1.1f%%',
    colors=['#4CAF50', '#FFC107', '#F44336'],
    startangle=90,
    pctdistance=0.85,
    wedgeprops={'linewidth': 1, 'edgecolor': 'white'},
    textprops={'fontsize': 9}
)
plt.setp(autotexts, size=9, weight="bold", color="white")
plt.setp(texts, size=9)
centre_circle = plt.Circle((0,0), 0.70, fc='white')
fig.gca().add_artist(centre_circle)
fig.patch.set_alpha(0)
ax.set_facecolor("none")
ax.legend(le.classes_, title="Kategori Kualitas Udara", loc="center left", bbox_to_anchor=(1, 0.5), fontsize=9)
ax.axis('equal')
col1, col2, col3 = st.columns([1,2,1])
with col2:
    st.pyplot(fig, use_container_width=True)


    
# FOOTER
st.markdown("---")
st.caption("Model dilatih menggunakan data kualitas udara Jakarta 2010-2025")
st.caption("By CC25-CR422")