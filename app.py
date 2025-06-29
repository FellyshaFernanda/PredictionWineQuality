import streamlit as st
import pandas as pd
import pickle
import numpy as np

# --- KONFIGURASI HALAMAN ---
st.set_page_config(
    page_title="Prediksi Kualitas Wine",
    page_icon="🍷",
    layout="wide"
)

# --- FUNGSI-FUNGSI ---

@st.cache_resource
def load_model(model_path):
    """Memuat model machine learning dari file .pkl."""
    try:
        with open(model_path, 'rb') as file:
            model = pickle.load(file)
        return model
    except FileNotFoundError:
        st.error(f"Error: File model '{model_path}' tidak ditemukan.")
        st.info("Pastikan Anda sudah menjalankan skrip 'buat_model_kategori_smote.py' untuk membuat model terbaru.")
        return None

@st.cache_data
def load_data(data_path):
    """Memuat data CSV untuk mendapatkan nama kolom dan nilai default."""
    try:
        df = pd.read_csv(data_path)
        return df
    except FileNotFoundError:
        st.error(f"Error: File data '{data_path}' tidak ditemukan.")
        return None

def user_input_features(columns, data_frame):
    """Membuat widget input manual di sidebar."""
    inputs = {}
    st.sidebar.header("Input Fitur Wine (Manual)")
    for feature in columns:
        label = feature.replace('_', ' ').title()
        default_val = float(data_frame[feature].mean())
        inputs[feature] = st.sidebar.number_input(
            label,
            value=default_val,
            format="%.4f"
        )
    return pd.DataFrame([inputs], columns=columns)

# --- MEMUAT DATA DAN MODEL ---
model = load_model('wine_model.pkl')
df_raw = load_data('wine_quality.csv')

if model is None or df_raw is None:
    st.stop()

X_columns = df_raw.drop('quality', axis=1, errors='ignore').columns

# --- TATA LETAK UTAMA ---
st.title("🍷 Aplikasi Prediksi Kategori Kualitas Wine")
st.write(
    "Aplikasi ini menggunakan _machine learning_ untuk memprediksi apakah sebuah wine termasuk dalam kategori **Good**, **Middle**, atau **Bad**."
)
st.markdown("---")

input_df = user_input_features(X_columns, df_raw)

st.subheader("Ringkasan Fitur yang Anda Masukkan:")
st.dataframe(input_df, use_container_width=True)

if st.button("✨ Prediksi Kategori Kualitas", type="primary", use_container_width=True):
    prediction = model.predict(input_df)
    hasil_prediksi = prediction[0]

    st.subheader("Hasil Prediksi")
    
    col1, col2 = st.columns([1, 2])

    with col1:
        if hasil_prediksi == 'Good':
            st.success(f"✅ Kategori: **{hasil_prediksi}**")
            # st.balloons() # <-- BARIS INI TELAH DIHAPUS
        elif hasil_prediksi == 'Middle':
            st.info(f"😐 Kategori: **{hasil_prediksi}**")
        else: # 'Bad'
            st.warning(f"👎 Kategori: **{hasil_prediksi}**")
    
    with col2:
        if hasattr(model, 'predict_proba'):
            st.markdown("**Tingkat Kepercayaan Model (Probabilitas)**")
            prediction_proba = model.predict_proba(input_df)
            
            proba_df = pd.DataFrame(
                prediction_proba,
                columns=model.classes_
            ).T
            proba_df.columns = ['Probabilitas']
            proba_df.sort_values(by='Probabilitas', ascending=False, inplace=True)
            
            st.bar_chart(proba_df)

# --- BAGIAN INFORMASI TAMBAHAN ---
st.markdown("---")
with st.expander("ℹ️ Tentang Model dan Kategori yang Digunakan"):
    st.markdown("""
    Aplikasi ini menggunakan model **Random Forest Classifier** yang dilatih untuk mengklasifikasikan wine ke dalam tiga kategori berdasarkan 11 fitur fisiko-kimiawinya. Untuk mengatasi data yang tidak seimbang, proses training menggunakan teknik **SMOTE-Tomek**.

    **Definisi Kategori:**
    - **Good**: Skor kualitas asli 7 atau 8.
    - **Middle**: Skor kualitas asli 5 atau 6.
    - **Bad**: Skor kualitas asli 3 atau 4.

    **Performa Model (Berdasarkan Hasil Training Anda):**
    Laporan di bawah ini adalah **hasil nyata** dari model yang digunakan, dievaluasi pada data yang belum pernah dilihat sebelumnya.
    """)
    
    classification_report_text = """
                  precision    recall  f1-score   support

             Bad       0.27      0.23      0.25        13
            Good       0.57      0.77      0.65        43
          Middle       0.92      0.88      0.90       264

        accuracy                           0.83       320
       macro avg       0.59      0.62      0.60       320
    weighted avg       0.85      0.83      0.84       320
    """
    st.code(classification_report_text, language='text')
    st.markdown("""
    **Cara Membaca:**
    - **Accuracy (0.83)**: Secara keseluruhan, model menebak dengan benar sekitar 83% dari waktu.
    - **Precision (Middle: 0.92)**: Jika model memprediksi 'Middle', 92% kemungkinannya prediksi itu benar.
    - **Recall (Good: 0.77)**: Model ini berhasil mengidentifikasi 77% dari semua wine 'Good' yang sebenarnya.
    - **f1-score (Bad: 0.25)**: Performa untuk kategori 'Bad' masih rendah karena jumlah datanya sangat sedikit, jadi prediksi untuk kategori ini kurang bisa diandalkan.
    """)