import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
import streamlit as st
import plotly.express as px

# Fungsi format harga ke jutaan rupiah
def format_rupiah_juta(nilai_int):
    juta = nilai_int / 1_000_000_00
    if juta.is_integer():
        juta = int(juta)
    return f"Rp{juta} jt"

def format_megapixel(mp):
    try:
        return f"{mp:,.1f}".replace(",", "X").replace(".", ",").replace("X", ".")
    except:
        return "-"

# Fungsi konversi string harga "Rp15,499,000.00" jadi int rupiah tanpa tanda
def rupiah_to_int(rp_str):
    if isinstance(rp_str, str):
        clean_str = rp_str.replace("Rp", "").replace(".", "").replace(",", "")
        try:
            return int(clean_str)
        except:
            return 0
    return 0

# LOAD & PREPROCESS DATA
df = pd.read_csv("camera_dataset.csv", sep=';')

# Konversi harga string ke integer rupiah
df['price_int'] = df['price'].apply(rupiah_to_int)

# Bersihkan megapixel: hanya angka dan titik
df['megapixel'] = df['megapixel'].astype(str).str.replace(",", ".", regex=False)
df['megapixel'] = df['megapixel'].replace('[^0-9.]', '', regex=True)
df['megapixel'] = pd.to_numeric(df['megapixel'], errors='coerce')

# Drop data yang ada NaN di kolom penting
df = df.dropna(subset=['camera_name', 'camera_type', 'megapixel', 'price_int', 'usage'])

# Encode kategori jenis kamera dan penggunaan
le_type = LabelEncoder()
le_usage = LabelEncoder()
df['camera_type'] = df['camera_type'].astype(str)
df['usage'] = df['usage'].astype(str)
df['type_enc'] = le_type.fit_transform(df['camera_type'])
df['usage_enc'] = le_usage.fit_transform(df['usage'])

# Fitur & label
X = df[['megapixel', 'price_int', 'type_enc', 'usage_enc']]
y = df['camera_name']

# Normalisasi fitur
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train model KNN
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_scaled, y)

# STREAMLIT UI
st.set_page_config(page_title="Rekomendasi Kamera AI", layout="wide")
st.title("🤖 Rekomendasi Kamera Berbasis AI (KNN)")
st.write("Temukan kamera yang cocok berdasarkan megapiksel, anggaran, jenis kamera, dan tujuan penggunaan.")

# Sidebar input
st.sidebar.header("🎛️ Filter Preferensi Anda")
megapixels = st.sidebar.slider("Megapiksel yang Diinginkan", 8, 35, 24)
price_million = st.sidebar.slider("Anggaran Maksimal (Juta Rupiah)", 2, 25, 10)
camera_type = st.sidebar.selectbox("Jenis Kamera", le_type.classes_)
usage = st.sidebar.selectbox("Kebutuhan Penggunaan", le_usage.classes_)

# Konversi input harga jutaan ke rupiah integer
price_int = price_million * 1_000_000_00

# Encode & normalisasi input user
type_encoded = le_type.transform([camera_type])[0]
usage_encoded = le_usage.transform([usage])[0]
user_input = [[megapixels, price_int, type_encoded, usage_encoded]]
user_input_scaled = scaler.transform(user_input)

# Visualisasi data awal
df_plot = df.copy()
df_plot['price_million'] = df_plot['price_int'] / 1_000_000_00

# Rekomendasi kamera dan visualisasi 
if st.sidebar.button("🔍 Rekomendasikan Kamera"):
    neighbors = knn.kneighbors(user_input_scaled, n_neighbors=3, return_distance=False)
    st.subheader("📸 Kamera yang Direkomendasikan:")

    recommended_cameras = df.iloc[neighbors[0]]

    for _, row in recommended_cameras.iterrows():
        st.markdown(f"""
        <div style="border:1px solid #ddd; border-radius:10px; padding:15px; margin-bottom:10px;">
            <h4 style="margin:0;">{row['camera_name']}</h4>
            <ul>
                <li><strong>Jenis:</strong> {row['camera_type']}</li>
                <li><strong>Megapiksel:</strong> {format_megapixel(row['megapixel'])} MP</li>
                <li><strong>Harga:</strong> {format_rupiah_juta(row['price_int'])}</li>
                <li><strong>Penggunaan:</strong> {row['usage']}</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    # Visualisasi dengan highlight
    fig = px.scatter(
        df_plot,
        x='megapixel',
        y='price_million',
        color='usage',
        hover_name='camera_name',
        hover_data={
            'camera_type': True,
            'megapixel': ':.1f',
            'price_million': ':,.2f',
            'usage': True,
        },
       
        
    )

    # Tambahkan highlight titik rekomendasi
    fig.add_scatter(
        x=recommended_cameras['megapixel'],
        y=recommended_cameras['price_int'] / 1_000_000_00,
        mode='markers+text',
        marker=dict(size=10, color='green', symbol='star'),
        text=recommended_cameras['camera_name'],
        textposition='top center',
        name='Rekomendasi'
    )

    st.plotly_chart(fig, use_container_width=True)
else:
    # Jika belum ada rekomendasi, tampilkan grafik biasa
    fig = px.scatter(
        df_plot,
        x='megapixel',
        y='price_million',
        color='usage',
        hover_name='camera_name',
        hover_data={
            'camera_type': True,
            'megapixel': ':.1f',
            'price_million': ':,.2f',
            'usage': True,
        },
        
    )

    st.plotly_chart(fig, use_container_width=True)
