"""Entry point for the Acne Analyzer Streamlit dashboard.

Run with:
    streamlit run app.py

Streamlit akan otomatis men-discover halaman di folder `pages/` untuk
navigasi sidebar:
- 1_Inference.py         -> Inference (upload / Raspberry Pi)
- 2_Model_Performance.py -> Model Performance & Confusion Matrix
"""
from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st


# ---------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.branding import render_page_header, render_footer  # noqa: E402


# ---------------------------------------------------------------
# Page config
# ---------------------------------------------------------------
st.set_page_config(
    page_title="Deteksi Lesi Kulit YOLOv8n | Delima Ester Purba",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------------
# Header (logo + hero + author)
# ---------------------------------------------------------------
render_page_header(
    page_label="Beranda",
    subtitle=(
        "Dashboard analisis berbasis <b>YOLOv8n</b> untuk membantu proses "
        "identifikasi dan penghitungan lesi kulit secara otomatis dan objektif."
    ),
    compact=False,
)


# ---------------------------------------------------------------
# Deskripsi utama
# ---------------------------------------------------------------
st.markdown(
    """
    Selamat datang di **Dashboard Analisis Lesi Kulit** berbasis **YOLOv8n**.
    Aplikasi ini dikembangkan sebagai bagian dari penelitian skripsi untuk
    mendeteksi delapan kelas lesi kulit berikut, ditambah satu kelas tambahan
    (`other`). Sistem memanfaatkan model **computer vision** untuk membantu
    proses identifikasi.
    """
)

st.markdown(
    """
    <div class="chip-row">
        <span class="chip">atrophic_scar</span>
        <span class="chip">comedo</span>
        <span class="chip">hypertrophic_scar</span>
        <span class="chip">melasma</span>
        <span class="chip">nevus</span>
        <span class="chip">nodule</span>
        <span class="chip">papule</span>
        <span class="chip">pustule</span>
        <span class="chip other">other</span>
    </div>
    """,
    unsafe_allow_html=True,
)

st.write("")


# ---------------------------------------------------------------
# Fitur utama
# ---------------------------------------------------------------
st.markdown("<div class='section-title'>Fitur Utama</div>", unsafe_allow_html=True)
st.markdown(
    """
    <div class="feature-grid">
        <div class="feature-card">
            <div class="feature-icon">📤</div>
            <div class="feature-title">Dua Sumber Input</div>
            <p class="feature-desc">
                Upload foto dari perangkat (ukuran bebas) atau capture langsung
                dari kamera <b>Raspberry Pi 12 MP</b> via SSH.
            </p>
        </div>
        <div class="feature-card">
            <div class="feature-icon">🧩</div>
            <div class="feature-title">Dynamic Padding + Sliding Window</div>
            <p class="feature-desc">
                Padding otomatis ke kelipatan 640 (<i>gray-114</i>, match training),
                lalu tile 640&times;640 dengan overlap 128&nbsp;px.
            </p>
        </div>
        <div class="feature-card">
            <div class="feature-icon">📊</div>
            <div class="feature-title">Statistik Objektif</div>
            <p class="feature-desc">
                Jumlah total lesi, distribusi per kelas, rata-rata ukuran bbox,
                dan luas area terdeteksi — fokus pada data kuantitatif.
            </p>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------
# Navigasi
# ---------------------------------------------------------------
st.markdown("<div class='section-title'>Navigasi Dashboard</div>", unsafe_allow_html=True)
nav1, nav2 = st.columns(2)
with nav1:
    st.markdown(
        """
        **🔬 Inference**  
        Unggah foto atau ambil langsung dari Raspberry Pi, lalu jalankan
        inferensi <i>dynamic-padding + sliding-window 640&times;640</i>.
        Hasilnya berupa gambar ter-anotasi dan statistik lesi.
        """,
        unsafe_allow_html=True,
    )
with nav2:
    st.markdown(
        """
        **📈 Model Performance**  
        Confusion matrix ter-normalisasi dan metrik per kelas
        (Precision / Recall / mAP@50 / mAP@50-95) dari proses training.
        """,
        unsafe_allow_html=True,
    )

st.info(
    "Gunakan menu di **sidebar kiri** untuk membuka halaman "
    "**Inference** atau **Model Performance**.",
    icon="➡️",
)


# ---------------------------------------------------------------
# Panduan singkat
# ---------------------------------------------------------------
with st.expander("📘 Panduan Pemakaian Singkat", expanded=False):
    st.markdown(
        """
        1. Buka halaman **Inference** dari sidebar.
        2. Pilih **sumber gambar**:
           - *Upload Foto* — ukuran bebas, langsung diproses.
           - *Raspberry Pi* — klik **Start Preview** (opsional) untuk
             audit posisi, lalu **Ambil Foto 12 MP**.
        3. Atur parameter di sidebar bila perlu (Confidence, IoU, Overlap).
        4. Klik **RUN INFERENCE**.
        5. Lihat ringkasan statistik, distribusi, tabel detail lesi,
           dan keterangan warna di tab hasil.
        6. Unduh gambar hasil deteksi sebagai **PNG** bila diperlukan.
        """
    )


# ---------------------------------------------------------------
# Disclaimer
# ---------------------------------------------------------------
st.warning(
    "**Disclaimer.** Aplikasi ini merupakan alat bantu berbasis "
    "**computer vision** untuk penghitungan dan identifikasi lesi kulit "
    "secara objektif. Penilaian keparahan (*severity*) dan diagnosis klinis "
    "tetap merupakan kewenangan **dokter spesialis kulit (dermatolog)**.",
    icon="⚕️",
)


render_footer()
