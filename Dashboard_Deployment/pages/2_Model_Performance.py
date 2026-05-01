"""Streamlit page: Model Performance (training results & confusion matrix)."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import CONFUSION_MATRIX_PATH
from assets.class_info import MODEL_METRICS_PER_CLASS, MODEL_METRICS_OVERALL
from core.visualization import make_metric_bar
from core.branding import render_page_header, render_footer

st.set_page_config(
    page_title="Model Performance | Delima Ester Purba",
    page_icon="📈",
    layout="wide",
)

render_page_header(
    page_label="Model Performance",
    subtitle=(
        "Evaluasi model <b>progressive_yolov8n</b> (<i>best.pt</i>) pada "
        "validation set: 3.595 gambar &middot; 15.379 instances."
    ),
    compact=True,
)

# ------------------------------------------------------------------
# Overall metrics
# ------------------------------------------------------------------
st.markdown(
    "<div class='section-title'>Metrik Keseluruhan (Validation Set)</div>",
    unsafe_allow_html=True,
)
ov = MODEL_METRICS_OVERALL
c1, c2, c3, c4 = st.columns(4)
c1.metric("Precision", f"{ov['P']:.3f}")
c2.metric("Recall", f"{ov['R']:.3f}")
c3.metric("mAP@50", f"{ov['mAP50']:.3f}")
c4.metric("mAP@50-95", f"{ov['mAP50_95']:.3f}")

c1, c2, c3 = st.columns(3)
c1.metric("Parameters", ov["params"])
c2.metric("GFLOPs", ov["gflops"])
c3.metric(
    "Inference Speed",
    f"{ov['speed_ms']['inference']:.1f} ms / image",
    help="Pre + Inference + Post = "
    f"{ov['speed_ms']['preprocess']:.1f} + "
    f"{ov['speed_ms']['inference']:.1f} + "
    f"{ov['speed_ms']['postprocess']:.1f} ms",
)

st.divider()

# ------------------------------------------------------------------
# Confusion matrix + per-class metrics
# ------------------------------------------------------------------
left, right = st.columns([1, 1])

with left:
    st.markdown(
        "<div class='section-title'>Confusion Matrix (Normalized)</div>",
        unsafe_allow_html=True,
    )
    cm_path = Path(CONFUSION_MATRIX_PATH)
    if cm_path.exists():
        st.image(str(cm_path), use_container_width=True)
    else:
        st.error(f"File confusion matrix tidak ditemukan: {cm_path}")

with right:
    st.markdown(
        "<div class='section-title'>Metrik per Kelas</div>",
        unsafe_allow_html=True,
    )
    rows = []
    for cls, m in MODEL_METRICS_PER_CLASS.items():
        rows.append({
            "Class": cls,
            "Images": m["images"],
            "Instances": m["instances"],
            "P": round(m["P"], 3),
            "R": round(m["R"], 3),
            "mAP50": round(m["mAP50"], 3),
            "mAP50-95": round(m["mAP50_95"], 3),
        })
    df = pd.DataFrame(rows).sort_values("mAP50", ascending=False).reset_index(drop=True)
    st.dataframe(df, use_container_width=True, hide_index=True)

st.divider()

# ------------------------------------------------------------------
# Bar chart mAP50
# ------------------------------------------------------------------
st.markdown(
    "<div class='section-title'>mAP@50 per Kelas</div>", unsafe_allow_html=True,
)
st.plotly_chart(make_metric_bar(MODEL_METRICS_PER_CLASS, "mAP50"), use_container_width=True)

c1, c2 = st.columns(2)
with c1:
    st.markdown(
        "<div class='section-title'>mAP@50-95 per Kelas</div>",
        unsafe_allow_html=True,
    )
    st.plotly_chart(make_metric_bar(MODEL_METRICS_PER_CLASS, "mAP50_95"), use_container_width=True)
with c2:
    st.markdown(
        "<div class='section-title'>Recall per Kelas</div>",
        unsafe_allow_html=True,
    )
    st.plotly_chart(make_metric_bar(MODEL_METRICS_PER_CLASS, "R"), use_container_width=True)

st.divider()

# ------------------------------------------------------------------
# Insights
# ------------------------------------------------------------------
st.markdown(
    "<div class='section-title'>Interpretasi Singkat</div>",
    unsafe_allow_html=True,
)

st.markdown("""
**Kelas dengan performa terbaik:**
- `nodule`        : mAP@50 = 0.993 - bentuk lesi sangat khas (besar, menonjol).
- `pustule`       : mAP@50 = 0.957 - kontras putih/kuning di puncak mudah dikenali.
- `nevus`         : mAP@50 = 0.911 - bercak gelap dengan border jelas.
- `hypertrophic_scar`: mAP@50 = 0.907.

**Kelas yang masih lemah:**
- `comedo`        : mAP@50 = 0.507 - ukuran sangat kecil dan sering tertukar dengan background/papule.
- `atrophic_scar` : mAP@50 = 0.627 - tekstur halus, sulit dipisahkan dari kulit normal.
- `other`         : mAP@50 = 0.371 - kelas heterogen dengan sample paling sedikit (69 instances).

**Pola kebingungan utama (dari confusion matrix normalized):**
- `comedo` <-> `background` (~0.43): banyak comedo kecil tidak terdeteksi.
- `atrophic_scar` <-> `background` (~0.33): textural lesion sulit dibedakan dari kulit.
- `other` <-> `background` (~0.43): mayoritas instance "other" terlewat.

**Rekomendasi peningkatan:**
1. Tambah augmentasi spesifik untuk lesi kecil (mosaic, scale-up crops) agar comedo lebih terdeteksi.
2. Pertimbangkan pre-processing CLAHE untuk meningkatkan kontras tekstur scar.
3. Re-balance dataset: instance `other` dan `nodule` jauh lebih sedikit dibanding kelas lain.
""")

render_footer()
