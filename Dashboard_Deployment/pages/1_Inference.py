"""Streamlit page: Inference.

Pipeline baru:
- Input gambar ukuran bebas (upload / Raspberry Pi).
- Padding putih dinamis ke kelipatan 640x640.
- Skenario A (padded == 640x640) -> inference langsung.
- Skenario B (lebih besar) -> sliding window 640x640 + overlap + skin-mask skip.
- NMS antar jendela + remap ke koordinat gambar ASLI.
"""
from __future__ import annotations

import sys
from pathlib import Path
import io

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import (
    RAW_W, RAW_H, DEFAULT_CONF, DEFAULT_IOU,
    TILE, OVERLAP_PX, USE_FACE_MASK, MIN_SKIN_RATIO,
    PI_DEFAULT_HOST, PI_DEFAULT_USER, PI_DEFAULT_PASS,
)
from core.inference import run_full_inference, load_model
from core.raspberry_pi import (
    capture_high_res, start_live_preview, stop_live_preview, is_preview_running,
)
from core.severity import build_summary, build_detection_table, build_text_conclusion
from core.visualization import (
    draw_bboxes, crop_bbox, make_bar_chart, build_legend_html,
)
from core.branding import render_page_header, render_footer
from assets.class_info import CLASS_INFO, SARAN_PER_KELAS

st.set_page_config(
    page_title="Inference | Delima Ester Purba",
    page_icon="🔬",
    layout="wide",
)

render_page_header(
    page_label="Inference",
    subtitle=(
        "Inferensi <b>YOLOv8n</b> dengan pipeline <i>dynamic padding</i> "
        "(kelipatan 640) + <i>sliding window</i>. "
        "Mendukung gambar ukuran bebas."
    ),
    compact=True,
)

# ------------------------------------------------------------------
# Session state init
# ------------------------------------------------------------------
for key, default in [
    ("raw_image", None),
    ("raw_image_source", None),
    ("detections", None),
    ("summary", None),
    ("rows", None),
    ("annotated", None),
    ("selected_id", None),
]:
    if key not in st.session_state:
        st.session_state[key] = default


# ------------------------------------------------------------------
# Sidebar - Source & Settings
# ------------------------------------------------------------------
st.sidebar.header("Sumber Gambar")
source = st.sidebar.radio(
    "Pilih sumber",
    ["Upload Foto", "Raspberry Pi"],
    horizontal=True,
)

if source == "Upload Foto":
    uploaded = st.sidebar.file_uploader(
        "Upload foto (ukuran bebas)",
        type=["png", "jpg", "jpeg"],
    )
    if uploaded is not None:
        try:
            img = Image.open(uploaded).convert("RGB")
            arr = np.array(img)
            st.sidebar.info(
                f"Ukuran input: {arr.shape[1]} x {arr.shape[0]} px. "
                f"Akan di-padding otomatis ke kelipatan {TILE}."
            )
            st.session_state.raw_image = arr
            st.session_state.raw_image_source = f"Upload: {uploaded.name}"
            st.session_state.detections = None
        except Exception as e:
            st.sidebar.error(f"Gagal membaca gambar: {e}")

else:
    st.sidebar.subheader("Raspberry Pi Credentials")
    pi_host = st.sidebar.text_input("IP Address", value=PI_DEFAULT_HOST)
    pi_user = st.sidebar.text_input("Username", value=PI_DEFAULT_USER)
    pi_pass = st.sidebar.text_input("Password", type="password", value=PI_DEFAULT_PASS)

    st.sidebar.markdown("**Live Preview (jendela OpenCV lokal)**")
    col_p1, col_p2 = st.sidebar.columns(2)
    if col_p1.button("Start Preview", use_container_width=True):
        ok, msg = start_live_preview(pi_host, pi_user, pi_pass)
        (st.sidebar.success if ok else st.sidebar.error)(msg)
    if col_p2.button("Stop Preview", use_container_width=True):
        ok, msg = stop_live_preview()
        st.sidebar.info(msg)

    if is_preview_running():
        st.sidebar.success("Preview aktif. Cek jendela OpenCV di taskbar laptop.")

    st.sidebar.markdown("**Capture Foto Resolusi Tinggi**")
    if st.sidebar.button(f"Ambil Foto 12MP ({RAW_W}x{RAW_H})", use_container_width=True, type="primary"):
        with st.spinner("Mengambil gambar dari Raspberry Pi..."):
            ok, msg, path = capture_high_res(pi_host, pi_user, pi_pass)
        if ok and path is not None:
            st.sidebar.success(msg)
            img = Image.open(path).convert("RGB")
            arr = np.array(img)
            st.session_state.raw_image = arr
            st.session_state.raw_image_source = f"Raspberry Pi: {path.name}"
            st.session_state.detections = None
        else:
            st.sidebar.error(msg)

st.sidebar.divider()
st.sidebar.header("Pengaturan Inferensi")
conf_th = st.sidebar.slider(
    "Confidence threshold",
    0.05, 0.95, DEFAULT_CONF, 0.05,
    help="Match training eval: 0.25",
)
iou_th = st.sidebar.slider(
    "IoU threshold (per-tile NMS)",
    0.1, 0.9, DEFAULT_IOU, 0.05,
    help="Match training eval: 0.45. Jangan naikkan tinggi tinggi kalau "
    "hasilnya jadi banyak duplikat.",
)

st.sidebar.markdown("**Sliding Window**")
overlap_px = st.sidebar.slider(
    "Overlap antar jendela (px)",
    min_value=0,
    max_value=TILE // 2,
    value=OVERLAP_PX,
    step=32,
    help="0 = tanpa overlap (tile berdampingan). "
    f"Default {OVERLAP_PX} px (20% dari {TILE}).",
)

# Skin-mask skip & parameter terkait sengaja TIDAK diekspos ke dashboard
# (permintaan revisi). Backend tetap mengaktifkannya memakai
# `USE_FACE_MASK` & `MIN_SKIN_RATIO` dari `config.py` supaya tile tanpa
# kulit tetap dilewati otomatis demi kecepatan inferensi.
use_mask = USE_FACE_MASK
min_skin = MIN_SKIN_RATIO

run_btn = st.sidebar.button(
    "RUN INFERENCE",
    use_container_width=True,
    type="primary",
    disabled=st.session_state.raw_image is None,
)


# ------------------------------------------------------------------
# Run inference
# ------------------------------------------------------------------
if run_btn and st.session_state.raw_image is not None:
    img = st.session_state.raw_image
    st.subheader("Menjalankan Inferensi...")
    progress = st.progress(0.0, text="0 / ? tile")
    info_box = st.empty()

    def _cb(done: int, total: int):
        progress.progress(
            done / max(total, 1),
            text=f"{done} / {total} tile",
        )

    try:
        load_model()
        det = run_full_inference(
            img,
            conf=conf_th,
            iou=iou_th,
            overlap_px=overlap_px,
            use_face_mask=use_mask,
            min_skin_ratio=min_skin,
            progress_cb=_cb,
        )
        st.session_state.detections = det
        st.session_state.summary = build_summary(det)
        st.session_state.rows = build_detection_table(det)

        annotated = draw_bboxes(img, det, show_labels=False)
        st.session_state.annotated = annotated
        st.session_state.selected_id = None

        progress.empty()
        info_box.success(
            f"Selesai (Skenario {det['scenario']}). "
            f"{det['boxes'].shape[0]} lesi terdeteksi dari "
            f"{det['tiles_inferred']}/{det['tile_count']} tile "
            f"({det['tiles_skipped']} skip via mask) dalam "
            f"{det['inference_time_ms']/1000:.2f} detik."
        )
    except Exception as e:
        progress.empty()
        st.error(f"Inferensi gagal: {e}")


# ------------------------------------------------------------------
# Show preview when no inference yet
# ------------------------------------------------------------------
if st.session_state.raw_image is not None and st.session_state.detections is None:
    st.subheader("Pratinjau Gambar Input")
    st.caption(st.session_state.raw_image_source or "")
    st.image(st.session_state.raw_image, use_container_width=True)
    st.info("Klik **RUN INFERENCE** di sidebar untuk memulai analisis.")

if st.session_state.raw_image is None and st.session_state.detections is None:
    st.info("Pilih sumber gambar di sidebar (Upload atau Raspberry Pi) untuk mulai.")


# ------------------------------------------------------------------
# Results
# ------------------------------------------------------------------
if st.session_state.detections is not None:
    det = st.session_state.detections
    summary = st.session_state.summary
    rows = st.session_state.rows
    img = st.session_state.raw_image
    annotated = st.session_state.annotated

    # ----- Score based on statistic -----
    st.markdown("### Score based on statistic")
    face_cov = float(summary.get("face_coverage_pct", 0.0))
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Lesi", summary["total"])
    col2.metric("Kelas Dominan", summary["dominant"] or "-")
    col3.metric("Lesi Inflamasi", summary["n_inflammatory"])
    col4.metric(
        "Cakupan Lesi di Wajah",
        f"{face_cov:.1f}%",
        help=(
            "Persentase luas wajah yang tertutup oleh kumpulan bounding "
            "box lesi. Denominator = area kulit wajah hasil skin-mask; "
            "numerator = union bbox lesi ∩ area kulit."
        ),
    )

    dev_raw = str(det.get("device_used", "-"))
    if dev_raw.isdigit():
        device_label = f"GPU (CUDA:{dev_raw})"
    elif dev_raw.lower().startswith("cuda"):
        device_label = f"GPU ({dev_raw})"
    elif dev_raw.lower() == "cpu":
        device_label = "CPU"
    else:
        device_label = dev_raw

    pi = det.get("pad_info")
    with st.expander("Info Pipeline"):
        st.markdown(
            f"- **Skenario:** {det.get('scenario', '-')} "
            f"({'padding tepat 640x640 - direct inference' if det.get('scenario') == 'A' else 'sliding window + overlap'})\n"
            f"- **Device:** `{device_label}`\n"
            f"- **Ukuran Original:** {img.shape[1]} x {img.shape[0]} px\n"
            f"- **Ukuran Padded:** {pi.padded_w} x {pi.padded_h} px "
            f"(+{pi.pad_left}L / +{pi.pad_right}R / +{pi.pad_top}T / +{pi.pad_bottom}B) \n"
            f"- **Jumlah Tile:** {det.get('tile_count', 0)} "
            f"(inferred {det.get('tiles_inferred', 0)}, skipped {det.get('tiles_skipped', 0)})\n"
            f"- **Waktu Inferensi:** {det.get('inference_time_ms', 0)/1000:.2f} s"
        )

    st.divider()

    # ----- Tabs -----
    tab_det, tab_dist, tab_table, tab_edu = st.tabs([
        "Deteksi", "Distribusi", "Detail Lesi", "Edukasi Kelas",
    ])

    # === Tab Deteksi ===
    with tab_det:
        show_box = st.checkbox("Tampilkan bounding box", value=True)
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Gambar Asli**")
            st.image(img, use_container_width=True)
        with c2:
            st.markdown("**Hasil Deteksi**")
            st.image(annotated if show_box else img, use_container_width=True)

        st.markdown("#### Keterangan Warna per Kelas")
        st.caption(
            "Bounding box hanya ditampilkan sebagai kotak berwarna tanpa "
            "teks. Gunakan tabel keterangan di bawah ini untuk memetakan "
            "warna -> kelas lesi, beserta jumlah deteksi pada gambar."
        )
        st.markdown(
            build_legend_html(summary["per_class_count"]),
            unsafe_allow_html=True,
        )

    # === Tab Distribusi ===
    with tab_dist:
        if summary["total"] == 0:
            st.info("Tidak ada lesi terdeteksi, distribusi tidak tersedia.")
        else:
            st.plotly_chart(
                make_bar_chart(summary["per_class_count"]),
                use_container_width=True,
            )

    # === Tab Detail Lesi ===
    with tab_table:
        if not rows:
            st.info("Tidak ada lesi terdeteksi.")
        else:
            df = pd.DataFrame(rows)
            # Gabung nama kelas + penjelasan singkat supaya user awam
            # langsung paham tanpa harus pindah ke tab edukasi.
            def _fmt_class(name: str) -> str:
                info = CLASS_INFO.get(name, {})
                label = info.get("label", name)
                simple = info.get("simple", "").strip()
                return f"{label} — {simple}" if simple else label

            df["Kelas"] = df["Class"].map(_fmt_class)

            display_cols = [
                "ID", "Kelas", "Confidence (%)",
                "Width (px)", "Height (px)", "Area (px)",
                "Relative Size (%)",
            ]
            display_df = df[display_cols]
            event = st.dataframe(
                display_df,
                use_container_width=True,
                hide_index=True,
                on_select="rerun",
                selection_mode="single-row",
                key="detail_table",
                column_config={
                    "Kelas": st.column_config.TextColumn(
                        "Kelas (Nama + Penjelasan)",
                        width="large",
                    ),
                },
            )
            sel_rows = event.selection.get("rows", []) if hasattr(event, "selection") else []
            if sel_rows:
                idx = sel_rows[0]
                r = rows[idx]
                st.session_state.selected_id = r["ID"]
                st.markdown(
                    f"#### Crop Zoom - ID {r['ID']} ({r['Class']}, {r['Confidence (%)']}%)"
                )
                crop = crop_bbox(img, (r["x1"], r["y1"], r["x2"], r["y2"]))
                cz1, cz2 = st.columns([2, 1])
                with cz1:
                    st.image(crop, use_container_width=True)
                with cz2:
                    info = CLASS_INFO.get(r["Class"], {})
                    st.markdown(f"**Kelas:** {info.get('label', r['Class'])}")
                    st.markdown(f"**Penjelasan:** {info.get('simple', '-')}")
                    st.markdown(f"**Ciri Khas:** {info.get('ciri', '-')}")
                    st.markdown(f"**Kategori:** {info.get('kategori', '-')}")
                    st.markdown(f"**Saran:** {SARAN_PER_KELAS.get(r['Class'], '-')}")
                    st.markdown(
                        f"**Posisi:** ({r['x1']}, {r['y1']}) - ({r['x2']}, {r['y2']})"
                    )
                    st.markdown(f"**Ukuran:** {r['Width (px)']} x {r['Height (px)']} px")
                    st.markdown(f"**Relative Size:** {r['Relative Size (%)']}%")
            else:
                st.caption("Klik salah satu baris di tabel untuk melihat crop zoom.")

    # === Tab Edukasi ===
    with tab_edu:
        edu_rows = []
        for k, v in CLASS_INFO.items():
            edu_rows.append({
                "Kelas": v["label"],
                "Penjelasan Sederhana": v["simple"],
                "Ciri Khas": v["ciri"],
                "Kategori": v["kategori"],
                "Saran Umum": SARAN_PER_KELAS.get(k, "-"),
            })
        st.dataframe(pd.DataFrame(edu_rows), use_container_width=True, hide_index=True)

    st.divider()

    # ----- Conclusion -----
    st.markdown("### Kesimpulan & Saran")
    st.markdown(build_text_conclusion(summary))

    if summary["total"] > 0 and summary["dominant"]:
        st.info(
            f"**Saran utama:** {SARAN_PER_KELAS.get(summary['dominant'], '-')}"
        )

    # ----- Download gambar hasil deteksi -----
    st.divider()
    buf = io.BytesIO()
    Image.fromarray(annotated).save(buf, format="PNG")
    st.download_button(
        label="Download Gambar Hasil Deteksi (PNG)",
        data=buf.getvalue(),
        file_name="acne_annotated.png",
        mime="image/png",
    )


render_footer()
