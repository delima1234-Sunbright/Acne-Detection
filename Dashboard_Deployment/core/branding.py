"""Shared branding / theme helpers untuk semua halaman Streamlit.

Semua halaman (landing `app.py`, Inference, Model Performance) memakai
header yang sama:
- CSS global (warna biru Prasmul + aksen)
- Row logo (Universitas + School of Applied STEM)
- Hero banner judul skripsi
- Kartu identitas penulis

Ini ditaruh di satu modul supaya gampang konsisten & di-update sekali.
"""
from __future__ import annotations

import base64
from pathlib import Path
from typing import Optional

import streamlit as st


# ---------------------------------------------------------------
# Paths
# ---------------------------------------------------------------
_ROOT = Path(__file__).resolve().parent.parent
_LOGO_DIR = _ROOT / "Logo"
_LOGO_UNIV = _LOGO_DIR / "Prasmul_logo_01 Biru.png"
_LOGO_STEM = _LOGO_DIR / "Logo School - Prasmul-STEM Color.png"

# Konstanta teks identitas — dipakai ulang di header & footer.
THESIS_TITLE_HTML = (
    "Implementasi Algoritma YOLOv8 untuk Deteksi Multi-Kelas<br/>"
    "Lesi Kulit: Acne Vulgaris dan Hiperpigmentasi"
)
AUTHOR_NAME = "Delima Ester Purba"
AUTHOR_PROGRAM = "Artificial Intelligence and Robotics 2022"
AUTHOR_UNIV = "Universitas Prasetiya Mulya"
AUTHOR_TAG = "Tugas Akhir"


# ---------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------
def _img_to_base64(path: Path) -> Optional[str]:
    try:
        return base64.b64encode(path.read_bytes()).decode("utf-8")
    except Exception:
        return None


def inject_global_css() -> None:
    """Suntik CSS global - panggil sekali per halaman (idempotent secara visual)."""
    st.markdown(
        """
        <style>
        /* ===== Hero banner ===== */
        .hero-card {
            background: linear-gradient(135deg, #0b2e6f 0%, #1f4fa6 55%, #2a7fb8 100%);
            padding: 30px 34px;
            border-radius: 18px;
            color: #ffffff;
            box-shadow: 0 8px 28px rgba(11, 46, 111, 0.25);
            margin-bottom: 18px;
        }
        .hero-title {
            font-size: 28px;
            font-weight: 800;
            line-height: 1.25;
            margin: 0 0 10px 0;
            letter-spacing: 0.2px;
        }
        .hero-title.small {
            font-size: 22px;
        }
        .hero-sub {
            font-size: 14px;
            opacity: 0.92;
            margin: 0;
        }
        .hero-tag {
            display: inline-block;
            padding: 4px 12px;
            border-radius: 999px;
            background: rgba(255, 255, 255, 0.15);
            font-size: 12px;
            font-weight: 600;
            letter-spacing: 1px;
            text-transform: uppercase;
            margin-bottom: 10px;
        }
        .hero-page {
            display: inline-block;
            padding: 3px 10px;
            border-radius: 999px;
            background: rgba(255, 255, 255, 0.18);
            font-size: 11px;
            font-weight: 700;
            letter-spacing: 0.6px;
            text-transform: uppercase;
            margin-left: 8px;
            vertical-align: middle;
        }

        /* ===== Logo row ===== */
        .logo-row {
            display: flex;
            align-items: center;
            gap: 22px;
            padding: 12px 18px;
            background: #ffffff;
            border: 1px solid #e7ecf3;
            border-radius: 14px;
            margin-bottom: 16px;
        }
        .logo-row img {
            height: 62px;
            object-fit: contain;
        }
        .logo-sep {
            width: 1px;
            height: 52px;
            background: #d9e1ec;
        }
        .logo-meta {
            font-size: 13px;
            color: #32445f;
            line-height: 1.45;
        }
        .logo-meta b { color: #0b2e6f; }

        /* ===== Author card ===== */
        .author-card {
            background: #f7fafd;
            border: 1px solid #dfe7f2;
            border-left: 5px solid #1f4fa6;
            border-radius: 12px;
            padding: 14px 20px;
            margin-bottom: 18px;
        }
        .author-name {
            font-size: 17px;
            font-weight: 700;
            color: #0b2e6f;
            margin: 0;
        }
        .author-meta {
            font-size: 13px;
            color: #45566f;
            margin: 2px 0 0 0;
        }

        /* ===== Feature cards ===== */
        .feature-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
            gap: 14px;
            margin: 6px 0 18px 0;
        }
        .feature-card {
            background: #ffffff;
            border: 1px solid #e7ecf3;
            border-radius: 12px;
            padding: 16px 18px;
            transition: transform 0.15s ease, box-shadow 0.15s ease;
        }
        .feature-card:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 18px rgba(11, 46, 111, 0.08);
        }
        .feature-icon { font-size: 22px; margin-bottom: 4px; }
        .feature-title {
            font-size: 15px;
            font-weight: 700;
            color: #0b2e6f;
            margin: 4px 0;
        }
        .feature-desc {
            font-size: 13px;
            color: #4a5c78;
            line-height: 1.5;
            margin: 0;
        }

        /* ===== Section title ===== */
        .section-title {
            font-size: 18px;
            font-weight: 700;
            color: #0b2e6f;
            border-left: 4px solid #1f4fa6;
            padding-left: 10px;
            margin: 18px 0 10px 0;
        }

        /* ===== Class chips ===== */
        .chip-row {
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
            margin: 6px 0 4px 0;
        }
        .chip {
            padding: 4px 12px;
            border-radius: 999px;
            font-size: 12px;
            font-weight: 600;
            background: #eaf1fb;
            color: #1f4fa6;
            border: 1px solid #d2dff2;
        }
        .chip.other { background: #f1f1f1; color: #666; border-color: #e0e0e0; }

        /* ===== Footer ===== */
        .footer-note {
            text-align: center;
            color: #7a8aa3;
            font-size: 12px;
            padding: 14px 8px 4px 8px;
        }

        /* ===== Metric card polish ===== */
        div[data-testid="stMetric"] {
            background: #ffffff;
            border: 1px solid #e7ecf3;
            border-radius: 12px;
            padding: 12px 14px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _render_logo_row() -> None:
    """Logo Universitas + School of Applied STEM dalam satu kartu."""
    b64_univ = _img_to_base64(_LOGO_UNIV)
    b64_stem = _img_to_base64(_LOGO_STEM)

    if b64_univ and b64_stem:
        st.markdown(
            f"""
            <div class="logo-row">
                <img src="data:image/png;base64,{b64_univ}" alt="Universitas Prasetiya Mulya"/>
                <div class="logo-sep"></div>
                <img src="data:image/png;base64,{b64_stem}" alt="School of Applied STEM"/>
                <div class="logo-sep"></div>
                <div class="logo-meta">
                    <b>Universitas Prasetiya Mulya</b><br/>
                    School of Applied STEM &middot; Artificial Intelligence and Robotics
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        # Fallback kalau file logo hilang.
        c1, c2 = st.columns([1, 4])
        with c1:
            if _LOGO_UNIV.exists():
                st.image(str(_LOGO_UNIV), width=110)
            if _LOGO_STEM.exists():
                st.image(str(_LOGO_STEM), width=150)
        with c2:
            st.markdown(
                f"**{AUTHOR_UNIV}**  \n"
                "School of Applied STEM &middot; Artificial Intelligence and Robotics"
            )


def _render_hero(
    page_label: Optional[str] = None,
    subtitle: Optional[str] = None,
    compact: bool = False,
) -> None:
    """Hero banner berisi judul skripsi + label halaman + subtitle opsional."""
    page_chip = (
        f"<span class='hero-page'>{page_label}</span>" if page_label else ""
    )
    sub = (
        f"<p class='hero-sub'>{subtitle}</p>" if subtitle else ""
    )
    title_class = "hero-title small" if compact else "hero-title"

    st.markdown(
        f"""
        <div class="hero-card">
            <div>
                <span class="hero-tag">{AUTHOR_TAG} &middot; Skripsi</span>
                {page_chip}
            </div>
            <div class="{title_class}">
                {THESIS_TITLE_HTML}
            </div>
            {sub}
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_author_card() -> None:
    """Kartu identitas penulis."""
    st.markdown(
        f"""
        <div class="author-card">
            <p class="author-name">{AUTHOR_NAME}</p>
            <p class="author-meta">
                {AUTHOR_PROGRAM} &middot;
                {AUTHOR_UNIV} &middot;
                <i>{AUTHOR_TAG}</i>
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_page_header(
    page_label: Optional[str] = None,
    subtitle: Optional[str] = None,
    compact: bool = True,
    show_author: bool = True,
) -> None:
    """Render lengkap: CSS + logo row + hero + author card.

    Panggil fungsi ini satu kali di bagian atas setiap halaman setelah
    `st.set_page_config`. Parameter:
    - `page_label`: teks kecil di kanan tag "Tugas Akhir" (mis. "Inference").
    - `subtitle`  : subjudul pendek di bawah judul skripsi.
    - `compact`   : True untuk halaman selain landing (judul lebih ringkas).
    - `show_author`: tampilkan kartu penulis (default True).
    """
    inject_global_css()
    _render_logo_row()
    _render_hero(page_label=page_label, subtitle=subtitle, compact=compact)
    if show_author:
        _render_author_card()


def render_footer() -> None:
    """Footer seragam dengan copyright & identitas penulis."""
    st.markdown(
        f"""
        <div class="footer-note">
            &copy; 2026 &middot; {AUTHOR_NAME} &middot;
            {AUTHOR_PROGRAM} &middot;
            {AUTHOR_UNIV}
        </div>
        """,
        unsafe_allow_html=True,
    )
