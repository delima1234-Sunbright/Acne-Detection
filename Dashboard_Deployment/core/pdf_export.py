"""PDF report export using fpdf2."""
from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import Dict, Any, List
import tempfile
import numpy as np
import cv2

try:
    from fpdf import FPDF
except ImportError:
    FPDF = None  # type: ignore


def _save_temp_image(img_rgb: np.ndarray, max_w: int = 1600) -> str:
    h, w = img_rgb.shape[:2]
    if w > max_w:
        ratio = max_w / w
        img_rgb = cv2.resize(img_rgb, (max_w, int(h * ratio)), interpolation=cv2.INTER_AREA)
    bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    cv2.imwrite(tmp.name, bgr)
    tmp.close()
    return tmp.name


def build_pdf_report(
    annotated_rgb: np.ndarray,
    summary: Dict[str, Any],
    rows: List[Dict[str, Any]],
) -> bytes:
    """Return PDF bytes containing the annotated image, summary, and table."""
    if FPDF is None:
        raise RuntimeError("fpdf2 belum terinstall. Jalankan: pip install fpdf2")

    pdf = FPDF(orientation="P", unit="mm", format="A4")
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, "Laporan Analisis Acne - YOLOv8n", ln=1, align="C")
    pdf.ln(2)

    pdf.set_font("Helvetica", "", 11)
    sev = summary["severity"]
    pdf.cell(0, 7, f"Total Lesi: {summary['total']}", ln=1)
    pdf.cell(0, 7, f"Kelas Dominan: {summary['dominant']}", ln=1)
    pdf.cell(0, 7, f"Severity: {sev['label']}", ln=1)
    pdf.cell(0, 7, f"Lesi Inflamasi: {summary['n_inflammatory']}", ln=1)
    pdf.cell(0, 7, f"Rata-rata Relative Size: {summary['avg_relative_size']:.3f}%", ln=1)
    if "image_hw" in summary:
        H, W = summary["image_hw"]
        pdf.cell(0, 7, f"Ukuran Gambar: {W} x {H} px", ln=1)
    pdf.ln(4)

    img_path = _save_temp_image(annotated_rgb)
    try:
        pdf.image(img_path, x=10, w=190)
    finally:
        try:
            Path(img_path).unlink()
        except Exception:
            pass

    pdf.ln(4)
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "Distribusi per Kelas", ln=1)
    pdf.set_font("Helvetica", "", 10)
    for k, v in sorted(summary["per_class_count"].items(), key=lambda x: -x[1]):
        pct = summary["per_class_pct"][k]
        pdf.cell(0, 6, f"  - {k}: {v} titik ({pct:.1f}%)", ln=1)

    if rows:
        pdf.add_page()
        pdf.set_font("Helvetica", "B", 12)
        pdf.cell(0, 8, "Detail Lesi", ln=1)
        pdf.set_font("Helvetica", "B", 9)
        col_w = [12, 38, 28, 22, 22, 30, 32]
        headers = ["ID", "Class", "Conf (%)", "W (px)", "H (px)", "Area (px)", "Rel Size (%)"]
        for h, w in zip(headers, col_w):
            pdf.cell(w, 7, h, border=1, align="C")
        pdf.ln()
        pdf.set_font("Helvetica", "", 9)
        for r in rows[:120]:
            pdf.cell(col_w[0], 6, str(r["ID"]), border=1, align="C")
            pdf.cell(col_w[1], 6, str(r["Class"])[:18], border=1)
            pdf.cell(col_w[2], 6, f"{r['Confidence (%)']}", border=1, align="R")
            pdf.cell(col_w[3], 6, str(r["Width (px)"]), border=1, align="R")
            pdf.cell(col_w[4], 6, str(r["Height (px)"]), border=1, align="R")
            pdf.cell(col_w[5], 6, str(r["Area (px)"]), border=1, align="R")
            pdf.cell(col_w[6], 6, f"{r['Relative Size (%)']}", border=1, align="R")
            pdf.ln()

    out = pdf.output(dest="S")
    if isinstance(out, str):
        return out.encode("latin-1")
    return bytes(out)
