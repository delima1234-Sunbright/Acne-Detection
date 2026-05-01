"""Visualization helpers: bbox drawing, charts, crop zoom, mask overlay."""
from __future__ import annotations

from typing import Dict, Any, Optional, Tuple
import numpy as np
import cv2

from config import CLASS_COLORS


def draw_bboxes(
    img_rgb: np.ndarray,
    detections: Dict[str, Any],
    show_labels: bool = False,
    line_thickness: Optional[int] = None,
    font_scale: Optional[float] = None,
) -> np.ndarray:
    """Return a copy of img_rgb with bounding boxes drawn on it.

    - Warna bbox mengikuti `CLASS_COLORS` (tiap kelas warna berbeda).
    - Default: HANYA kotak berwarna, tanpa teks nama kelas / confidence.
      Identifikasi kelas dilakukan lewat legenda warna terpisah
      (`build_legend_html`).
    - `show_labels=True` (opsional) akan menampilkan label
      `Nama: 94.5%` di atas kotak - dipakai hanya kalau butuh,
      misal untuk debugging.
    - `line_thickness` & `font_scale` otomatis proporsional terhadap
      ukuran gambar kalau tidak di-set.
    """
    canvas = img_rgb.copy()
    boxes = detections["boxes"]
    names = detections["class_names"]
    scores = detections["scores"]

    H, W = img_rgb.shape[:2]
    # Auto-scale: ~0.002 dari max(H, W), dibatasi minimal 2 dan maksimal 8.
    if line_thickness is None:
        line_thickness = int(np.clip(max(H, W) * 0.002, 2, 8))
    if font_scale is None:
        font_scale = float(np.clip(max(H, W) / 1500.0, 0.5, 1.8))

    for box, name, score in zip(boxes, names, scores):
        x1, y1, x2, y2 = [int(round(v)) for v in box]
        color_rgb = CLASS_COLORS.get(name, (0, 255, 0))
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color_rgb, line_thickness)
        if show_labels:
            label = f"{name}: {score * 100:.1f}%"
            (tw, th), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, max(1, line_thickness // 2)
            )
            ytop = max(0, y1 - th - baseline - 4)
            cv2.rectangle(
                canvas,
                (x1, ytop),
                (x1 + tw + 6, ytop + th + baseline + 4),
                color_rgb,
                thickness=-1,
            )
            cv2.putText(
                canvas,
                label,
                (x1 + 3, ytop + th + 1),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (255, 255, 255),
                max(1, line_thickness // 2),
                cv2.LINE_AA,
            )
    return canvas


def build_legend_html(
    per_class_count: Dict[str, int],
    include_zero: bool = False,
) -> str:
    """Bangun tabel legenda warna -> nama kelas -> jumlah deteksi.

    Rendernya HTML table yang siap dipakai di `st.markdown(..., unsafe_allow_html=True)`.

    - `per_class_count`: hasil `summary["per_class_count"]` (dict kelas -> count).
    - `include_zero`: kalau True, ikut tampilkan kelas yang tidak terdeteksi
      (count = 0) dengan warna lebih pucat. Default False supaya legenda ringkas.
    """
    if include_zero:
        # Pastikan semua kelas yang punya warna muncul, walau count-nya 0.
        items = [
            (name, int(per_class_count.get(name, 0)))
            for name in CLASS_COLORS.keys()
        ]
    else:
        items = [
            (name, int(count))
            for name, count in per_class_count.items()
            if int(count) > 0
        ]

    items.sort(key=lambda x: (-x[1], x[0]))

    if not items:
        return (
            "<div style='color:#7a8aa3;font-size:13px;padding:8px 2px;'>"
            "Tidak ada lesi terdeteksi pada gambar ini."
            "</div>"
        )

    rows_html = []
    for name, count in items:
        r, g, b = CLASS_COLORS.get(name, (128, 128, 128))
        muted = (count == 0)
        swatch_style = (
            f"display:inline-block;width:22px;height:14px;"
            f"background:rgb({r},{g},{b});border-radius:3px;"
            f"border:1px solid rgba(0,0,0,0.15);"
            f"{'opacity:0.35;' if muted else ''}"
        )
        name_style = (
            "font-family:'SF Mono','Menlo',monospace;font-size:13px;"
            f"color:{'#9aa4b2' if muted else '#1f2a44'};"
        )
        count_style = (
            "font-size:13px;font-weight:700;"
            f"color:{'#9aa4b2' if muted else '#0b2e6f'};"
            "text-align:right;"
        )
        rows_html.append(
            "<tr>"
            f"<td style='padding:6px 10px;width:34px;'><span style='{swatch_style}'></span></td>"
            f"<td style='padding:6px 10px;{name_style}'>{name}</td>"
            f"<td style='padding:6px 10px;{count_style}'>{count}</td>"
            "</tr>"
        )

    table = (
        "<div style='border:1px solid #e7ecf3;border-radius:10px;overflow:hidden;"
        "background:#ffffff;'>"
        "<table style='width:100%;border-collapse:collapse;'>"
        "<thead>"
        "<tr style='background:#f4f7fb;color:#32445f;font-size:12px;"
        "text-transform:uppercase;letter-spacing:0.4px;'>"
        "<th style='padding:8px 10px;text-align:left;'>Warna</th>"
        "<th style='padding:8px 10px;text-align:left;'>Kelas</th>"
        "<th style='padding:8px 10px;text-align:right;'>Jumlah</th>"
        "</tr>"
        "</thead>"
        "<tbody>"
        + "".join(rows_html)
        + "</tbody></table></div>"
    )
    return table


def overlay_dim_non_skin(
    img_rgb: np.ndarray,
    skin_mask: Optional[np.ndarray],
    dim_strength: float = 0.55,
) -> np.ndarray:
    """Gelapkan area non-kulit untuk menandai secara visual bahwa area
    tersebut dilewati oleh AI (opsional)."""
    if skin_mask is None:
        return img_rgb.copy()
    if skin_mask.shape[:2] != img_rgb.shape[:2]:
        skin_mask = cv2.resize(
            skin_mask, (img_rgb.shape[1], img_rgb.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )
    dim_strength = float(np.clip(dim_strength, 0.0, 1.0))
    canvas = img_rgb.astype(np.float32)
    dark = canvas * (1.0 - dim_strength)
    skin_bool = skin_mask > 0
    out = np.where(skin_bool[..., None], canvas, dark)
    return out.clip(0, 255).astype(np.uint8)


def crop_bbox(
    img_rgb: np.ndarray,
    box_xyxy: Tuple[int, int, int, int],
    pad_ratio: float = 0.3,
) -> np.ndarray:
    """Crop around a bbox with relative padding for nicer zoom view.

    Menggunakan ukuran gambar ASLI (`img_rgb.shape`) sebagai batas clip,
    bukan konstanta global, sehingga ikut support ukuran bebas.
    """
    H, W = img_rgb.shape[:2]
    x1, y1, x2, y2 = [int(round(v)) for v in box_xyxy]
    w = x2 - x1
    h = y2 - y1
    pad_x = int(w * pad_ratio)
    pad_y = int(h * pad_ratio)
    cx1 = max(0, x1 - pad_x)
    cy1 = max(0, y1 - pad_y)
    cx2 = min(W, x2 + pad_x)
    cy2 = min(H, y2 + pad_y)
    return img_rgb[cy1:cy2, cx1:cx2].copy()


def make_pie_chart(per_class_count: Dict[str, int]):
    """Create a Plotly pie chart with consistent class colors."""
    import plotly.graph_objects as go

    labels = list(per_class_count.keys())
    values = list(per_class_count.values())
    colors = [
        f"rgb{CLASS_COLORS.get(name, (128, 128, 128))}"
        for name in labels
    ]
    fig = go.Figure(
        data=[
            go.Pie(
                labels=labels,
                values=values,
                marker=dict(colors=colors, line=dict(color="#ffffff", width=2)),
                hole=0.45,
                textinfo="label+percent",
            )
        ]
    )
    fig.update_layout(
        title="Distribusi Kelas Lesi",
        showlegend=True,
        height=420,
        margin=dict(l=10, r=10, t=50, b=10),
    )
    return fig


def make_bar_chart(per_class_count: Dict[str, int]):
    """Create a Plotly bar chart of per-class counts."""
    import plotly.graph_objects as go

    items = sorted(per_class_count.items(), key=lambda x: -x[1])
    labels = [k for k, _ in items]
    values = [v for _, v in items]
    colors = [
        f"rgb{CLASS_COLORS.get(name, (128, 128, 128))}"
        for name in labels
    ]
    fig = go.Figure(
        data=[
            go.Bar(
                x=labels,
                y=values,
                marker_color=colors,
                text=values,
                textposition="outside",
            )
        ]
    )
    fig.update_layout(
        title="Jumlah Lesi per Kelas",
        xaxis_title="Kelas",
        yaxis_title="Jumlah",
        height=420,
        margin=dict(l=10, r=10, t=50, b=10),
    )
    return fig


def make_metric_bar(metrics_per_class: Dict[str, Dict[str, float]], metric_key: str = "mAP50"):
    """Bar chart of a per-class metric (e.g. mAP50) for the Performance page."""
    import plotly.graph_objects as go

    items = sorted(metrics_per_class.items(), key=lambda x: -x[1][metric_key])
    labels = [k for k, _ in items]
    values = [v[metric_key] for _, v in items]
    colors = []
    for v in values:
        if v >= 0.85:
            colors.append("rgb(46, 204, 113)")
        elif v >= 0.65:
            colors.append("rgb(241, 196, 15)")
        else:
            colors.append("rgb(231, 76, 60)")
    fig = go.Figure(
        data=[
            go.Bar(
                x=labels,
                y=values,
                marker_color=colors,
                text=[f"{v:.3f}" for v in values],
                textposition="outside",
            )
        ]
    )
    fig.update_layout(
        title=f"{metric_key} per Kelas",
        xaxis_title="Kelas",
        yaxis_title=metric_key,
        yaxis=dict(range=[0, 1.05]),
        height=460,
        margin=dict(l=10, r=10, t=50, b=10),
    )
    return fig
