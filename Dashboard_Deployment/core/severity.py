"""Severity scoring & summary builder from detection results.

Sekarang dimensi gambar DIINFER dari dict detections (`original_hw`)
agar mendukung gambar ukuran bebas (bukan lagi hardcode 4608x2592).
"""
from __future__ import annotations

from typing import Dict, Any, List, Tuple
import numpy as np

from config import INFLAMMATORY, RAW_W, RAW_H


def _severity_from_inflammatory(n_inflam: int) -> Dict[str, str]:
    if n_inflam < 5:
        return {"label": "Ringan", "level": "green", "emoji": "🟢"}
    if n_inflam <= 15:
        return {"label": "Sedang", "level": "yellow", "emoji": "🟡"}
    return {"label": "Berat", "level": "red", "emoji": "🔴"}


def _get_hw(detections: Dict[str, Any]) -> Tuple[int, int]:
    """Ambil (H, W) gambar original dari detections, fallback ke default."""
    hw = detections.get("original_hw")
    if hw and len(hw) == 2:
        return int(hw[0]), int(hw[1])
    return RAW_H, RAW_W


def _compute_face_coverage_pct(
    boxes: np.ndarray,
    H: int,
    W: int,
    skin_mask: "np.ndarray | None" = None,
) -> float:
    """Hitung persentase area wajah yang tertutup oleh lesi terdeteksi.

    Denominator:
      - Kalau `skin_mask` tersedia (hasil `build_skin_mask` pada gambar
        original) -> pakai jumlah piksel kulit sebagai proxy luas wajah.
      - Kalau tidak, fallback ke total area gambar (W*H).

    Numerator:
      - Union area dari SEMUA bounding box lesi (bukan sekadar jumlah,
        supaya bbox yang overlap tidak di-double count) dihitung lewat
        mask boolean berukuran HxW.
      - Kalau skin_mask ada, union area diintersect ke skin_mask dulu
        supaya bbox yang bocor ke rambut/latar tidak ikut dihitung.

    Return persentase (0..100), di-clip supaya tidak lebih dari 100.
    """
    if len(boxes) == 0 or H <= 0 or W <= 0:
        return 0.0

    lesion_mask = np.zeros((H, W), dtype=bool)
    for b in boxes:
        x1 = int(max(0, round(float(b[0]))))
        y1 = int(max(0, round(float(b[1]))))
        x2 = int(min(W, round(float(b[2]))))
        y2 = int(min(H, round(float(b[3]))))
        if x2 > x1 and y2 > y1:
            lesion_mask[y1:y2, x1:x2] = True

    if skin_mask is not None and skin_mask.shape[:2] == (H, W):
        skin_bool = skin_mask > 0
        denom = int(np.count_nonzero(skin_bool))
        if denom <= 0:
            denom = int(H * W)
            numer = int(np.count_nonzero(lesion_mask))
        else:
            numer = int(np.count_nonzero(lesion_mask & skin_bool))
    else:
        denom = int(H * W)
        numer = int(np.count_nonzero(lesion_mask))

    if denom <= 0:
        return 0.0
    pct = 100.0 * numer / float(denom)
    return float(min(100.0, max(0.0, pct)))


def build_summary(detections: Dict[str, Any]) -> Dict[str, Any]:
    """Aggregate detection results into a clinical-style summary."""
    boxes = detections["boxes"]
    names = detections["class_names"]
    scores = detections["scores"]
    skin_mask = detections.get("skin_mask")

    H, W = _get_hw(detections)
    total = int(len(boxes))
    raw_area = float(W * H)

    per_class_count: Dict[str, int] = {}
    per_class_conf: Dict[str, List[float]] = {}
    per_class_area: Dict[str, List[float]] = {}

    rel_sizes: List[float] = []
    for box, name, score in zip(boxes, names, scores):
        per_class_count[name] = per_class_count.get(name, 0) + 1
        per_class_conf.setdefault(name, []).append(float(score))
        w = max(0.0, float(box[2] - box[0]))
        h = max(0.0, float(box[3] - box[1]))
        area = w * h
        rel = (area / raw_area) * 100.0 if raw_area > 0 else 0.0
        rel_sizes.append(rel)
        per_class_area.setdefault(name, []).append(rel)

    per_class_pct = {
        k: (v / total * 100.0) if total > 0 else 0.0
        for k, v in per_class_count.items()
    }

    dominant = max(per_class_count, key=per_class_count.get) if per_class_count else None

    n_inflam = sum(per_class_count.get(k, 0) for k in INFLAMMATORY)
    severity = _severity_from_inflammatory(n_inflam)

    avg_rel = float(np.mean(rel_sizes)) if rel_sizes else 0.0

    face_coverage_pct = _compute_face_coverage_pct(boxes, H, W, skin_mask=skin_mask)

    return {
        "total": total,
        "per_class_count": per_class_count,
        "per_class_pct": per_class_pct,
        "per_class_avg_conf": {k: float(np.mean(v)) for k, v in per_class_conf.items()},
        "per_class_avg_area": {k: float(np.mean(v)) for k, v in per_class_area.items()},
        "dominant": dominant,
        "n_inflammatory": n_inflam,
        "severity": severity,
        "avg_relative_size": avg_rel,
        "relative_sizes": rel_sizes,
        "image_hw": (H, W),
        "face_coverage_pct": face_coverage_pct,
    }


def build_detection_table(detections: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Build per-detection rows for st.dataframe."""
    boxes = detections["boxes"]
    names = detections["class_names"]
    scores = detections["scores"]

    H, W = _get_hw(detections)
    raw_area = float(W * H)
    rows = []
    for i, (box, name, score) in enumerate(zip(boxes, names, scores), start=1):
        w = max(0.0, float(box[2] - box[0]))
        h = max(0.0, float(box[3] - box[1]))
        area = w * h
        rel = (area / raw_area) * 100.0 if raw_area > 0 else 0.0
        rows.append({
            "ID": i,
            "Class": name,
            "Confidence (%)": round(float(score) * 100.0, 1),
            "Width (px)": int(round(w)),
            "Height (px)": int(round(h)),
            "Area (px)": int(round(area)),
            "Relative Size (%)": round(rel, 3),
            "x1": int(round(float(box[0]))),
            "y1": int(round(float(box[1]))),
            "x2": int(round(float(box[2]))),
            "y2": int(round(float(box[3]))),
        })
    return rows


def build_text_conclusion(summary: Dict[str, Any]) -> str:
    """Produce a markdown conclusion block."""
    if summary["total"] == 0:
        return "**Tidak ada lesi terdeteksi.** Kondisi kulit terlihat bersih pada gambar ini."

    lines = [
        "**KESIMPULAN ANALISIS**",
        "",
        f"- **Total Objek Terdeteksi:** {summary['total']} titik.",
        f"- **Kelas Dominan:** `{summary['dominant']}` "
        f"({summary['per_class_count'][summary['dominant']]} titik).",
        f"- **Lesi Inflamasi (papule + pustule + nodule):** {summary['n_inflammatory']} titik.",
        f"- **Cakupan Lesi pada Wajah:** "
        f"{summary.get('face_coverage_pct', 0.0):.2f}% dari area wajah terdeteksi.",
        f"- **Rata-rata Ukuran Lesi:** {summary['avg_relative_size']:.3f}% dari area gambar.",
        "",
        "**Detail Per Kelas:**",
    ]
    for k in sorted(summary["per_class_count"], key=lambda x: -summary["per_class_count"][x]):
        cnt = summary["per_class_count"][k]
        pct = summary["per_class_pct"][k]
        lines.append(f"  - `{k}`: {cnt} titik ({pct:.1f}%).")
    return "\n".join(lines)
