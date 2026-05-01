"""Skin / face mask detection.

Dipakai di Tahap 2 untuk memutuskan apakah sebuah tile 640x640 perlu
di-inference (ada area kulit) atau bisa dilewati (latar belakang /
rambut murni).

Pendekatan: gabungan thresholding di ruang warna YCrCb dan HSV - klasik,
cepat, tanpa dependency tambahan (cukup numpy + opencv). Cukup akurat
untuk use case akne photography yang objek utamanya adalah kulit wajah.
"""
from __future__ import annotations

from typing import Tuple
import numpy as np
import cv2


def build_skin_mask(img_rgb: np.ndarray) -> np.ndarray:
    """Return a binary skin mask (uint8, 0 atau 255) dengan ukuran sama
    seperti `img_rgb`.

    Heuristik:
    - YCrCb: 77 <= Cr <= 180, 133 <= Cb <= 173 (range standar kulit).
    - HSV : 0 <= H <= 30 (kulit cenderung di rentang oranye/merah).
    - Intersect kedua mask -> morphological close + open untuk bersihkan
      lubang & noise kecil.
    - Area yang sangat terang (hampir putih murni, mis. padding yang
      sudah kita tambahkan) dikecualikan.

    Catatan: ini *skin detector*, bukan face segmentation. Cukup untuk
    memutuskan tile mana yang pantas di-inference.
    """
    if img_rgb.ndim != 3 or img_rgb.shape[2] != 3:
        raise ValueError(f"Expected HxWx3 RGB image, got {img_rgb.shape}")

    bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    ycrcb = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

    mask_ycrcb = cv2.inRange(ycrcb, (0, 133, 77), (255, 173, 127))
    mask_hsv = cv2.inRange(hsv, (0, 30, 60), (25, 255, 255))

    mask = cv2.bitwise_and(mask_ycrcb, mask_hsv)

    # Cleanup: tutup lubang kecil, buang bintik noise.
    # Catatan: padding gray-114 secara alami TIDAK masuk rentang skin
    # YCrCb (Cr=128 vs range 133-173) maupun HSV (S=0 vs threshold 30),
    # jadi tidak perlu special-case exclude padding.
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    return mask


def tile_has_skin(
    mask: np.ndarray,
    x0: int,
    y0: int,
    tile: int,
    min_ratio: float = 0.01,
) -> Tuple[bool, float]:
    """Cek apakah crop `mask[y0:y0+tile, x0:x0+tile]` memuat cukup
    piksel kulit untuk di-inference.

    Returns: (has_skin, ratio)
      - has_skin: True bila ratio >= min_ratio
      - ratio   : proporsi piksel kulit di dalam tile (0..1)
    """
    crop = mask[y0:y0 + tile, x0:x0 + tile]
    if crop.size == 0:
        return False, 0.0
    ratio = float(np.count_nonzero(crop)) / float(crop.size)
    return ratio >= min_ratio, ratio


def dim_non_skin_overlay(
    img_rgb: np.ndarray,
    mask: np.ndarray,
    dim_strength: float = 0.55,
) -> np.ndarray:
    """Hasilkan copy gambar dengan area NON-kulit di-gelapkan.

    dim_strength = 0 -> tidak berubah, 1 -> total hitam.
    Area kulit tetap utuh sehingga bounding box tetap jelas terlihat.
    """
    dim_strength = float(np.clip(dim_strength, 0.0, 1.0))
    canvas = img_rgb.astype(np.float32)
    dark = canvas * (1.0 - dim_strength)

    skin_bool = mask > 0
    if skin_bool.shape != img_rgb.shape[:2]:
        skin_bool = cv2.resize(
            mask, (img_rgb.shape[1], img_rgb.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        ) > 0

    out = np.where(skin_bool[..., None], canvas, dark)
    return out.clip(0, 255).astype(np.uint8)
