"""Sliding window utilities - dynamic padding + overlap.

Perubahan dari versi lama:
- Padding TIDAK lagi di-hardcode untuk 4608x2592. Sekarang gambar ukuran
  BEBAS -> padding putih simetris sampai tinggi & lebar menjadi kelipatan
  TILE (640). (Tahap 1)
- Sliding window bisa pakai OVERLAP (default 128 px = 20%). Ini penting
  supaya objek yang terpotong di tepi jendela A tetap terdeteksi utuh
  di jendela B. (Tahap 2 - Skenario B)
- Hasil bbox bisa di-remap ke koordinat gambar ORIGINAL (sebelum padding)
  lewat `PadInfo`.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Generator, Tuple, List
import math
import numpy as np
import cv2

from config import TILE, OVERLAP_PX, PAD_COLOR


# -----------------------------------------------------------------
# Dynamic padding
# -----------------------------------------------------------------
@dataclass
class PadInfo:
    """Menyimpan info padding agar bisa reverse-remap bbox ke koordinat
    gambar ORIGINAL (sebelum padding)."""
    orig_h: int
    orig_w: int
    padded_h: int
    padded_w: int
    pad_top: int
    pad_bottom: int
    pad_left: int
    pad_right: int
    tile: int

    @property
    def n_tiles_x(self) -> int:
        return self.padded_w // self.tile

    @property
    def n_tiles_y(self) -> int:
        return self.padded_h // self.tile


def _ceil_to_multiple(value: int, multiple: int) -> int:
    """Bulatkan `value` ke atas ke kelipatan `multiple` terdekat.
    Minimal sama dengan `multiple` (misal 300 -> 640, 900 -> 1280)."""
    if value <= multiple:
        return multiple
    return int(math.ceil(value / multiple) * multiple)


def pad_to_multiple(
    img: np.ndarray,
    tile: int = TILE,
    pad_color: Tuple[int, int, int] = PAD_COLOR,
) -> Tuple[np.ndarray, PadInfo]:
    """Pad gambar simetris ke ukuran kelipatan `tile` terdekat (>= tile).

    Bias sisa piksel ke kanan & bawah (selisih 1 px hanya terjadi kalau
    total padding ganjil). Ini sesuai permintaan: "putih di sisi kanan
    dan bawah (atau proporsional kiri/kanan/atas/bawah)".

    Returns: (padded_img, PadInfo)
    """
    if img.ndim != 3 or img.shape[2] != 3:
        raise ValueError(f"Expected HxWx3 image, got shape {img.shape}")

    h, w = img.shape[:2]
    target_h = _ceil_to_multiple(h, tile)
    target_w = _ceil_to_multiple(w, tile)

    pad_h = target_h - h
    pad_w = target_w - w
    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left

    if pad_top == 0 and pad_bottom == 0 and pad_left == 0 and pad_right == 0:
        padded = img.copy()
    else:
        padded = cv2.copyMakeBorder(
            img,
            top=pad_top,
            bottom=pad_bottom,
            left=pad_left,
            right=pad_right,
            borderType=cv2.BORDER_CONSTANT,
            value=pad_color,
        )

    info = PadInfo(
        orig_h=h,
        orig_w=w,
        padded_h=target_h,
        padded_w=target_w,
        pad_top=pad_top,
        pad_bottom=pad_bottom,
        pad_left=pad_left,
        pad_right=pad_right,
        tile=tile,
    )
    return padded, info


# -----------------------------------------------------------------
# Sliding window iterator (with overlap)
# -----------------------------------------------------------------
def _compute_starts(length: int, tile: int, stride: int) -> List[int]:
    """Hitung daftar koordinat awal jendela supaya full-cover [0, length).

    Jendela berukuran `tile`, bergeser sejauh `stride`. Selalu
    tambahkan posisi terakhir (`length - tile`) jika belum tercakup
    agar sisi kanan / bawah tidak ada zona yang terlewat.
    """
    if length <= tile:
        return [0]
    starts: List[int] = []
    s = 0
    last = length - tile
    while s < last:
        starts.append(s)
        s += stride
    if not starts or starts[-1] != last:
        starts.append(last)
    return starts


def iter_tiles_overlap(
    padded: np.ndarray,
    tile: int = TILE,
    overlap: int = OVERLAP_PX,
) -> Generator[Tuple[np.ndarray, int, int], None, None]:
    """Yield (tile_img, x0, y0) untuk setiap jendela `tile x tile`
    dengan overlap `overlap` piksel.

    stride = tile - overlap. Contoh: tile=640, overlap=128 -> stride=512.
    """
    h, w = padded.shape[:2]
    stride = max(1, tile - overlap)

    ys = _compute_starts(h, tile, stride)
    xs = _compute_starts(w, tile, stride)

    for y0 in ys:
        for x0 in xs:
            crop = padded[y0:y0 + tile, x0:x0 + tile]
            # Safety: kalau ukurannya kurang (edge case gambar <= tile),
            # pad lagi dengan putih.
            if crop.shape[0] != tile or crop.shape[1] != tile:
                crop = cv2.copyMakeBorder(
                    crop,
                    top=0, bottom=max(0, tile - crop.shape[0]),
                    left=0, right=max(0, tile - crop.shape[1]),
                    borderType=cv2.BORDER_CONSTANT,
                    value=PAD_COLOR,
                )
            yield crop.copy(), int(x0), int(y0)


def count_tiles(pad_info: PadInfo, overlap: int = OVERLAP_PX) -> int:
    """Hitung jumlah tile yang akan dihasilkan `iter_tiles_overlap`."""
    tile = pad_info.tile
    stride = max(1, tile - overlap)
    ys = _compute_starts(pad_info.padded_h, tile, stride)
    xs = _compute_starts(pad_info.padded_w, tile, stride)
    return len(ys) * len(xs)


# -----------------------------------------------------------------
# Bbox remapping
# -----------------------------------------------------------------
def remap_bbox_to_original(
    box_xyxy: np.ndarray,
    tile_x0: int,
    tile_y0: int,
    pad_info: PadInfo,
) -> np.ndarray:
    """Translate bbox dari koordinat tile-local -> koordinat gambar
    ORIGINAL (sebelum padding).

    box_xyxy: shape (4,) atau (N, 4) pada range [0, tile].
    Output di-clip ke [0, orig_w] / [0, orig_h].
    """
    box = np.asarray(box_xyxy, dtype=np.float32).copy()
    if box.ndim == 1:
        box = box[None, :]
    box[:, [0, 2]] += tile_x0 - pad_info.pad_left
    box[:, [1, 3]] += tile_y0 - pad_info.pad_top
    box[:, [0, 2]] = np.clip(box[:, [0, 2]], 0, pad_info.orig_w)
    box[:, [1, 3]] = np.clip(box[:, [1, 3]], 0, pad_info.orig_h)
    return box


def is_box_valid(box: np.ndarray, min_size: float = 2.0) -> bool:
    """Cek bbox masih punya area > min_size pixel di kedua dimensi."""
    w = box[2] - box[0]
    h = box[3] - box[1]
    return bool(w > min_size and h > min_size)


# -----------------------------------------------------------------
# NMS (class-wise)
# -----------------------------------------------------------------
def nms_per_class(
    boxes: np.ndarray,
    scores: np.ndarray,
    classes: np.ndarray,
    iou_threshold: float = 0.5,
) -> np.ndarray:
    """Class-wise NMS untuk buang duplikat antar jendela yang overlap.
    Return: indices yang disimpan."""
    if len(boxes) == 0:
        return np.array([], dtype=np.int64)

    keep_indices: List[int] = []
    for cls in np.unique(classes):
        cls_mask = classes == cls
        cls_idx = np.where(cls_mask)[0]
        cls_boxes = boxes[cls_idx]
        cls_scores = scores[cls_idx]
        kept = _nms_numpy(cls_boxes, cls_scores, iou_threshold)
        keep_indices.extend(cls_idx[kept].tolist())

    return np.array(sorted(keep_indices), dtype=np.int64)


def _nms_numpy(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> np.ndarray:
    """Standard greedy NMS, returns local indices to keep."""
    if len(boxes) == 0:
        return np.array([], dtype=np.int64)
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    areas = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(int(i))
        if order.size == 1:
            break
        rest = order[1:]
        xx1 = np.maximum(x1[i], x1[rest])
        yy1 = np.maximum(y1[i], y1[rest])
        xx2 = np.minimum(x2[i], x2[rest])
        yy2 = np.minimum(y2[i], y2[rest])
        inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        union = areas[i] + areas[rest] - inter + 1e-9
        iou = inter / union
        order = rest[iou <= iou_threshold]
    return np.array(keep, dtype=np.int64)
