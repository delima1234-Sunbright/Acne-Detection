"""YOLOv8n inference engine - pipeline dynamic-padding + sliding-window.

Catatan mismatch penting yang diperbaiki:

1. **Channel order BGR** (CRITICAL).
   Ultralytics `model.predict(source=<np.ndarray>)` ASUMSI input adalah
   BGR (konvensi OpenCV). Internal `preprocess` melakukan `im[..., ::-1]`
   untuk BGR->RGB sebelum di-feed ke model.
   Sumber gambar kita di Streamlit dibuka lewat PIL
   (`Image.open(...).convert("RGB")`) -> array RGB. Kalau dikirim apa
   adanya, akan dibalik jadi BGR di dalam model -> channel R & B tertukar
   relatif terhadap training -> akurasi anjlok.
   Solusi: konversi RGB -> BGR SEBELUM `model.predict`.

2. **Parameter `iou` default = 0.45**.
   Notebook training final eval memakai `iou=0.45, conf=0.25`. Default
   lama (0.7) adalah NMS threshold saat training (lebih longgar),
   menyebabkan banyak duplikat bocor ke NMS antar-jendela.

3. **Padding color = gray 114**.
   Ultralytics LetterBox (training) memakai padding value 114 (abu-abu
   medium). Kita ikuti supaya "tepi dummy" yang dilihat model tidak
   berupa putih 255 yang tidak pernah dilihat saat training.

4. **Device eksplisit**.
   Kita pilih device secara proaktif (CUDA kalau ada) agar inference
   tidak nyasar ke CPU saat pemanggilan pertama.
"""
from __future__ import annotations

import time
from typing import Optional, Callable, Dict, Any, List

import numpy as np
import torch
import streamlit as st

from core.model_ema import (
    EMA, C2fWithEMA, inject_ema_to_neck, register_for_unpickle,
)

from ultralytics import YOLO

from config import (
    MODEL_PATH, TILE, OVERLAP_PX, USE_FACE_MASK, MIN_SKIN_RATIO,
    DEFAULT_CONF, DEFAULT_IOU, NMS_BETWEEN_TILES_IOU,
    CLASS_NAMES, MAX_DET,
)
from core.sliding_window import (
    pad_to_multiple, iter_tiles_overlap, count_tiles,
    remap_bbox_to_original, nms_per_class, is_box_valid, PadInfo,
)
from core.face_mask import build_skin_mask, tile_has_skin


def _auto_device() -> str | int:
    """Pilih device terbaik: CUDA:0 kalau tersedia, else CPU.

    Bentuk yang diterima Ultralytics: `0` (int) untuk GPU pertama,
    `"cpu"` (str) untuk CPU.
    """
    return 0 if torch.cuda.is_available() else "cpu"


def _has_c2f_with_ema(yolo_model: YOLO) -> bool:
    """Cek apakah model sudah memuat modul C2fWithEMA di neck."""
    try:
        for layer in yolo_model.model.model:
            if isinstance(layer, C2fWithEMA):
                return True
    except Exception:
        pass
    return False


@st.cache_resource(show_spinner="Memuat model YOLOv8n (EMA-aware loader)...")
def load_model(model_path: str = MODEL_PATH) -> YOLO:
    """Load YOLO checkpoint dengan EMA-aware compatibility (cached)."""
    register_for_unpickle()
    model = YOLO(model_path)
    if _has_c2f_with_ema(model):
        pass
    device = _auto_device()
    try:
        model.to(device)
    except Exception:
        pass
    return model


def _rgb_to_bgr(img: np.ndarray) -> np.ndarray:
    """Bolak-balikkan channel terakhir (RGB<->BGR) dan pastikan kontigu.

    CRITICAL untuk inference Ultralytics: predictor mengasumsikan numpy
    input = BGR lalu swap ke RGB secara internal. Maka kita WAJIB kirim
    BGR dari sisi kita.
    """
    if img.ndim != 3 or img.shape[2] != 3:
        return img
    return np.ascontiguousarray(img[:, :, ::-1])


def run_full_inference(
    img_rgb: np.ndarray,
    conf: float = DEFAULT_CONF,
    iou: float = DEFAULT_IOU,
    overlap_px: int = OVERLAP_PX,
    use_face_mask: bool = USE_FACE_MASK,
    min_skin_ratio: float = MIN_SKIN_RATIO,
    progress_cb: Optional[Callable[[int, int], None]] = None,
) -> Dict[str, Any]:
    """Jalankan sliding-window YOLO inference pada gambar RGB ukuran bebas.

    `img_rgb` HARUS dalam channel order RGB (hasil PIL). Konversi ke BGR
    dilakukan internal sebelum memanggil `model.predict`.

    Returns dict: boxes (xyxy di koordinat ASLI), scores, classes,
    class_names, tile_count, tiles_skipped, tiles_inferred,
    inference_time_ms, scenario ("A"/"B"), pad_info, skin_mask,
    original_hw, device_used.
    """
    model = load_model()
    device = _auto_device()

    padded, pad_info = pad_to_multiple(img_rgb, tile=TILE)

    skin_mask_padded = None
    skin_mask_original = None
    if use_face_mask:
        skin_mask_padded = build_skin_mask(padded)
        skin_mask_original = skin_mask_padded[
            pad_info.pad_top:pad_info.pad_top + pad_info.orig_h,
            pad_info.pad_left:pad_info.pad_left + pad_info.orig_w,
        ].copy()

    if padded.shape[0] == TILE and padded.shape[1] == TILE:
        scenario = "A"
        total_tiles = 1
    else:
        scenario = "B"
        total_tiles = count_tiles(pad_info, overlap=overlap_px)

    all_boxes: List[np.ndarray] = []
    all_scores: List[float] = []
    all_classes: List[int] = []

    t0 = time.perf_counter()
    processed = 0
    inferred = 0
    skipped = 0

    def _infer_one_tile(tile_img_rgb: np.ndarray, x0: int, y0: int) -> None:
        """Run inference pada satu tile 640x640 (RGB), lalu remap bbox
        ke koordinat gambar ORIGINAL."""
        nonlocal inferred
        tile_bgr = _rgb_to_bgr(tile_img_rgb)
        results = model.predict(
            source=tile_bgr,
            imgsz=TILE,
            conf=conf,
            iou=iou,
            device=device,
            augment=False,
            max_det=MAX_DET,
            agnostic_nms=False,
            half=False,
            verbose=False,
        )
        inferred += 1
        r = results[0]
        if r.boxes is None or len(r.boxes) == 0:
            return
        tile_xyxy = r.boxes.xyxy.cpu().numpy()
        tile_conf = r.boxes.conf.cpu().numpy()
        tile_cls = r.boxes.cls.cpu().numpy().astype(np.int32)
        mapped = remap_bbox_to_original(tile_xyxy, x0, y0, pad_info)
        for b, s, c in zip(mapped, tile_conf, tile_cls):
            if is_box_valid(b):
                all_boxes.append(b)
                all_scores.append(float(s))
                all_classes.append(int(c))

    if scenario == "A":
        _infer_one_tile(padded, 0, 0)
        processed = 1
        if progress_cb is not None:
            progress_cb(processed, total_tiles)
    else:
        for tile_img, x0, y0 in iter_tiles_overlap(padded, tile=TILE, overlap=overlap_px):
            do_inference = True
            if use_face_mask and skin_mask_padded is not None:
                has_skin, _ratio = tile_has_skin(
                    skin_mask_padded, x0, y0, TILE, min_ratio=min_skin_ratio,
                )
                do_inference = has_skin

            if do_inference:
                _infer_one_tile(tile_img, x0, y0)
            else:
                skipped += 1

            processed += 1
            if progress_cb is not None:
                progress_cb(processed, total_tiles)

    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    if len(all_boxes) == 0:
        return {
            "boxes": np.zeros((0, 4), dtype=np.float32),
            "scores": np.zeros((0,), dtype=np.float32),
            "classes": np.zeros((0,), dtype=np.int32),
            "class_names": [],
            "tile_count": total_tiles,
            "tiles_skipped": skipped,
            "tiles_inferred": inferred,
            "inference_time_ms": elapsed_ms,
            "scenario": scenario,
            "pad_info": pad_info,
            "skin_mask": skin_mask_original,
            "original_hw": (pad_info.orig_h, pad_info.orig_w),
            "device_used": str(device),
        }

    boxes = np.stack(all_boxes).astype(np.float32)
    scores = np.array(all_scores, dtype=np.float32)
    classes = np.array(all_classes, dtype=np.int32)

    keep = nms_per_class(boxes, scores, classes, iou_threshold=NMS_BETWEEN_TILES_IOU)
    boxes = boxes[keep]
    scores = scores[keep]
    classes = classes[keep]

    model_names = model.names if hasattr(model, "names") else None
    if isinstance(model_names, dict):
        names = [
            model_names.get(int(c), CLASS_NAMES[int(c)] if int(c) < len(CLASS_NAMES) else str(int(c)))
            for c in classes
        ]
    else:
        names = [
            CLASS_NAMES[int(c)] if int(c) < len(CLASS_NAMES) else str(int(c))
            for c in classes
        ]

    return {
        "boxes": boxes,
        "scores": scores,
        "classes": classes,
        "class_names": names,
        "tile_count": total_tiles,
        "tiles_skipped": skipped,
        "tiles_inferred": inferred,
        "inference_time_ms": elapsed_ms,
        "scenario": scenario,
        "pad_info": pad_info,
        "skin_mask": skin_mask_original,
        "original_hw": (pad_info.orig_h, pad_info.orig_w),
        "device_used": str(device),
    }
