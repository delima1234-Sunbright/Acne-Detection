"""Global configuration for the Acne Detection Dashboard.

Pipeline (dynamic padding + sliding window overlap):
- Input gambar ukuran bebas (RGB).
- Padding gray-114 (match Ultralytics LetterBox training) sampai tinggi
  & lebar jadi kelipatan `TILE` (640). Gray 114 dipilih karena itulah
  nilai yang dipakai saat training (`padding_value=114`), sehingga
  model melihat "tepi dummy" yang sama familiernya seperti saat training.
- Jika hasil padding persis `TILE x TILE` -> inference langsung (Scenario A).
- Jika lebih besar -> sliding window 640x640 dengan overlap `OVERLAP_PX`
  (Scenario B), dan sebelum inference di-validasi dulu lewat mask kulit wajah.
- Gabung semua deteksi -> class-wise NMS -> remap ke koordinat gambar ASLI
  (sebelum padding).
"""
from pathlib import Path

MODEL_PATH = r"C:\Dokumen\SKRIPSI\Bimbingan\Dari Server Kampus\HASIL FIX\runs_v5\progressive_yolov8n\weights\best.pt"
CONFUSION_MATRIX_PATH = r"C:\Dokumen\SKRIPSI\Bimbingan\Dari Server Kampus\HASIL FIX\runs_v5\progressive_yolov8n\confusion_matrix_normalized.png"

# ---------------------------------------------------------------
# Tile & padding (pipeline baru - dinamis)
# ---------------------------------------------------------------
TILE = 640
# Besar overlap antar jendela sliding (dalam piksel). 128 px = 20% dari 640.
OVERLAP_PX = 128
# Warna padding (RGB). Gray 114 = nilai Ultralytics LetterBox default
# yang dipakai saat training. Wajib sama supaya distribusi tepi gambar
# match antara training & inference. JANGAN ganti ke putih (255) kecuali
# memang ada alasan kuat: model tidak pernah melihat 255-white borders
# selama training -> bisa trigger false-positive di tile tepi.
PAD_COLOR = (114, 114, 114)

# Default hint untuk UI (bukan hard requirement lagi):
# resolusi kamera Raspberry Pi HQ 12MP.
RAW_W, RAW_H = 4608, 2592

# ---------------------------------------------------------------
# Face / skin mask (Tahap 2 - validasi sebelum inference)
# ---------------------------------------------------------------
# Aktifkan validasi mask kulit wajah sebelum run inference per-tile.
USE_FACE_MASK = True
# Minimum proporsi piksel kulit di dalam tile 640x640 supaya tile
# dianggap "ada wajah". 0.0 = skip HANYA tile yang benar-benar tidak
# punya piksel kulit sama sekali (rambut/latar belakang murni). Semakin
# besar nilai ini semakin agresif skip-nya. Dipakai di backend saja,
# tidak diekspos ke dashboard.
MIN_SKIN_RATIO = 0.0

# ---------------------------------------------------------------
# Inference / NMS
# ---------------------------------------------------------------
# MATCH training eval notebook: conf=0.25, iou=0.45.
# (Notebook Cell eval final memanggil `eval_model.predict(..., conf=0.25,
#  iou=0.45)`. Jadi ini adalah operating point yang sudah di-tune.)
DEFAULT_CONF = 0.25
DEFAULT_IOU = 0.45
# IoU untuk menggabung deteksi duplikat antar jendela (post-processing
# class-wise NMS). Lebih longgar dari per-tile IoU supaya overlap tiles
# yang pasti akan double-detect bisa digabung.
NMS_BETWEEN_TILES_IOU = 0.5
# Max detections per-tile (match Ultralytics default saat training).
MAX_DET = 300

# ---------------------------------------------------------------
# Classes
# ---------------------------------------------------------------
CLASS_NAMES = [
    "atrophic_scar",
    "comedo",
    "hypertrophic_scar",
    "melasma",
    "nevus",
    "nodule",
    "other",
    "papule",
    "pustule",
]

# Palet warna unik per kelas (RGB). Dipakai untuk bounding box & chart.
CLASS_COLORS = {
    "atrophic_scar":     (199, 125, 255),
    "comedo":            (255, 215,  64),
    "hypertrophic_scar": (147, 112, 219),
    "melasma":           (139,  90,  43),
    "nevus":             ( 80,  80,  80),
    "nodule":            (220,  20,  60),
    "other":             (128, 128, 128),
    "papule":            (255, 105, 180),
    "pustule":           (255, 140,   0),
}

INFLAMMATORY = {"papule", "pustule", "nodule"}

# ---------------------------------------------------------------
# Paths & Raspberry Pi
# ---------------------------------------------------------------
ASSETS_DIR = Path(__file__).parent / "assets"
TMP_DIR = Path(__file__).parent / "tmp"
TMP_DIR.mkdir(exist_ok=True)

PI_DEFAULT_HOST = "10.10.169.199"
PI_DEFAULT_USER = "delima1234"
PI_DEFAULT_PASS = "delima1234"
# Path file capture sementara DI Raspberry Pi.
# Pakai /tmp supaya:
#  - selalu writable oleh user mana pun (tidak tergantung home dir),
#  - otomatis dibersihkan oleh OS saat reboot,
#  - tidak ada masalah permission / SELinux / quota.
# JPG dipilih (bukan PNG) karena 5-10x lebih cepat di-encode &
# transfer-nya juga lebih ringan untuk 12 MP.
PI_REMOTE_CAPTURE_PATH = "/tmp/acne_capture.jpg"
PI_PREVIEW_PORT = 8888
PI_PREVIEW_W = 1280
PI_PREVIEW_H = 720

# ---------------------------------------------------------------
# Orientasi hasil capture Raspberry Pi
# ---------------------------------------------------------------
# Kamera Raspberry Pi di rig ini terpasang landscape, sementara pasien
# berbaring sehingga kepala muncul "miring" di hasil capture (crown di
# sisi kanan frame, dagu di sisi kiri). Supaya tampilan & inferensi
# bekerja pada pose wajah upright, hasil JPG di-rotate CCW 90 derajat
# begitu ter-download. Konvensi sudut mengikuti PIL.Image.rotate:
#   positif = berlawanan arah jarum jam (CCW), expand=True otomatis.
# Nilai yang didukung: 0, 90, 180, 270 (derajat CCW).
PI_CAPTURE_ROTATION_DEG = 90
