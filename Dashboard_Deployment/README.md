# Acne Analyzer Dashboard (YOLOv8n)

Streamlit dashboard untuk inferensi model deteksi lesi kulit YOLOv8n hasil training progressive tuning, dengan dua sumber input (upload foto / capture langsung dari Raspberry Pi via SSH) dan pipeline **dynamic-padding + sliding-window 640x640 + skin-mask skip**.

## Fitur Utama

- **Dua Sumber Gambar**
  - Upload foto **ukuran bebas** (tidak lagi di-lock ke `4608 x 2592`).
  - Capture langsung dari Raspberry Pi (12MP, `rpicam-still`) via SSH + SCP.
  - **Live Preview lokal** dari kamera Pi di jendela OpenCV native (di luar Streamlit) untuk audit posisi sebelum capture.
- **Dynamic Padding (Tahap 1)**
  - Tinggi & lebar gambar input di-padding **gray-114** (match Ultralytics LetterBox training) secara simetris hingga menjadi **kelipatan 640** terdekat (minimal 640). Aspek rasio asli dipertahankan.
  - Contoh: `4608 x 2592` → `5120 x 3200` (+256 L/R, +304 T/B).
  - Contoh: `900 x 500` → `1280 x 640`.
  - **Kenapa gray 114, bukan putih?** Ultralytics LetterBox saat training memakai `padding_value=114`. Kalau kita pakai putih (255), tile di area tepi akan berisi banyak piksel putih murni yang tidak pernah dilihat model saat training -> bisa memicu false-positive atau missed-detection di tepi.
- **Dua Skenario Inference (Tahap 2)**
  - **Skenario A** — jika padded tepat `640 x 640`: inference langsung 1x.
  - **Skenario B** — jika lebih besar: sliding window `640 x 640` dengan **overlap** (default `128 px` / 20%). Sebelum inference, tiap tile divalidasi lewat **mask kulit wajah** (YCrCb + HSV). Tile tanpa area kulit (rambut / background murni) akan **di-skip** untuk menghemat waktu.
- **Post-Processing (Tahap 4)**
  - **Class-wise NMS** (`IoU=0.5`) untuk buang deteksi duplikat antar jendela overlap.
  - **Koreksi Koordinat**: bbox di-remap dari koordinat tile → koordinat gambar **ASLI** (sebelum padding).
  - **Visualisasi**: warna bbox unik per kelas (palet `CLASS_COLORS`), label format `Nama: 94.5%` (1 decimal).
  - **Overlay Opsional**: area non-kulit bisa di-gelapkan (dimmed) untuk menandai secara visual bahwa AI melewati area tersebut.
- **Health Score Card**: Total lesi, kelas dominan, lesi inflamasi, severity (Hijau/Kuning/Merah).
- **Visualisasi**: gambar asli vs annotated, toggle bbox, pie/bar chart, crop zoom interaktif.
- **Export**: PNG hasil deteksi + laporan PDF.
- **Halaman Model Performance**: confusion matrix normalized + metrik P/R/mAP per kelas.

## Struktur

```
Deployment/
  app.py                       # entry Streamlit
  config.py                    # TILE, OVERLAP_PX, USE_FACE_MASK, dll.
  requirements.txt
  pages/
    1_Inference.py
    2_Model_Performance.py
  core/
    sliding_window.py          # pad_to_multiple + iter_tiles_overlap + PadInfo + NMS
    face_mask.py               # skin mask (YCrCb+HSV) untuk tile skip & overlay dim
    inference.py               # YOLO load + run_full_inference (Skenario A/B)
    raspberry_pi.py            # SSH capture + MJPEG preview thread
    severity.py                # ringkasan klinis (dimensi dinamis)
    visualization.py           # draw_bboxes + overlay_dim_non_skin + plotly
    pdf_export.py              # fpdf2 report
  assets/
    class_info.py              # ciri khas tiap kelas + metrik training
  tmp/                         # auto-created untuk hasil capture
  dashboard.py                 # (legacy, referensi capture awal)
```

## Setup

```powershell
cd Deployment
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Pastikan path model di `config.py` benar:

```python
MODEL_PATH = r"C:\Dokumen\SKRIPSI\Bimbingan\Dari Server Kampus\HASIL FIX\runs_v5\progressive_yolov8n\weights\best.pt"
```

## Menjalankan

```powershell
cd Deployment
streamlit run app.py
```

Buka `http://localhost:8501`.

## Parameter Kunci (`config.py`)

| Parameter | Default | Keterangan |
|---|---|---|
| `TILE` | `640` | Ukuran jendela inference (wajib = model imgsz). |
| `OVERLAP_PX` | `128` | Overlap antar jendela (20% dari 640). 0 = non-overlap. |
| `PAD_COLOR` | `(114, 114, 114)` | Warna padding. **Gray 114 match Ultralytics LetterBox training.** Jangan diubah ke putih kecuali ada alasan kuat. |
| `USE_FACE_MASK` | `True` | Aktifkan skin-mask skip sebelum inference. |
| `MIN_SKIN_RATIO` | `0.01` | Rasio min piksel kulit dalam tile supaya tidak di-skip. |
| `DEFAULT_CONF` | `0.25` | Confidence threshold default. **Match notebook eval.** |
| `DEFAULT_IOU` | `0.45` | IoU threshold per-tile NMS. **Match notebook eval.** |
| `NMS_BETWEEN_TILES_IOU` | `0.5` | IoU untuk class-wise NMS antar-jendela. |
| `MAX_DET` | `300` | Max detections per tile (Ultralytics default). |

Semua parameter di atas juga bisa di-override langsung dari **sidebar** halaman Inference.

## Parity Matrix (Training Notebook <-> Deployment)

Semua mismatch yang sudah diperbaiki supaya hasil inferensi di deployment SAMA dengan hasil evaluasi di notebook training:

| Aspek | Training (`Fixed Training Model.ipynb`) | Deployment (Streamlit) | Status |
|---|---|---|---|
| Channel order sampai ke model | RGB (Ultralytics swap BGR→RGB) | **RGB→BGR manual sebelum predict** | Match |
| Image loader | OpenCV (BGR) + Ultralytics LetterBox | PIL (RGB) + pipeline kita + manual BGR swap | Match (efektif) |
| Ukuran inference | `imgsz=640` | `imgsz=640` (TILE) | Match |
| Letterbox padding | Gray 114 (default Ultralytics) | **Pad gray 114** untuk outer padding | Match |
| Confidence threshold | `0.25` (Cell eval) | `DEFAULT_CONF = 0.25` | Match |
| IoU threshold (NMS per-image) | `0.45` (Cell eval) | `DEFAULT_IOU = 0.45` | Match |
| `augment` (TTA) | `False` | `augment=False` eksplisit | Match |
| `max_det` | `300` | `max_det=300` eksplisit | Match |
| `agnostic_nms` | `False` | `agnostic_nms=False` eksplisit | Match |
| `half` (FP16) | `False` | `half=False` eksplisit | Match |
| Device | `device=0` (GPU) | Auto `0` kalau CUDA, else CPU | Match |
| Class names order | 9 kelas sesuai `CLASS_NAMES` | 9 kelas sesuai `config.CLASS_NAMES` | Match |
| Model arch | YOLOv8n + EMA injection di neck | EMA-aware loader via `model_ema.py` | Match |

Catatan: `iou=0.7` di training config hanya dipakai untuk NMS **internal loss computation**, bukan untuk inference. Yang dipakai saat `eval_model.predict(...)` adalah `iou=0.45`.

## Alur Kerja Final

1. **Input** → Gambar ukuran bebas (upload / Raspberry Pi).
2. **Padding Dinamis** → Putih solid sampai kelipatan 640×640.
3. **Skenario A** (padded == 640×640) → inference langsung.
   **Skenario B** (lebih besar) → sliding window + overlap + skip via skin mask.
4. **Gabung & NMS** → class-wise IoU 0.5.
5. **Remap** → koordinat bbox kembali ke gambar ASLI.
6. **Visualisasi** → bbox warna per kelas + label `Nama: 94.5%` (+ opsional overlay dim non-kulit).

## Catatan Raspberry Pi

- Pi harus terinstall `libcamera-vid` dan `rpicam-still`.
- Live preview menggunakan TCP MJPEG di port `8888`. Pastikan port terbuka di firewall Pi.
- Capture 12MP otomatis menghentikan live preview untuk membebaskan kamera.
- Default credentials di `config.py` (`PI_DEFAULT_*`) bisa diubah dari sidebar.

## Troubleshooting

- **Model lambat di-load**: model dicache via `@st.cache_resource`, hanya load 1x per session.
- **Tile inference lambat pada gambar besar**: aktifkan **skin-mask skip** di sidebar untuk melewati tile tanpa kulit. Untuk gambar 4608×2592 dengan overlap 128, total tile ≈ 80 - bisa turun signifikan setelah skip.
- **Objek di tepi tile terpotong / miss**: tambah `OVERLAP_PX` (misal 192 atau 256) dari sidebar. NMS akan menggabung ulang duplikatnya.
- **Live preview tidak muncul**: pastikan port 8888 di Pi tidak dipakai proses lain. Cek log `/tmp/preview.log` di Pi.
- **Kulit tidak terdeteksi / semua tile di-skip**: turunkan **Min rasio kulit dalam tile** di sidebar, atau matikan skin-mask skip.
