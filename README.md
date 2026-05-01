# Lampiran Skripsi — Folder LAMPIRAN

Dokumen ini menjelaskan isi folder **LAMPIRAN** yang dipakai sebagai lampiran skripsi : kode training, dataset, artefak hasil training, dan aplikasi inferensi berbasis Streamlit.

---

## Ringkasan isi folder


| Item                                                                              | Jenis                       | Keterangan singkat                                                                                 |
| --------------------------------------------------------------------------------- | --------------------------- | -------------------------------------------------------------------------------------------------- |
| `Baseline Model Yolov8n/`                                                         | Folder + notebook           | Baseline YOLOv8n pada **data original**                                                            |
| `Dashboard_Deployment/`                                                           | Folder (Python + Streamlit) | Inferensi + dashboard (kamera Raspberry Pi & upload lokal)                                         |
| `Dataset Clean/`                                                                  | Folder (dataset)            | Dataset **final** untuk melatih YOLOv8n                                                            |
| `Dataset Ori/`                                                                    | Folder (dataset)            | Dataset **original** dari Roboflow                                                                 |
| `Hasil Model Train Fix/`                                                          | Folder (artefak training)   | Metrik evaluasi, run pelatihan, dan trial Optuna (skenario **tanpa** data leakage)                 |
| `Hasil pada Data Leakage/`                                                        | Folder (artefak training)   | Metrik evaluasi, run pelatihan, dan trial Optuna (skenario **dengan** data leakage / perbandingan) |
| `Augmentasi Data Minoritas.ipynb`                                                 | Notebook                    | Augmentasi khusus kelas minoritas                                                                  |
| `EDA Data_Cleaning Data_WindowCroppingWholeBBOX_Rename Class_Clean Dataset.ipynb` | Notebook                    | EDA, pembersihan data, window cropping bbox, rename kelas, penyusunan dataset bersih               |
| `Fixed Training Model.ipynb`                                                      | Notebook                    | Pelatihan YOLOv8 yang telah dimodifikasi (model “fix”)                                             |
| `Hyperparameter Tuning Optuna.ipynb`                                              | Notebook                    | Pencarian hyperparameter optimal dengan Optuna                                                     |
| `Photobooth Design.f3d`                                                           | File desain                 | Desain CAD (Fusion 360 `.f3d`), terkait photobooth/perangkat keras bila dijelaskan di naskah       |


---

## Alur kerja penelitian (disarankan)

Urutan logis berdasarkan notebook dan dataset di folder ini:

1. `**Dataset Ori/`** — sumber awal dari Roboflow.
2. `**EDA Data_Cleaning Data_WindowCroppingWholeBBOX_Rename Class_Clean Dataset.ipynb`** — eksplorasi, pembersihan, cropping, standarisasi kelas → menghasilkan/`memetakan` ke `**Dataset Clean/`**.
3. `**Augmentasi Data Minoritas.ipynb`** — menyeimbangkan/menambah sampel kelas yang sedikit **sebelum** atau **sesuai** alur training (sesuai metodologi di skripsi).
4. `**Baseline Model Yolov8n/`** — pelatihan baseline pada data **original** untuk perbandingan.
5. `**Hyperparameter Tuning Optuna.ipynb`** — eksperimen Optuna; artefak trial bisa dipetakan ke folder `**Hasil Model Train Fix/`** dan `**Hasil pada Data Leakage/`**.
6. `**Fixed Training Model.ipynb`** — pelatihan final dengan konfigurasi YOLOv8 yang dimodifikasi (mis. progressive training, parameter terpilih dari tuning, dll.—sesuai isi notebook).
7. `**Dashboard_Deployment/**` — deployment inferensi dan visualisasi untuk kamera (Raspberry Pi) dan unggahan gambar dari komputer lokal.

---

## Penjelasan per bagian

### 1. Baseline Model Yolov8n

- **Tujuan:** Baseline **YOLOv8n** pada dataset **original** (bukan versi “clean” final), agar ada pembanding sebelum/ di luar pipeline pembersihan penuh.
- **Isi utama :**
  - `Train Model Yolov8n.ipynb` — skrip pelatihan baseline.
  - `runs/baseline_yolov8n/` — Output Ultralytics 

### 2. Dashboard_Deployment

- **Tujuan:** **Inferensi** lewat **Streamlit**: menampilkan dashboard, mengambil input dari **kamera yang terhubung ke Raspberry Pi**, dan dari **unggahan file lokal**.
- **Teknis:** Aplikasi Python terstruktur (`app.py`, `dashboard.py`, modul di `core/`, halaman di `pages/`).
- **Dokumentasi tambahan:** Lihat `Dashboard_Deployment/README.md` dan `requirements.txt` di dalam folder tersebut untuk dependensi dan cara menjalankan.

### 3. Dataset Clean

- **Tujuan:** Dataset **akhir** yang dipakai untuk melatih model YOLOv8n setelah pipeline EDA, cleaning, cropping, dan penyesuaian kelas (format umumnya mengikuti struktur YOLO / Roboflow export).

### 4. Dataset Ori

- **Tujuan:** Salinan dataset **asli** dari **Roboflow** sebelum pemrosesan lengkap; berguna sebagai referensi baseline dan dokumentasi sumber data.

### 5. Hasil Model Train Fix

- **Tujuan:** Menyimpan **hasil evaluasi** dan **direktori run** pelatihan untuk skenario model yang “diperbaiki” / final (tanpa skenario data leakage yang dibahas terpisah).
- **Isi tipikal:**
  - `progressive_yolov8n/` — ringkasan metrik per epoch (`results.csv`), konfigurasi run (`args.yaml`).
  - `optuna_trials_v5/` — trial-trial Optuna (`trial_XXX/` dengan `results.csv`, `args.yaml`), plus `best_params.json`.
  - Berkas konfigurasi pendukung seperti `best_params.json`, `data_runtime.yaml` (sesuai yang diekspor saat eksperimen).

### 6. Hasil pada Data Leakage

- **Tujuan:** Artefak serupa dengan **Hasil Model Train Fix**, tetapi untuk eksperimen di mana **data leakage** disengaja atau dianalisis (mis. split yang bermasalah), termasuk **hasil perbandingan** jika ada.
- **Isi tipikal:**
  - `progressive_yolov8n/`, `optuna_trials_data_leakage/`, `best_params.json`, `data_runtime.yaml`.
  - `comparison_results.json` — ringkatan hasil perbandingan (jika dihasilkan oleh pipeline skripsi).

---

## Notebook inti (kode utama training & praproses)


| Notebook                                                                            | Fungsi                                                                                                     |
| ----------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------- |
| **Augmentasi Data Minoritas.ipynb**                                                 | Augmentasi pada kelas dengan jumlah sampel kecil agar distribusi lebih seimbang.                           |
| **EDA Data_Cleaning Data_WindowCroppingWholeBBOX_Rename Class_Clean Dataset.ipynb** | EDA, cleaning, window cropping berbasis bbox utuh, penamaan ulang kelas, dan penyusunan **Dataset Clean**. |
| **Fixed Training Model.ipynb**                                                      | Implementasi pelatihan **YOLOv8** yang sudah **dimodifikasi** (bukan sekadar baseline default).            |
| **Hyperparameter Tuning Optuna.ipynb**                                              | **Optimasi hyperparameter** dengan **Optuna** untuk menemukan kombinasi parameter yang lebih optimal.      |


---

## Catatan untuk pembaca / penguji 

- Jalankan notebook dengan **kernel** dan **versi pustaka** yang sama (atau setidaknya kompatibel) dengan saat eksperimen dicatat di skripsi (Ultralytics, PyTorch, Optuna, OpenCV, dll.).
- Path absolut di notebook mungkin mengacu ke mesin asli penulis; sesuaikan ke lokasi folder `**LAMPIRAN`** di komputer Anda.
- Folder dataset (`Dataset Clean`, `Dataset Ori`) bisa berukuran besar; pastikan ruang disk dan waktu sinkronisasi memadai jika dipindahkan.

---

## Struktur daftar isi tingkat atas (referensi cepat)

```
LAMPIRAN/
├── README.md                          ← file ini
├── Augmentasi Data Minoritas.ipynb
├── EDA Data_Cleaning ... Dataset.ipynb
├── Fixed Training Model.ipynb
├── Hyperparameter Tuning Optuna.ipynb
├── Photobooth Design.f3d
├── Baseline Model Yolov8n/
├── Dashboard_Deployment/
├── Dataset Clean/
├── Dataset Ori/
├── Hasil Model Train Fix/
└── Hasil pada Data Leakage/
```

