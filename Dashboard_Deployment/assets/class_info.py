"""Educational info for each detected class."""

CLASS_INFO = {
    "comedo": {
        "label": "Comedo",
        "simple": "Si kecil awal mula jerawat.",
        "ciri": "Titik putih atau hitam (pori tersumbat).",
        "kategori": "Non-inflamasi",
    },
    "papule": {
        "label": "Papule",
        "simple": "Jerawat merah yang mulai meradang.",
        "ciri": "Benjolan merah kecil, belum ada 'mata' nanahnya.",
        "kategori": "Inflamasi ringan",
    },
    "pustule": {
        "label": "Pustule",
        "simple": "Jerawat yang sudah 'matang'.",
        "ciri": "Merah di bawah, putih/kuning di puncak (ada nanahnya).",
        "kategori": "Inflamasi sedang",
    },
    "nodule": {
        "label": "Nodule",
        "simple": "Jerawat batu yang nakal.",
        "ciri": "Besar, merah, keras, dan biasanya sakit kalau ditekan.",
        "kategori": "Inflamasi berat",
    },
    "atrophic_scar": {
        "label": "Atrophic Scar",
        "simple": "Bekas jerawat yang mencekung.",
        "ciri": "Lubang-lubang kecil atau bopeng di kulit.",
        "kategori": "Bekas luka",
    },
    "hypertrophic_scar": {
        "label": "Hypertrophic Scar",
        "simple": "Bekas jerawat yang menonjol.",
        "ciri": "Daging tumbuh atau keloid yang muncul setelah luka sembuh.",
        "kategori": "Bekas luka",
    },
    "nevus": {
        "label": "Nevus",
        "simple": "Nama keren untuk tahi lalat.",
        "ciri": "Bercak cokelat/hitam yang biasanya menetap.",
        "kategori": "Lesi pigmen",
    },
    "melasma": {
        "label": "Melasma",
        "simple": "Flek atau bercak gelap.",
        "ciri": "Area kulit yang warnanya lebih gelap, biasanya karena matahari.",
        "kategori": "Hiperpigmentasi",
    },
    "other": {
        "label": "Other",
        "simple": "Lesi lain di luar 8 kelas utama.",
        "ciri": "Bentuk tidak spesifik, dianggap kategori umum.",
        "kategori": "Lainnya",
    },
}

SARAN_PER_KELAS = {
    "comedo": "Bersihkan wajah 2x sehari dengan pembersih ringan, hindari produk komedogenik.",
    "papule": "Gunakan produk dengan salicylic acid atau benzoyl peroxide ringan, jangan dipencet.",
    "pustule": "Hindari memencet pustule agar tidak menjadi scar. Gunakan spot treatment.",
    "nodule": "Konsultasi dengan dokter kulit. Nodule sering memerlukan perawatan medis.",
    "atrophic_scar": "Pertimbangkan treatment seperti microneedling atau laser di klinik kulit.",
    "hypertrophic_scar": "Konsultasi dokter kulit untuk treatment silicone gel atau injeksi.",
    "nevus": "Pantau perubahan ukuran/warna. Konsultasi jika berubah cepat.",
    "melasma": "Selalu gunakan sunscreen SPF 30+ setiap hari, hindari paparan matahari langsung.",
    "other": "Konsultasi dokter kulit untuk diagnosis lebih akurat.",
}

MODEL_METRICS_PER_CLASS = {
    "atrophic_scar":     {"images": 868,  "instances": 2801, "P": 0.630, "R": 0.617, "mAP50": 0.627, "mAP50_95": 0.261},
    "comedo":            {"images": 1348, "instances": 2408, "P": 0.560, "R": 0.487, "mAP50": 0.507, "mAP50_95": 0.195},
    "hypertrophic_scar": {"images": 173,  "instances": 724,  "P": 0.829, "R": 0.845, "mAP50": 0.907, "mAP50_95": 0.754},
    "melasma":           {"images": 643,  "instances": 3431, "P": 0.810, "R": 0.810, "mAP50": 0.877, "mAP50_95": 0.657},
    "nevus":             {"images": 756,  "instances": 1541, "P": 0.807, "R": 0.938, "mAP50": 0.911, "mAP50_95": 0.682},
    "nodule":            {"images": 41,   "instances": 302,  "P": 0.910, "R": 0.987, "mAP50": 0.993, "mAP50_95": 0.940},
    "other":             {"images": 40,   "instances": 69,   "P": 0.695, "R": 0.290, "mAP50": 0.371, "mAP50_95": 0.166},
    "papule":            {"images": 1191, "instances": 2445, "P": 0.700, "R": 0.711, "mAP50": 0.778, "mAP50_95": 0.451},
    "pustule":           {"images": 422,  "instances": 1658, "P": 0.877, "R": 0.897, "mAP50": 0.957, "mAP50_95": 0.788},
}

MODEL_METRICS_OVERALL = {
    "images": 3595,
    "instances": 15379,
    "P": 0.757,
    "R": 0.731,
    "mAP50": 0.770,
    "mAP50_95": 0.544,
    "params": "3,007,403",
    "gflops": 8.1,
    "speed_ms": {"preprocess": 0.1, "inference": 2.5, "postprocess": 1.1},
}
