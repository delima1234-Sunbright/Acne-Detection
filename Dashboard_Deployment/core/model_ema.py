"""EMA + C2fWithEMA injection for YOLOv8 neck.

Disalin persis dari Cell 3 notebook training
(`Dari Server Kampus/Fixed Training Model.ipynb`).

Class EMA dan C2fWithEMA WAJIB tersedia saat loading `best.pt`,
karena checkpoint di-pickle dengan struktur model yang sudah berisi
modul C2fWithEMA. Pickle akan mencari class ini di namespace
`__main__` (karena training dilakukan di notebook), sehingga modul
ini juga mendaftarkan kedua class tersebut ke `sys.modules['__main__']`
melalui fungsi `register_for_unpickle()`.
"""
from __future__ import annotations

import sys
import torch
import torch.nn as nn

from ultralytics.nn.modules import C2f
from ultralytics.nn.modules.block import SPPF


# ═══════════════════════════════════════════════════════════════════════════════
# 1.  EFFICIENT MULTI-SCALE ATTENTION (EMA)
# ═══════════════════════════════════════════════════════════════════════════════

class EMA(nn.Module):
    """Efficient Multi-Scale Attention - plug-in attention module untuk neck YOLO."""

    def __init__(self, channels: int, factor: int = 8):
        super().__init__()
        self.groups = factor
        assert channels // self.groups > 0, "channels harus kelipatan factor"
        self.softmax = nn.Softmax(dim=-1)
        self.agp = nn.AdaptiveAvgPool2d((1, 1))
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        self.gn = nn.GroupNorm(channels // self.groups, channels // self.groups)
        self.conv1x1 = nn.Conv2d(channels // self.groups, channels // self.groups,
                                  kernel_size=1, bias=False)
        self.conv3x3 = nn.Conv2d(channels // self.groups, channels // self.groups,
                                  kernel_size=3, padding=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.size()
        group_x = x.reshape(b * self.groups, -1, h, w)
        x_h = self.pool_h(group_x)
        x_w = self.pool_w(group_x).permute(0, 1, 3, 2)
        hw = self.conv1x1(torch.cat([x_h, x_w], dim=2))
        x_h, x_w = torch.split(hw, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)
        a1 = torch.sigmoid(self.gn(x_h))
        a2 = torch.sigmoid(self.gn(x_w))
        group_x = group_x * a1.expand_as(group_x) * a2.expand_as(group_x)
        x1 = self.agp(group_x)
        x2 = self.conv3x3(group_x).reshape(b * self.groups, -1, 1)
        x11 = self.softmax(x1.reshape(b * self.groups, -1, 1))
        x12 = x1.reshape(b * self.groups, -1, 1)
        x21 = self.softmax(x2)
        x22 = x2
        weights = (torch.matmul(x11.transpose(1, 2), x12) +
                   torch.matmul(x21.transpose(1, 2), x22)).reshape(b * self.groups, 1, h, w)
        out = group_x * weights.sigmoid().expand_as(group_x)
        return out.reshape(b, c, h, w)


# ═══════════════════════════════════════════════════════════════════════════════
# 2.  C2f + EMA wrapper
# ═══════════════════════════════════════════════════════════════════════════════

class C2fWithEMA(nn.Module):
    """C2f + EMA attention wrapper untuk neck YOLOv8."""

    def __init__(self, c2f_module: C2f):
        super().__init__()
        self.c2f = c2f_module
        ch = c2f_module.cv2.conv.out_channels
        factor = 8
        while ch % factor != 0 and factor > 1:
            factor //= 2
        self.ema = EMA(channels=ch, factor=factor)

    def forward(self, x):
        return self.ema(self.c2f(x))


# ═══════════════════════════════════════════════════════════════════════════════
# 3.  Inject EMA ke semua C2f di neck (after SPPF)
# ═══════════════════════════════════════════════════════════════════════════════

def inject_ema_to_neck(model, verbose: bool = True):
    """Deteksi akhir backbone via SPPF layer, ganti semua C2f di neck dgn C2fWithEMA.

    Aman dipanggil pada model yang sudah ber-C2fWithEMA: isinstance check
    `C2f` mengembalikan False untuk modul C2fWithEMA, jadi fungsi ini
    menjadi no-op pada model yang sudah di-inject.
    """
    backbone_end = None
    for i, layer in enumerate(model.model.model):
        if isinstance(layer, SPPF):
            backbone_end = i
            break
    neck_start = (backbone_end if backbone_end is not None else 9) + 1

    injected = 0
    for i, layer in enumerate(model.model.model):
        if i >= neck_start and isinstance(layer, C2f):
            model.model.model[i] = C2fWithEMA(layer)
            injected += 1

    if verbose:
        print(
            f"EMA injection: {injected} C2f -> C2fWithEMA at neck "
            f"(backbone_end={backbone_end}, neck_start={neck_start})"
        )
    return model


# ═══════════════════════════════════════════════════════════════════════════════
# 4.  Pickle compatibility - daftarkan class ke __main__
# ═══════════════════════════════════════════════════════════════════════════════

def register_for_unpickle():
    """Mendaftarkan EMA dan C2fWithEMA ke namespace __main__.

    Wajib dipanggil SEBELUM `YOLO(best.pt)` karena checkpoint training
    dibuat di Jupyter notebook; pickle menyimpan class dengan path
    `__main__.EMA` dan `__main__.C2fWithEMA`. Tanpa ini, pickle akan
    gagal dengan AttributeError saat load.
    """
    main_mod = sys.modules.get("__main__")
    if main_mod is None:
        return
    if not hasattr(main_mod, "EMA"):
        main_mod.EMA = EMA
    if not hasattr(main_mod, "C2fWithEMA"):
        main_mod.C2fWithEMA = C2fWithEMA
    if not hasattr(main_mod, "inject_ema_to_neck"):
        main_mod.inject_ema_to_neck = inject_ema_to_neck


register_for_unpickle()
