"""Raspberry Pi camera control: high-res capture (SCP) and live preview (cv2)."""
from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Optional, Tuple

import cv2
import paramiko
from PIL import Image

from config import (
    PI_REMOTE_CAPTURE_PATH, PI_PREVIEW_PORT,
    PI_PREVIEW_W, PI_PREVIEW_H, TMP_DIR,
    PI_CAPTURE_ROTATION_DEG,
)


def _auto_rotate_capture(local_path: Path) -> None:
    """Rotate captured JPG in-place sesuai orientasi rig kamera.

    Kamera Pi di rig ini memasang sensor landscape tetapi pasien
    berbaring sehingga crown kepala muncul di sisi kanan frame. Agar
    tampilan & inferensi konsisten (wajah upright), hasil capture
    di-rotate CCW sesuai `PI_CAPTURE_ROTATION_DEG`.

    Mengikuti konvensi PIL: positif = berlawanan arah jarum jam
    (CCW), `expand=True` supaya kanvas mengikuti hasil rotasi
    (tidak terpotong). File di-overwrite dengan kualitas JPG 95 supaya
    loss kompresi minimal sebelum masuk ke pipeline inferensi.
    """
    deg = int(PI_CAPTURE_ROTATION_DEG) % 360
    if deg == 0:
        return
    try:
        with Image.open(local_path) as im:
            im = im.convert("RGB")
            rotated = im.rotate(deg, expand=True)
            rotated.save(local_path, format="JPEG", quality=95, optimize=True)
    except Exception:
        # Rotasi best-effort - jangan blokir alur utama kalau gagal.
        pass


class PiPreviewThread(threading.Thread):
    """Background thread that opens a local cv2 window streaming MJPEG from the Pi."""

    def __init__(self, host: str, port: int = PI_PREVIEW_PORT,
                 window_title: str = "Raspberry Pi Live Preview - Tekan Q untuk close"):
        super().__init__(daemon=True)
        self.host = host
        self.port = port
        self.window_title = window_title
        self._stop_event = threading.Event()
        self.error: Optional[str] = None

    def stop(self) -> None:
        self._stop_event.set()

    def run(self) -> None:
        url = f"tcp://{self.host}:{self.port}"
        cap = None
        try:
            for _ in range(30):
                if self._stop_event.is_set():
                    return
                cap = cv2.VideoCapture(url)
                if cap.isOpened():
                    break
                time.sleep(0.5)

            if cap is None or not cap.isOpened():
                self.error = f"Tidak dapat membuka stream dari {url}"
                return

            cv2.namedWindow(self.window_title, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(self.window_title, 960, 540)

            while not self._stop_event.is_set():
                ok, frame = cap.read()
                if not ok or frame is None:
                    time.sleep(0.05)
                    continue
                cv2.imshow(self.window_title, frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q") or key == 27:
                    self._stop_event.set()
                    break
        except Exception as e:
            self.error = str(e)
        finally:
            if cap is not None:
                cap.release()
            try:
                cv2.destroyWindow(self.window_title)
            except Exception:
                pass


_active_preview: Optional[PiPreviewThread] = None
_active_ssh: Optional[paramiko.SSHClient] = None


def _ssh_connect(host: str, user: str, password: str) -> paramiko.SSHClient:
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(hostname=host, username=user, password=password, timeout=10)
    return client


def start_live_preview(host: str, user: str, password: str) -> Tuple[bool, str]:
    """Start libcamera-vid MJPEG TCP server on Pi + local cv2 preview window.

    Returns (ok, message).
    """
    global _active_preview, _active_ssh

    if _active_preview is not None and _active_preview.is_alive():
        return True, "Preview sudah aktif."

    try:
        client = _ssh_connect(host, user, password)
        client.exec_command("sudo pkill -f libcamera-vid; sudo pkill -f rpicam-vid")
        time.sleep(1.0)

        cmd = (
            f"nohup libcamera-vid -t 0 --inline --nopreview "
            f"--width {PI_PREVIEW_W} --height {PI_PREVIEW_H} "
            f"--codec mjpeg --framerate 15 "
            f"-o tcp://0.0.0.0:{PI_PREVIEW_PORT} --listen "
            f"> /tmp/preview.log 2>&1 &"
        )
        client.exec_command(cmd)
        time.sleep(2.0)

        _active_ssh = client
        _active_preview = PiPreviewThread(host=host)
        _active_preview.start()

        time.sleep(0.5)
        if _active_preview.error:
            return False, f"Preview thread error: {_active_preview.error}"

        return True, f"Live preview aktif di jendela OpenCV. Stream: tcp://{host}:{PI_PREVIEW_PORT}"
    except Exception as e:
        return False, f"Gagal start preview: {e}"


def stop_live_preview() -> Tuple[bool, str]:
    """Stop the local cv2 window and kill libcamera-vid on the Pi."""
    global _active_preview, _active_ssh

    msg = []
    if _active_preview is not None:
        _active_preview.stop()
        _active_preview.join(timeout=3)
        _active_preview = None
        msg.append("Jendela preview ditutup.")

    if _active_ssh is not None:
        try:
            _active_ssh.exec_command("sudo pkill -f libcamera-vid; sudo pkill -f rpicam-vid")
            time.sleep(0.5)
            _active_ssh.close()
            msg.append("Proses kamera Pi dihentikan.")
        except Exception as e:
            msg.append(f"Warning saat menutup SSH: {e}")
        _active_ssh = None

    if not msg:
        return True, "Tidak ada preview aktif."
    return True, " ".join(msg)


def is_preview_running() -> bool:
    return _active_preview is not None and _active_preview.is_alive()


def _run_ssh_cmd(client: paramiko.SSHClient, cmd: str, timeout: int = 30) -> Tuple[int, str, str]:
    """Eksekusi 1 perintah SSH, tunggu sampai selesai, return (exit_code, stdout, stderr)."""
    stdin, stdout, stderr = client.exec_command(cmd, timeout=timeout)
    exit_code = stdout.channel.recv_exit_status()
    out = stdout.read().decode(errors="ignore")
    err = stderr.read().decode(errors="ignore")
    return exit_code, out, err


def capture_high_res(
    host: str,
    user: str,
    password: str,
    save_dir: Optional[Path] = None,
) -> Tuple[bool, str, Optional[Path]]:
    """Capture a 4608x2592 image on the Pi via rpicam-still and SCP back to laptop.

    Robust version:
    - Hentikan proses kamera lain & preview supaya kamera bebas.
    - Tulis ke `PI_REMOTE_CAPTURE_PATH` (default /tmp/acne_capture.jpg).
    - Cek exit code rpicam-still + stderr.
    - Verifikasi file ada di Pi via SFTP stat sebelum download (bukan
      hanya menerka path).
    - Hapus file remote setelah transfer supaya /tmp tidak penuh.

    Returns (ok, message, local_path).
    """
    if is_preview_running():
        stop_live_preview()
        time.sleep(1.0)

    save_dir = save_dir or TMP_DIR
    save_dir.mkdir(parents=True, exist_ok=True)
    local_path = save_dir / f"capture_{int(time.time())}.jpg"

    client: Optional[paramiko.SSHClient] = None
    try:
        client = _ssh_connect(host, user, password)

        # 1) Bebaskan kamera (kalau ada proses gantung).
        _run_ssh_cmd(
            client,
            "sudo pkill -f rpicam-still; sudo pkill -f libcamera-still; "
            "sudo pkill -f libcamera-vid; sudo pkill -f rpicam-vid; sleep 1",
            timeout=15,
        )
        time.sleep(1.5)

        # 2) Pastikan direktori target ada & writable.
        remote_dir = str(Path(PI_REMOTE_CAPTURE_PATH).parent.as_posix())
        _run_ssh_cmd(client, f"mkdir -p {remote_dir} && rm -f {PI_REMOTE_CAPTURE_PATH}")

        # 3) Capture. JANGAN pakai --zsl (sering bikin file tidak ter-flush
        #    di beberapa firmware). JPG quality 95 cukup untuk inference.
        cmd = (
            f"rpicam-still -o {PI_REMOTE_CAPTURE_PATH} "
            f"--width 4608 --height 2592 --quality 95 "
            f"--nopreview --timeout 3000"
        )
        exit_code, out, err = _run_ssh_cmd(client, cmd, timeout=60)

        # rpicam-still kadang menulis log ke stderr walau sukses -
        # andalkan exit code & verifikasi file.
        fatal_markers = ("failed to acquire", "Failed to start", "no cameras available")
        if any(m in err for m in fatal_markers):
            return False, f"Kamera gagal diakses di Pi: {err.strip()[:300]}", None

        # 4) Tunggu sebentar agar file ter-flush ke disk.
        time.sleep(0.7)

        # 5) Verifikasi keberadaan file via SFTP stat (lebih akurat
        #    daripada langsung sftp.get yang melempar errno 2).
        sftp = client.open_sftp()
        try:
            try:
                attrs = sftp.stat(PI_REMOTE_CAPTURE_PATH)
            except IOError:
                # File memang tidak ada -> kasih pesan yang berguna.
                ls_code, ls_out, ls_err = _run_ssh_cmd(
                    client, f"ls -la {PI_REMOTE_CAPTURE_PATH} 2>&1"
                )
                return (
                    False,
                    "File capture tidak ditemukan di Pi setelah perintah selesai. "
                    f"Path: `{PI_REMOTE_CAPTURE_PATH}` | exit_code rpicam={exit_code} | "
                    f"stderr={err.strip()[:200] or '-'} | ls={ls_out.strip()[:200] or ls_err.strip()[:200]}",
                    None,
                )

            if attrs.st_size <= 0:
                return False, "File capture di Pi kosong (0 byte). Coba ulangi.", None

            sftp.get(PI_REMOTE_CAPTURE_PATH, str(local_path))
        finally:
            sftp.close()

        # 6) Cleanup remote.
        _run_ssh_cmd(client, f"rm -f {PI_REMOTE_CAPTURE_PATH}")

        if not local_path.exists() or local_path.stat().st_size <= 0:
            return False, "File tidak terdeteksi / kosong setelah transfer ke laptop.", None

        # 7) Rotasi agar kepala menghadap ke atas pada hasil capture.
        _auto_rotate_capture(local_path)

        size_mb = local_path.stat().st_size / (1024 * 1024)
        rot_note = ""
        if int(PI_CAPTURE_ROTATION_DEG) % 360 != 0:
            rot_note = f" | auto-rotate {int(PI_CAPTURE_ROTATION_DEG) % 360}deg CCW"
        return True, f"Capture berhasil: {local_path.name} ({size_mb:.2f} MB){rot_note}", local_path

    except paramiko.AuthenticationException:
        return False, "Autentikasi SSH gagal. Periksa username & password.", None
    except paramiko.SSHException as e:
        return False, f"Koneksi SSH bermasalah: {e}", None
    except Exception as e:
        return False, f"Gagal capture: {e}", None
    finally:
        if client is not None:
            try:
                client.close()
            except Exception:
                pass
