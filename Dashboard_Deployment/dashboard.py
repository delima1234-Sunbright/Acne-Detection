# Path Model : "C:\Dokumen\SKRIPSI\Bimbingan\Dari Server Kampus\HASIL FIX\runs_v5\progressive_yolov8n\weights\best.pt" 


import os
import time

import streamlit as st
import paramiko
from PIL import Image

st.set_page_config(page_title="Remote High-Res Capture", layout="wide")

st.title("📸 Remote Skin Capture (via SSH)")

# --- Sidebar untuk Koneksi ---
st.sidebar.header("🔑 Raspberry Pi Credentials")
host = st.sidebar.text_input("IP Address", value="10.10.169.249")
user = st.sidebar.text_input("Username", value="delima1234")
password = st.sidebar.text_input("Password", type="password", value="delima1234")
# Pakai /tmp supaya selalu writable & tidak tergantung home dir.
remote_path = "/tmp/acne_capture.jpg"
local_path = "captured_from_pi.jpg"

def capture_and_transfer():
    try:
        # 1. Membuat Koneksi SSH
        client = paramiko.SSHClient()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        
        st.info(f"Menghubungkan ke {host}...")
        client.connect(hostname=host, username=user, password=password)
        
        # 2. Bebaskan kamera dari proses lain yang sedang berjalan
        st.info("Membebaskan kamera dari proses lain...")
        client.exec_command(
            "sudo pkill -f rpicam-still; sudo pkill -f libcamera-still; "
            "sudo pkill -f libcamera-vid; sudo pkill -f rpicam-vid; sleep 1"
        )
        time.sleep(2)

        # 3. Pastikan target dir bersih.
        client.exec_command(f"rm -f {remote_path}")
        time.sleep(0.5)

        # 4. Perintah Ambil Gambar (Resolusi Maksimal 12MP).
        #    JANGAN pakai --zsl: di beberapa firmware bikin file tidak ter-flush.
        cmd = (
            f"rpicam-still -o {remote_path} "
            f"--width 4608 --height 2592 --quality 95 --nopreview --timeout 3000"
        )
        st.info("Kamera sedang mengambil gambar (12MP)...")
        stdin, stdout, stderr = client.exec_command(cmd)
        exit_code = stdout.channel.recv_exit_status()
        error = stderr.read().decode(errors="ignore")

        if any(m in error for m in ("failed to acquire", "Failed to start", "no cameras available")):
            st.error(f"Kamera gagal diakses. Detail: {error[:300]}")
            client.close()
            return False

        # 5. Verifikasi file ada via SFTP stat dulu, baru download.
        time.sleep(0.7)
        sftp = client.open_sftp()
        try:
            try:
                attrs = sftp.stat(remote_path)
            except IOError:
                st.error(
                    f"File capture tidak ditemukan di Pi: `{remote_path}` "
                    f"(exit_code={exit_code}, stderr={error.strip()[:200] or '-'})"
                )
                sftp.close()
                client.close()
                return False
            if attrs.st_size <= 0:
                st.error("File capture di Pi kosong (0 byte). Coba ulangi.")
                sftp.close()
                client.close()
                return False
            st.info("Mentransfer file ke laptop...")
            sftp.get(remote_path, local_path)
        finally:
            sftp.close()

        # 6. Cleanup remote.
        client.exec_command(f"rm -f {remote_path}")
        client.close()
        return True
    except Exception as e:
        st.error(f"Gagal: {e}")
        return False

# --- Tombol Utama ---
if st.button('🚀 AMBIL GAMBAR SEKARANG'):
    if capture_and_transfer():
        # Menampilkan Gambar di Dashboard
        if os.path.exists(local_path):
            img = Image.open(local_path)
            
            st.success(f"Selesai! Gambar diterima: {img.size[0]}x{img.size[1]} px")
            
            col1, col2 = st.columns([2, 1])
            with col1:
                st.image(img, caption="Hasil Jepretan Full Resolution", use_container_width=True)
            
            with col2:
                st.write("### Detail File")
                st.write(f"**Format:** JPG (Quality 95)")
                st.write(f"**Resolusi:** 4608 x 2592")
                st.write(f"**Ukuran:** {os.path.getsize(local_path) / (1024*1024):.2f} MB")

                with open(local_path, "rb") as file:
                    st.download_button(
                        label="Download Gambar Asli",
                        data=file,
                        file_name="skripsi_acne_12mp.jpg",
                        mime="image/jpeg"
                    )