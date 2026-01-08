import os
import cv2
import streamlit as st
from data import data
# ===============================
# Fungsi ekstraksi frame dengan progress bar
# ===============================
def extract_frames(video_path, out_dir):
    if not data.is_done_extract:

        os.makedirs(out_dir, exist_ok=True)
        vidcap = cv2.VideoCapture(video_path)

        total_frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
        count = 0

        progress_bar = st.progress(0, text="Ekstraksi frame sedang berjalan...")
        status_text = st.empty()

        while True:
            success, frame = vidcap.read()
            if not success:
                break

            cv2.imwrite(f"{out_dir}/frame_{count:05d}.jpg", frame)
            count += 1

            progress = int((count / total_frames) * 100)
            progress_bar.progress(progress, text=f"Ekstraksi {progress}% ({count}/{total_frames} frame)")
            status_text.text(f"Frame tersimpan: {count}/{total_frames}")

        vidcap.release()
        data.is_done_extract = True
        progress_bar.progress(100, text="✅ Ekstraksi selesai!")
        st.success(f"Selesai! Total {count} frame berhasil disimpan di {out_dir}")
    space, col1, space = st.columns([0.01, 1, 0.01])
    with col1:
        if st.button("Next", key="next-to-anotasi"):
            st.session_state.ibba_stage = "automatic annotation"
            data.is_done_auto_annot = False
            st.rerun()

def extract_frames_random(video_path, out_dir):
    if not data.is_done_extract:

        import random

        os.makedirs(out_dir, exist_ok=True)
        vidcap = cv2.VideoCapture(video_path)

        total_frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))

        progress_bar = st.progress(0, text="Ekstraksi frame sedang berjalan...")
        status_text = st.empty()

        # === BACA SEMUA FRAME ===
        frames = []
        while True:
            success, frame = vidcap.read()
            if not success:
                break
            frames.append(frame)

        vidcap.release()

        # === ACAK URUTAN FRAME ===
        random.shuffle(frames)

        # === SIMPAN DENGAN PENAMAAN YANG SAMA ===
        count = 0
        for frame in frames:
            cv2.imwrite(f"{out_dir}/frame_{count:05d}.jpg", frame)
            count += 1

            progress = int((count / total_frames) * 100)
            progress_bar.progress(
                progress,
                text=f"Ekstraksi Random {progress}% ({count}/{total_frames} frame)"
            )
            status_text.text(f"Frame tersimpan: {count}/{total_frames}")

        data.is_done_extract = True
        progress_bar.progress(100, text="✅ Ekstraksi selesai!")
        st.success(f"Selesai! Total {count} frame berhasil disimpan di {out_dir}")

    space, col1, space = st.columns([0.01, 1, 0.01])
    with col1:
        if st.button("Next", key="next-to-anotasi"):
            st.session_state.ibba_stage = "automatic annotation"
            data.is_done_auto_annot = False
            st.rerun()
