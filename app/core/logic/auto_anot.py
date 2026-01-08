import os
import shutil
import math
import streamlit as st
from ultralytics import YOLO
from data import data

def automatic_annotationation(model_path, frame_path, reject_path, out_dir, conf_thresh=0.5, max_iter=1):
    """
    Melakukan prediksi hanya pada sebagian frame (.jpg) di folder frame_path sesuai iterasi.
    Hasil berupa gambar dan label YOLO disimpan di out_dir.
    """
    st.info(f"melakukan sebanyak {max_iter} iterasi")
    st.session_state.frame_index = 0

    frames = sorted([f for f in os.listdir(frame_path) if f.lower().endswith(".jpg")])
    total_frames = len(frames)
    
    frames_per_iter = math.ceil(total_frames / max_iter)
    if data.iteration > 1:
        start_idx = frames_per_iter * (data.iteration - 1)
        st.session_state.frame_index = min(start_idx, total_frames)

    if "reject_index" not in st.session_state:
        st.session_state.reject_index = 0
    
    i = data.iteration
    st.write(f"🧠 Mulai semi-anotate labeling (Iterasi {i})")

    
    rejects = rejects = sorted([f for f in os.listdir(reject_path) if f.lower().endswith(".jpg")]) if os.path.exists(reject_path) else []
    total_rejects = len(rejects)
    st.info(f"TOTAL REJECT = {total_rejects}")
    st.info(f"START ANOT = {st.session_state.frame_index}")


    dataset_before = f"{data.working_dir}/{i-1}-iter/data_iter{i-1}"
    path_img_before = f"{dataset_before}/images"
    if os.path.exists(dataset_before):
        st.info(f"TOTAL BEFORE = {len(os.listdir(path_img_before))}")
        for root, dirs, files in os.walk(dataset_before):
            # Hitung path relatif dari dataset_before agar struktur folder tetap sama
            rel_path = os.path.relpath(root, dataset_before)
            dest_dir = os.path.join(out_dir, rel_path)
            os.makedirs(dest_dir, exist_ok=True)

            for file in files:
                src_file = os.path.join(root, file)
                dest_file = os.path.join(dest_dir, file)
                shutil.copy2(src_file, dest_file)

    if not frames:
        st.warning(f"Tidak ada file .jpg ditemukan di folder: {frame_path}")
        return

    # Hitung subset frame untuk iterasi saat ini
    start_frame_idx = st.session_state.frame_index
    end_frame_idx = min(total_frames, start_frame_idx + frames_per_iter)

    start_reject_idx = st.session_state.reject_index
    end_reject_idx = total_rejects

    # Cek apakah iterasi ini sudah selesai
    if start_frame_idx >= total_frames and start_reject_idx>=end_reject_idx:
        st.success("✅ Semua frame sudah selesai diproses!")
        st.session_state.ibba_iter_stage = "correction"
        return

    # Ambil subset frame untuk iterasi ini
    current_frames = frames[start_frame_idx:end_frame_idx]
    st.info(f"Memproses frame {start_frame_idx + 1} hingga {end_frame_idx} dari total {total_frames}")
    current_reject = rejects[start_reject_idx:end_reject_idx]
    st.info(f"Memproses frame {start_reject_idx + 1} hingga {end_reject_idx} dari rejections")

    # Cek model
    if i > 1:
        model_path = f"{data.working_dir}/{i-1}-iter/model{i-1}/weights/best.pt"
    if not os.path.exists(model_path):
        st.error(f"Model tidak ditemukan: {model_path}")
        return

    # Load model
    st.info(f"Memuat model dari: {model_path}")
    model = YOLO(model_path)

    # Siapkan folder output
    images_out = os.path.join(out_dir, "images")
    labels_out = os.path.join(out_dir, "labels")
    os.makedirs(images_out, exist_ok=True)
    os.makedirs(labels_out, exist_ok=True)

    # Inisialisasi progress bar
    progress_bar = st.progress(0)
    progress_text = st.empty()
    

    # Loop hanya pada subset frame
    for j, f in enumerate(current_frames, start=1):
        img_path = os.path.join(frame_path, f)
        results = model.predict(img_path, conf=conf_thresh, verbose=False)
        r = results[0]

        # Simpan gambar hasil prediksi
        pred_img = r.plot()
        # cv2.imwrite(os.path.join(images_out, f), pred_img)
        shutil.copy2(img_path,os.path.join(images_out, f))

        # Simpan label YOLO
        out_label_path = os.path.join(labels_out, os.path.splitext(f)[0] + ".txt")
        with open(out_label_path, "w") as label_file:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                if conf < conf_thresh:
                    continue
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                h_img, w_img = r.orig_img.shape[:2]
                x_c = ((x1 + x2) / 2) / w_img
                y_c = ((y1 + y2) / 2) / h_img
                w = (x2 - x1) / w_img
                h = (y2 - y1) / h_img
                label_file.write(f"{cls_id} {x_c:.6f} {y_c:.6f} {w:.6f} {h:.6f}\n")

        # Update progress hanya berdasarkan subset frame
        progress = j / len(current_frames)
        progress_bar.progress(progress)
        progress_text.text(f"Memproses frame {j}/{end_frame_idx-start_frame_idx} ({progress*100:.1f}%)")

    if data.is_include_reject_in_auto_annot:
        # Loop hanya pada subset reject
        for j, f in enumerate(current_reject, start=1):
            img_path = os.path.join(reject_path, f)
            results = model.predict(img_path, conf=conf_thresh, verbose=False)
            r = results[0]

            # Simpan gambar hasil prediksi
            pred_img = r.plot()
            # cv2.imwrite(os.path.join(images_out, f), pred_img)
            shutil.copy2(img_path,os.path.join(images_out, f))

            # Simpan label YOLO
            out_label_path = os.path.join(labels_out, os.path.splitext(f)[0] + ".txt")
            with open(out_label_path, "w") as label_file:
                for box in r.boxes:
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    if conf < conf_thresh:
                        continue
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    h_img, w_img = r.orig_img.shape[:2]
                    x_c = ((x1 + x2) / 2) / w_img
                    y_c = ((y1 + y2) / 2) / h_img
                    w = (x2 - x1) / w_img
                    h = (y2 - y1) / h_img
                    label_file.write(f"{cls_id} {x_c:.6f} {y_c:.6f} {w:.6f} {h:.6f}\n")

            # Update progress hanya berdasarkan subset frame
            progress = j / len(current_reject)
            progress_bar.progress(progress)
            progress_text.text(f"Memproses Reject {j}/{end_reject_idx-start_reject_idx} ({progress*100:.1f}%)")
    
    # Update index untuk iterasi berikutnya
    st.session_state.frame_index = end_frame_idx

    st.success(f"Selesai iterasi {i}! Frame {start_frame_idx + 1}–{end_frame_idx} telah disimpan di: {out_dir}")
    data.is_done_auto_annot = True
    if st.button("Next", key="btn-manual-correction"):
        st.rerun()

