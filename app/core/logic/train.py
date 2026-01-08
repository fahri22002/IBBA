import os
import shutil
import streamlit as st
from ultralytics import YOLO
from data import data
workdir = data.working_dir
CLASS_NAME = "pelat"
# === Fungsi bantu: Split dataset menjadi train dan val ===
def split(data_correction,train_dir,val_dir):

    os.makedirs(f"{train_dir}/images", exist_ok=True)
    os.makedirs(f"{train_dir}/labels", exist_ok=True)
    os.makedirs(f"{val_dir}/images", exist_ok=True)
    os.makedirs(f"{val_dir}/labels", exist_ok=True)

    # ambil semua file gambar dari data_correction
    all_images = sorted(os.listdir(f"{data_correction}/images"))
    n_total = len(all_images)
    n_val = max(1, int(0.2 * n_total))  # 20%

    # acak urutan supaya tidak bias
    import random
    random.shuffle(all_images)

    val_images = set(all_images[:n_val])
    train_images = set(all_images[n_val:])

    # copy train data
    for img_name in train_images:
        label_name = img_name.replace(".jpg", ".txt")
        src_img = os.path.join(data_correction, "images", img_name)
        src_label = os.path.join(data_correction, "labels", label_name)
        dst_img = os.path.join(train_dir, "images", img_name)
        dst_label = os.path.join(train_dir, "labels", label_name)
        shutil.copy(src_img, dst_img)
        if os.path.exists(src_label):
            shutil.copy(src_label, dst_label)

    # copy val data
    for img_name in val_images:
        label_name = img_name.replace(".jpg", ".txt")
        src_img = os.path.join(data_correction, "images", img_name)
        src_label = os.path.join(data_correction, "labels", label_name)
        dst_img = os.path.join(val_dir, "images", img_name)
        dst_label = os.path.join(val_dir, "labels", label_name)
        shutil.copy(src_img, dst_img)
        if os.path.exists(src_label):
            shutil.copy(src_label, dst_label)

    st.success(
        f"Dataset telah dibagi: {len(train_images)} train dan {len(val_images)} val "
        f"({len(all_images)} total)."
    )

# === Tahap 3: Training model baru ===
def train():
    if not data.is_done_train:
        i = data.iteration
        st.write(f"🚀 Training model baru (Iterasi {i})...")

        # Split dataset
        data_correction = f"{data.working_dir}/{i}-iter/data_iter{i}"
        dataset_dir = f"{data.working_dir}/{i}-iter/datatrain_iter{i}"
        train_dir = f"{dataset_dir}/train"
        val_dir = f"{dataset_dir}/val"
        split(data_correction=data_correction, train_dir=train_dir, val_dir=val_dir)
        for d in [train_dir, val_dir]:
            os.makedirs(f"{d}/images", exist_ok=True)
            os.makedirs(f"{d}/labels", exist_ok=True)

        # Buat YAML
        yaml_path = f"{dataset_dir}/data.yaml"
        with open(yaml_path, "w") as f:
            f.write(f"""
    train: {train_dir}
    val: {val_dir}

    nc: 1
    names: ['{CLASS_NAME}']
    """)

        # Training model
        model_path_weights = f"{workdir}/{i-1}-iter/model{i-1}/weights/best.pt"
        model_path_root = data.model_path
        if os.path.exists(model_path_weights):
            model_path = model_path_weights
        elif os.path.exists(model_path_root):
            st.warning(f"Model dari iterasi sebelumnya tidak ditemukan, menggunakan model awal: {model_path_root}")
            model_path = model_path_root
        else:
            st.error(f"Model tidak ditemukan: {model_path_weights} atau {model_path_root}")
            return
        new_model = YOLO(model_path)
        run_name = f"model{i}"
        project_path = f"{workdir}/{i}-iter"
        st.info("Epochs: " + str(data.epochs_per_iteration))
        new_model.train(
            data=yaml_path,
            epochs=data.epochs_per_iteration,
            imgsz=640,
            project=project_path,
            name=run_name
        )
        # === Evaluasi model (tampilkan gambar + metrik) ===
        from core.utils import show_evaluation_summary, simpan_evaluasi_otomatisasi
        precision, recall, map50, map5095, est_deletion, est_addition = show_evaluation_summary(workdir, i, new_model, yaml_path, CLASS_NAME, run_name)
        simpan_evaluasi_otomatisasi(model_path=f"{project_path}/{run_name}/weights/best.pt", precision=precision, recall=recall, map50=map50, map5095=map5095, iteration=i, ril_deletion=data.deletion_count, ril_addition=data.addition_count, est_deletion=est_deletion, est_addition=est_addition)