import streamlit as st
from ultralytics import YOLO
from data import data
import os

done = False
def inputs():
    st.header("Input Data untuk Training Awal")
    data.epochs_first_training = st.number_input("Masukkan banyak epoch :", min_value=1, value=data.epochs_first_training, step=1, format="%d")
    data.first_training_dataset_dir = st.text_input("Masukkan Path dataset :",  "C:/Users/USER/Documents/a skripsi/work/test otomatisasi")
    data.train_awal_dir = st.text_input("Masukkan Path Destinasi Model :", os.path.join(os.path.dirname(os.getcwd()), "self-training"))

    if st.button("Training", key="Training-btn"):
        st.session_state.first_train_stage = "training"
        done = False
        st.rerun()

def train():
    i = "Awal"
    st.write(f"🚀 Training model Baseline...")

    dataset_dir = data.first_training_dataset_dir
    train_dir = f"{dataset_dir}/train"
    val_dir = f"{dataset_dir}/val"
    
    yaml_path = f"{dataset_dir}/data.yaml"
    if not os.path.exists(yaml_path):
        st.error(f"❌ File konfigurasi tidak ditemukan di: {yaml_path}")
        return

    # Training model
    new_model = YOLO("yolo11n")
    run_name = f"model{i}"
    project_dir = f"{data.train_awal_dir}/first_training"
    new_model.train(
        data=yaml_path,
        epochs=data.epochs_first_training,
        imgsz=640,
        project=project_dir,
        name=run_name
    )
    global done
    done = True

    # --- BAGIAN EVALUASI DAN TAMPILAN HASIL ---
    st.header("Hasil Evaluasi Model")
    
    # Path tempat hasil disimpan: project/run_name
    results_dir = os.path.join(project_dir, run_name)
    
    if os.path.exists(results_dir):
        # 1. Confusion Matrix
        cm_path = os.path.join(results_dir, "confusion_matrix.png")
        if os.path.exists(cm_path):
            st.subheader("Confusion Matrix")
            st.image(cm_path, caption="Confusion Matrix Model")
            # Trigger image untuk Confusion Matrix
            
        # 2. Precision-Recall Curve (P-R Curve)
        pr_path = os.path.join(results_dir, "PR_curve.png")
        if os.path.exists(pr_path):
            st.subheader("Precision-Recall (P-R) Curve")
            st.image(pr_path, caption="Precision-Recall Curve per Kelas")
            
        # 3. Metrik Lain (Jika ada)
        st.subheader("Metrik Kinerja")
        metrics_path = os.path.join(results_dir, "results.png")
        if os.path.exists(metrics_path):
            st.image(metrics_path, caption="Grafik Metrik Kinerja (Loss, mAP, dll.) selama Epoch")
            
        # 4. Contoh Hasil Validasi
        val_batch0_pred_path = os.path.join(results_dir, "val_batch0_pred.jpg")
        if os.path.exists(val_batch0_pred_path):
            st.subheader("Contoh Hasil Deteksi pada Data Validasi")
            st.image(val_batch0_pred_path, caption="Deteksi pada Batch Pertama Data Validasi")
            
    else:
        st.error("Folder hasil training tidak ditemukan. Pastikan path project dan name sudah benar.")


def main():
    if "first_train_stage" not in st.session_state:
        st.session_state.first_train_stage = "input"
    
    if st.session_state.first_train_stage == "input":
        inputs()
    elif st.session_state.first_train_stage == "training":
        if done:
            st.info("SUDAH TRAINING")
        else:
            train()