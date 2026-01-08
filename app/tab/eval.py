import streamlit as st

def show():
    st.info("INI EVAL")

import os
import shutil

# ===== PATH KONFIGURASI =====
SRC_IMAGES_DIR = r"C:\Users\USER\Documents\a skripsi\riset\history\10-ep-5-it\frames_iter"

DST_IMAGES_DIR = r"C:\Users\USER\Documents\a skripsi\riset\history\10-ep-5-it\5-iter\data_iter5 - Copy\images"
DST_LABELS_DIR = r"C:\Users\USER\Documents\a skripsi\riset\history\10-ep-5-it\5-iter\data_iter5 - Copy\labels"

# Ekstensi gambar yang dianggap valid
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")

# ===== PASTIKAN FOLDER TUJUAN ADA =====
os.makedirs(DST_IMAGES_DIR, exist_ok=True)
os.makedirs(DST_LABELS_DIR, exist_ok=True)

copied_count = 0
skipped_count = 0

for filename in os.listdir(SRC_IMAGES_DIR):
    if not filename.lower().endswith(IMAGE_EXTENSIONS):
        continue

    src_image_path = os.path.join(SRC_IMAGES_DIR, filename)
    dst_image_path = os.path.join(DST_IMAGES_DIR, filename)

    # Jika gambar sudah ada, abaikan
    if os.path.exists(dst_image_path):
        skipped_count += 1
        continue

    # Copy gambar
    shutil.copy2(src_image_path, dst_image_path)
    copied_count += 1

    # Buat file label kosong jika belum ada
    label_name = os.path.splitext(filename)[0] + ".txt"
    dst_label_path = os.path.join(DST_LABELS_DIR, label_name)

    if not os.path.exists(dst_label_path):
        open(dst_label_path, "w").close()

print("Proses selesai")
print(f"Gambar dicopy   : {copied_count}")
print(f"Gambar di-skip  : {skipped_count}")

import os
import math
import csv
import yaml
from ultralytics import YOLO

# =========================
# 1. KONFIGURASI PATH
# =========================
# Path ke data.yaml asli Anda
ORIGINAL_YAML = r"C:\Users\USER\Documents\a skripsi\riset\history\10-ep-5-it\test_est\data.yaml"
# Folder tempat gambar test berada
TEST_IMAGE_DIR = r"C:\Users\USER\Documents\a skripsi\riset\history\10-ep-5-it\test_est\test\images"
ROOT_DIR = r"C:/Users/USER/Documents/a skripsi/riset/history/10-ep-5-it"
OUTPUT_DIR = os.path.abspath("./hasil_evaluasi_split")

MODEL_PATHS = [
    r"C:\Users\USER\Documents\a skripsi\riset\model_awal\best.pt",
    os.path.join(ROOT_DIR, "1-iter/model1/weights/best.pt"),
    os.path.join(ROOT_DIR, "2-iter/model2/weights/best.pt"),
    os.path.join(ROOT_DIR, "3-iter/model3/weights/best.pt"),
    os.path.join(ROOT_DIR, "4-iter/model4/weights/best.pt"),
]

CSV_OUT = os.path.join(OUTPUT_DIR, "metrics_per_slice.csv")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================
# 2. AMBIL SEMUA GAMBAR & BAGI MENJADI 5 CHUNK
# ============================================
all_images = [os.path.join(TEST_IMAGE_DIR, f) for f in os.listdir(TEST_IMAGE_DIR) 
              if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
all_images.sort()

total_images = len(all_images)
num_models = len(MODEL_PATHS)
images_per_model = math.ceil(total_images / num_models)

# =========================
# 3. HEADER CSV
# =========================
header = [
    "iteration", "TP", "FP", "FN", "precision", "recall", "f1_score", 
    "mAP50", "mAP50_95", "total_images_in_slice", "start_file", "end_file"
]
rows = []

# ============================================
# 4. LOOP EVALUASI PER MODEL (1/5 DATA)
# ============================================
for idx, model_path in enumerate(MODEL_PATHS, start=1):
    start_idx = (idx - 1) * images_per_model
    end_idx = min(idx * images_per_model, total_images)
    subset_images = all_images[start_idx:end_idx]

    if not subset_images:
        continue

    print(f"\n>>> Mengevaluasi Model {idx} pada gambar {start_idx+1} sampai {end_idx}...")

    # A. Buat file .txt sementara yang berisi daftar path gambar untuk slice ini
    temp_list_path = os.path.join(OUTPUT_DIR, f"temp_list_iter_{idx}.txt")
    with open(temp_list_path, "w") as f:
        for img_path in subset_images:
            f.write(img_path + "\n")

    # B. Buat file data_temp.yaml yang mengarah ke file .txt tadi
    # Ini trik agar YOLO val HANYA membaca gambar di dalam list tersebut
    with open(ORIGINAL_YAML, 'r') as f:
        yaml_content = yaml.safe_load(f)
    
    yaml_content['val'] = temp_list_path # Ubah path validasi ke list sementara
    
    temp_yaml_path = os.path.join(OUTPUT_DIR, f"temp_data_iter_{idx}.yaml")
    with open(temp_yaml_path, 'w') as f:
        yaml.dump(yaml_content, f)

    # C. Jalankan Validasi
    model = YOLO(model_path)
    metrics = model.val(
        data=temp_yaml_path,
        imgsz=640,
        conf=0.51,
        iou=0.8,
        project=OUTPUT_DIR,
        name=f"val_results_iter_{idx}",
        plots=True,
        save_json=False
    )

    # D. Ekstraksi Metrik dari Confusion Matrix
    # YOLO v8/v11: cm[0,0]=TP, cm[0,1]=FP (Background), cm[1,0]=FN
    cm = metrics.confusion_matrix.matrix
    TP = int(cm[0, 0])
    FP = int(cm[0, 1])
    FN = int(cm[1, 0])
    
    # Metrik Tambahan
    # p[0], r[0], f1[0] mengambil nilai untuk class pertama (indeks 0)
    precision = float(metrics.box.p[0])
    recall = float(metrics.box.r[0])
    f1 = float(metrics.box.f1[0])
    mAP50 = float(metrics.box.map50)
    mAP5095 = float(metrics.box.map)

    # Simpan ke list rows
    rows.append([
        idx, TP, FP, FN, precision, recall, f1, 
        mAP50, mAP5095, len(subset_images),
        os.path.basename(subset_images[0]), 
        os.path.basename(subset_images[-1])
    ])
    
    # Hapus file temporary agar folder bersih
    os.remove(temp_list_path)
    os.remove(temp_yaml_path)

# =========================
# 5. SIMPAN KE CSV AKHIR
# =========================
with open(CSV_OUT, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(header)
    writer.writerows(rows)

print(f"\n✅ Selesai! Statistik per 1/5 bagian dataset disimpan di: {CSV_OUT}")