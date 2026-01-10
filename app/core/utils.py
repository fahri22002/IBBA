import csv
import os
import streamlit as st
from data import data

def show_auto_anotation():
    st.header("Automatic Annotation")
    st.info("Apakah Anda ingin melakukan automatic annotation pada frame yang telah diekstrak?")

def add_iteration():
    data.iteration += 1
    data.deletion_count = 0
    data.addition_count = 0
    data.is_done_auto_annot = False
    data.is_done_correction = False
    data.is_done_train = False

def show_iteration_info():
    st.subheader(f"Iterasi - {data.iteration}")

def count_ground_truth_boxes(label_dir):
    total = 0
    for f in os.listdir(label_dir):
        if f.endswith(".txt"):
            with open(os.path.join(label_dir, f), "r") as lf:
                for line in lf:
                    if line.strip():
                        total += 1
    return total


def show_evaluation_summary(workdir, i, new_model, yaml_path, CLASS_NAME, run_name):
    # === Evaluasi model (tampilkan gambar + metrik) ===
    st.success(f"✅ Iterasi {i} selesai! Model baru: {workdir}/{i}-iter/{run_name}/weights/best.pt")
    st.info("Menjalankan evaluasi pada dataset val...")
    val_results = None
    try:
        # val_results = new_model.val(data=yaml_path, imgsz=640, verbose=False, plots=True)
        val_results = new_model.val(
            data=yaml_path,
            imgsz=640,
            project=f"{workdir}/{i}-iter",
            name="val_internal",
            plots=True
        )

    except Exception:
        print("Evaluasi via `model.val()` gagal atau tidak tersedia. Mencari file plot yang di-generate jika ada...")

    # cari gambar evaluasi di lokasi run dan di runs/val
    candidate_dirs = [f"{workdir}/{i}-iter/{run_name}", f"{workdir}/{i}-iter/{run_name}/results", "runs/val"]
    eval_images = []
    for d in candidate_dirs:
        if os.path.exists(d):
            for root, _, files in os.walk(d):
                for f in files:
                    if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                        if any(k in f.lower() for k in ('confusion', 'pr', 'precision', 'recall', 'results')):
                            eval_images.append(os.path.join(root, f))
    if not eval_images:
        for d in candidate_dirs:
            if os.path.exists(d):
                for root, _, files in os.walk(d):
                    for f in files:
                        if f.lower().endswith('.png'):
                            eval_images.append(os.path.join(root, f))

    if eval_images:
        st.subheader("Gambar Evaluasi")
        for p in eval_images[:6]:
            try:
                st.image(p)
            except Exception:
                st.write(p)
    else:
        st.info("Tidak ditemukan gambar evaluasi otomatis (confusion/pr).")

    # Tampilkan metrik jika val_results tersedia
    st.subheader("Metrik Evaluasi (ringkasan)")
    if val_results is not None:
        try:
            # coba ambil metrics dari object
            metrics = getattr(val_results, 'metrics', None)
            if metrics and isinstance(metrics, dict):
                for k, v in metrics.items():
                    st.write(f"- **{k}**: {v}")
            else:
                # coba cari file results.csv yang di-generate oleh YOLO
                results_csv = f"{workdir}/{i}-iter/{run_name}/results.csv"
                if os.path.exists(results_csv):
                    import pandas as pd
                    try:
                        df = pd.read_csv(results_csv)
                        st.dataframe(df, use_container_width=True)
                    except Exception:
                        st.text("File results.csv ditemukan tapi tidak bisa dibaca.")
                else:
                    st.text("Metrik detail tidak tersedia, tapi training selesai.")
        except Exception as e:
            print(f"Gagal mengekstrak metrik: {e}")
    else:
        st.info("Hasil validasi tidak tersedia.")

    st.markdown("---")
    
    # === Evaluasi pada dataset external (otomatisasi test) ===
    st.subheader("Evaluasi pada Dataset External")

    if data.eval_path and os.path.exists(data.eval_path):
        eval_test_dir = os.path.join(data.eval_path, "test")

        if os.path.exists(eval_test_dir):
            try:
                # cari yaml di eval_path
                eval_yaml_path = None
                for f in os.listdir(data.eval_path):
                    if f.endswith(".yaml") or f.endswith(".yml"):
                        eval_yaml_path = os.path.join(data.eval_path, f)
                        break

                yaml_to_use = eval_yaml_path
                temp_yaml = None

                if eval_yaml_path:
                    with open(eval_yaml_path, "r") as yf:
                        txt = yf.read()

                    if "val:" not in txt:
                        temp_yaml = os.path.join(f"{workdir}/{i}-iter", f"temp_eval_{i}.yaml")
                        with open(temp_yaml, "w") as tf:
                            tf.write(
                                f"val: {eval_test_dir}\n"
                                f"nc: 1\n"
                                f"names: ['{CLASS_NAME}']\n"
                            )
                        yaml_to_use = temp_yaml
                else:
                    temp_yaml = os.path.join(f"{workdir}/{i}-iter", f"temp_eval_{i}.yaml")
                    with open(temp_yaml, "w") as tf:
                        tf.write(
                            f"val: {eval_test_dir}\n"
                            f"nc: 1\n"
                            f"names: ['{CLASS_NAME}']\n"
                        )
                    yaml_to_use = temp_yaml

                # jalankan evaluasi
                val_results = new_model.val(
                    data=yaml_to_use,
                    conf=0.5,
                    imgsz=640,
                    verbose=False,
                    plots=True,
                    project=f"{workdir}/{i}-iter",
                    name=f"eval_external_{i}"
                )

                # tampilkan metrik utama
                st.subheader("📊 Metrik Evaluasi (External Dataset)")

                precision = val_results.box.mp
                recall = val_results.box.mr
                map50 = val_results.box.map50
                map5095 = val_results.box.map

                c1, c2, c3, c4 = st.columns(4)
                with c1:
                    st.metric("Precision", f"{precision:.4f}")
                with c2:
                    st.metric("Recall", f"{recall:.4f}")
                with c3:
                    st.metric("mAP@50", f"{map50:.4f}")
                with c4:
                    st.metric("mAP@50-95", f"{map5095:.4f}")

                # hitung ngt (ground truth boxes)
                label_dir = os.path.join(eval_test_dir, "labels")
                ngt = count_ground_truth_boxes(label_dir) if os.path.exists(label_dir) else 0

                # hitung np (predicted boxes)
                np = val_results.confusion_matrix.matrix[0, :].sum()

                est_add = ngt * (1 - recall)
                est_del = np * (1 - precision)

                if get_rill_values_by_iteration(f"{workdir}/count.csv", i):
                    rill_del, rill_add = get_rill_values_by_iteration(f"{workdir}/count.csv", i)
                    if rill_del >= 0 and rill_add >= 0:
                        data.deletion_count = rill_del
                        data.addition_count = rill_add
                c3, c4 = st.columns(2)
                with c3:
                    st.metric("Riil Deletion", f"{data.deletion_count:.4f}")
                with c4:
                    st.metric("Riil Addition", f"{data.addition_count:.4f}")

                # st.info(f"np = {np} ngt = {ngt}")

                st.image(f"{workdir}/{i}-iter/eval_external_{i}/confusion_matrix.png", caption="Confusion Matrix", use_column_width=True)


                # hapus yaml sementara
                if temp_yaml and os.path.exists(temp_yaml):
                    os.remove(temp_yaml)

            except Exception as e:
                print(f"Evaluasi dataset external gagal: {e}")
        else:
            print(f"Folder test tidak ditemukan di {data.eval_path}")
    else:
        st.info("Dataset external belum dikonfigurasi")

    st.markdown("---")

    return precision, recall, map50, map5095, 0, 0

def show_training_confirmation(): 
    st.header("Training")
    st.info("Apakah Anda ingin memulai proses training dengan data yang telah dikoreksi secara manual?")
def show_correction_confirmation(): 
    st.header("Corrections")
    st.info("Apakah Anda ingin memulai proses correction dengan data yang dianotasi otomatis?")

def simpan_evaluasi_otomatisasi(
    model_path,
    precision,
    recall,
    map50,
    map5095,
    iteration,
    ril_deletion,
    ril_addition,
    est_deletion,
    est_addition
):
    eval_dir = f"{data.working_dir}/evaluasi_summary"
    os.makedirs(eval_dir, exist_ok=True)

    csv_path = os.path.join(eval_dir, "otomatisasi_eval.csv")
    file_exists = os.path.exists(csv_path)

    with open(csv_path, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        # tulis header hanya jika file belum ada
        if not file_exists:
            writer.writerow([
                "model_path",
                "iteration",
                "precision",
                "recall",
                "mAP50",
                "mAP50_95",
                "ril_deletion",
                "ril_addition",
                # "est_deletion",
                # "est_addition"
            ])

        # tambahkan baris evaluasi
        writer.writerow([
            model_path,
            iteration,
            f"{precision:.6f}",
            f"{recall:.6f}",
            f"{map50:.6f}",
            f"{map5095:.6f}",
            ril_deletion,
            ril_addition,
            # est_deletion,
            # est_addition
        ])



def get_rill_values_by_iteration(csv_path, iteration):
    if not os.path.isfile(csv_path):
        return 0, 0

    with open(csv_path, mode='r', newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if int(row['iteration']) == int(iteration):
                return int(row['rill_del']), int(row['rill_add'])

    return -1, -1


def save_iteration_result(csv_path, iteration, rill_del, rill_add):
    file_exists = os.path.isfile(csv_path)

    with open(csv_path, mode='a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)

        # tulis header jika file belum ada
        if not file_exists:
            writer.writerow(['iteration', 'rill_del', 'rill_add'])

        writer.writerow([iteration, rill_del, rill_add])

#belum dipakai
def simpan_evaluasi_generalization(
    model_path,
    precision,
    recall,
    map50,
    map5095,
    iteration
):
    eval_dir = f"{data.working_dir}/evaluasi_summary"
    os.makedirs(eval_dir, exist_ok=True)

    csv_path = os.path.join(eval_dir, "evaluasi_generalization.csv")
    file_exists = os.path.exists(csv_path)

    with open(csv_path, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        # header hanya ditulis sekali
        if not file_exists:
            writer.writerow([
                "model_path",
                "iteration",
                "precision",
                "recall",
                "mAP50",
                "mAP50_95"
            ])

        # append hasil evaluasi generalization
        writer.writerow([
            model_path,
            iteration,
            f"{precision:.6f}",
            f"{recall:.6f}",
            f"{map50:.6f}",
            f"{map5095:.6f}"
        ])



