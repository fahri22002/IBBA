import streamlit as st
from streamlit_drawable_canvas import st_canvas
from PIL import Image
import os
from data import data
import shutil
from core.logic.corrections.utils import back_n_frames
from core.utils import get_rill_values_by_iteration, save_iteration_result
def manual_addition():
    workdir = data.working_dir
    # st.set_page_config(layout="wide")
    st.title("Bounding Box Tool (Drag to Draw)")
    iter_num = data.iteration
    # st.header(f"ITERASI KE-{iter_num}")
    rejected_dir = f"{workdir}/{iter_num}-iter/reject"
    dataset_dir = f"{workdir}/{iter_num}-iter/data_iter{iter_num}"


    os.makedirs(f"{dataset_dir}/images", exist_ok=True)
    os.makedirs(f"{dataset_dir}/labels", exist_ok=True)
    os.makedirs(rejected_dir, exist_ok=True)

    rejected_image = sorted(os.listdir(f"{rejected_dir}"))

    if "current_rejected_image_idx" not in st.session_state:
        st.session_state.current_rejected_image_idx = 0
    img_idx = st.session_state.current_rejected_image_idx
    if img_idx >= len(rejected_image):
        st.success("Semua gambar reject telah selesai dikoreksi.")
        st.session_state.current_rejected_image_idx = 0
        data.manual_corrections_mode = 0
        # st.session_state.on_addition = False
        # st.session_state.on_correction = True
        data.is_done_correction = True
        data.is_on_correction = False
        st.session_state.ibba_stage = "training"
        save_iteration_result(f"{workdir}/count.csv", iter_num, data.deletion_count, data.addition_count)
        st.rerun()
        return
    
    img_name = rejected_image[img_idx]
    IMAGE_PATH = f"{rejected_dir}/{img_name}"
    st.info(f"Saat ini sedang di {img_idx} / {len(rejected_image)}")
    # ------------------------------------------------


    if not os.path.exists(IMAGE_PATH):
        st.error(f"Gambar '{IMAGE_PATH}' tidak ditemukan.")
        st.stop()

    # Load image
    img = Image.open(IMAGE_PATH).convert("RGB")
    W, H = img.size
    # Scale faktor dan ukuran canvas (gunakan int)
    scale = 0.4
    canvas_w = int(W * scale)
    canvas_h = int(H * scale)

    # Initialize session
    if "bboxes" not in st.session_state:
        st.session_state.bboxes = []

    # Canvas options
    st.write("Drag mouse untuk menggambar bounding box di atas gambar.")

    if "canvas_key" not in st.session_state:
        st.session_state.canvas_key = 0 
    
    # st.markdown("""
    #     <style>
    #     .stCanvasToolbar {display: none !important;}
    #     </style>
    # """, unsafe_allow_html=True)
    st.markdown(
    f"""
    <style>
    [data-testid="stCanvas"] iframe {{
        width:{canvas_w}px !important;
        height:{canvas_h}px !important;
    }}
    </style>
    """,
    unsafe_allow_html=True
)
    _, col, _ = st.columns([0.1, 4, 0.1])
    with col:
        canvas_result = st_canvas(
            fill_color="rgba(255, 0, 0, 0.2)",  # transparan merah
            stroke_width=2,
            stroke_color="red",
            background_image=img,
            update_streamlit=True,
            height=canvas_h,
            width=canvas_w,
            drawing_mode="rect",  # mode kotak
            key=f"canvas{st.session_state.canvas_key}",
            # css={"width": f"{canvas_w}px", "height": f"{canvas_h}px"}
        )

    # Proses hasil bounding box
    if canvas_result.json_data is not None:
        import json
        # objs = json.loads(canvas_result.json_data)["objects"]
        objs = canvas_result.json_data["objects"]

        bboxes = []
        for obj in objs:
            if obj["type"] == "rect":
                x = obj["left"]
                y = obj["top"]
                w = obj["width"]
                h = obj["height"]
                # konversi ke koordinat gambar asli (undo scale)
                # hindari div0 walau tidak mungkin karena scale > 0
                x1_orig = int(round(x / scale))
                y1_orig = int(round(y / scale))
                x2_orig = int(round((x + w) / scale))
                y2_orig = int(round((y + h) / scale))

                # clamp agar berada di dalam batas gambar asli
                x1_orig = max(0, min(W, x1_orig))
                x2_orig = max(0, min(W, x2_orig))
                y1_orig = max(0, min(H, y1_orig))
                y2_orig = max(0, min(H, y2_orig))

                bboxes.append((x1_orig, y1_orig, x2_orig, y2_orig))
        st.session_state.bboxes = bboxes

    # Tampilkan hasil
    st.write("Bounding boxes (pixel):")
    if st.session_state.bboxes:
        for i, (x1, y1, x2, y2) in enumerate(st.session_state.bboxes):
            st.write(f"{i+1}. ({x1},{y1}) → ({x2},{y2})")
    else:
        st.write("Belum ada bounding box.")

    # Tombol simpan YOLO
    save_name = st.text_input("Nama file label (tanpa ekstensi):", value=os.path.splitext(os.path.basename(IMAGE_PATH))[0])

    if st.button("💾 Save (YOLO Format)"):
        if not st.session_state.bboxes:
            st.warning("Tidak ada bounding box untuk disimpan.")
        else:
            img_n = rejected_image[st.session_state.current_rejected_image_idx]
            ip = f"{rejected_dir}/{img_n}"
            shutil.copy(ip, f"{dataset_dir}/images/{img_n}")
            os.makedirs(f"{dataset_dir}/labels", exist_ok=True)
            with open(f"{dataset_dir}/labels/{save_name}.txt", "w") as f:
                for (x1, y1, x2, y2) in st.session_state.bboxes:
                    xa, xb = sorted([x1, x2])
                    ya, yb = sorted([y1, y2])
                    x_center = (xa + xb) / 2.0 / W
                    y_center = (ya + yb) / 2.0 / H
                    w_norm = (xb - xa) / W
                    h_norm = (yb - ya) / H
                    f.write(f"0 {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}\n")
                    data.addition_count += 1
            # Hapus file dari rejected_dir
            p = f"{rejected_dir}/{img_n}"
            if os.path.exists(p):
                os.remove(p)
            st.session_state.action_stack.append(("y", img_n))
            st.success(f"Saved {len(st.session_state.bboxes)} boxes to labels/{save_name}.txt")
            st.success(f"Saved {len(st.session_state.bboxes)} images to {dataset_dir}/images/{img_n}")
            st.session_state.bboxes = []
        st.session_state.canvas_key += 1
        st.rerun()
    with st.container():
        fastinput = st.columns([1])[0]
        with fastinput:
            data.addition_skip_num = st.number_input("Jumlah Fast forward/backward", min_value=1, value=data.addition_skip_num,
                            step=1, key="fast_num_reject")
        skip_count = back_count = data.addition_skip_num
    with st.container():
        st.markdown("<div style='text-align:center;'>", unsafe_allow_html=True)
        space, c2, c1, space1 = st.columns([0.1, 1, 1, 0.1], gap="medium")
        with c1:
            if st.button(f"⏭️ Skip {skip_count}", key="skip10_btn"):
                for _ in range(skip_count):
                    if st.session_state.current_rejected_image_idx >= len(rejected_image): break
                    img_n = rejected_image[st.session_state.current_rejected_image_idx]
                    # ip = f"{rejected_dir}/{img_n}"
                    # shutil.copy(ip, f"{rejected_dir}/{img_n}")
                    st.session_state.action_stack.append(("n", img_n))
                    st.session_state.current_rejected_image_idx += 1
                st.rerun()

        with c2:
            if st.button(f"⬅️ Back {back_count}", key="back10_btn"):
                back_n_frames(back_count, dataset_dir, rejected_dir)
                st.session_state.current_rejected_image_idx = max(0, st.session_state.current_rejected_image_idx - back_count)
                st.rerun()
    
