from PIL import Image, ImageDraw, ImageFont
from data import data
import streamlit as st
import os
import shutil
from core.logic.corrections.utils import draw_all_boxes, draw_single_box, back_n_frames, reject_image, save_image, push_action
def manual_correction():
    workdir = data.working_dir
    iter_num = data.iteration
    # st.header(f"ITERASI KE-{iter_num}")
    rejected_dir = f"{workdir}/{iter_num}-iter/reject"
    pseudo_dir = f"{workdir}/{iter_num}-iter/pseudo"
    dataset_dir = f"{workdir}/{iter_num}-iter/data_iter{iter_num}"
    # if "fast_num" not in st.session_state:
    #     st.session_state.fast_num = 10

    if "action_stack" not in st.session_state:
        st.session_state.action_stack = []

    #STACK UNTUK UNDO CHECKER
    # if st.session_state.action_stack:
        # recent_actions = st.session_state.action_stack[-10:]  # Ambil 10 terbaru
        # action_text = " | ".join([f"<b>{action}</b>: {img_name}" for action, img_name in recent_actions])
        # st.markdown(f"**Action Stack:**<br>{action_text}", unsafe_allow_html=True)
        # st.info(f"Banyak Stack: \n{len(st.session_state.action_stack)}")
    # else:
        # st.info("Action Stack: Kosong")

    os.makedirs(f"{dataset_dir}/images", exist_ok=True)
    os.makedirs(f"{dataset_dir}/labels", exist_ok=True)
    os.makedirs(rejected_dir, exist_ok=True)

    pseudo_images = sorted(os.listdir(f"{pseudo_dir}/images"))

    if "current_image_idx" not in st.session_state:
        st.session_state.current_image_idx = 0

    img_idx = st.session_state.current_image_idx
    if img_idx >= len(pseudo_images):
        st.success("Semua gambar telah selesai dikoreksi.")
        data.manual_corrections_mode = 1
        st.session_state.current_image_idx = 0
        # st.session_state.on_addition = True
        # st.session_state.on_correction = False
        st.rerun()
        return
    
    img_name = pseudo_images[img_idx]
    img_path = f"{pseudo_dir}/images/{img_name}"
    label_path = f"{pseudo_dir}/labels/{img_name.replace('.jpg', '.txt')}"
    st.info(f"Saat ini sedang di {img_idx} / {len(pseudo_images)}")
    
    if not os.path.exists(label_path) or os.path.getsize(label_path) == 0:
        reject_image(img_name, img_path, rejected_dir)
        push_action("N", img_name)
        st.session_state.current_image_idx += 1
        st.rerun()
    # ==== MODE UTAMA ====
    # if st.session_state.mode == "image":
    if data.deletion_display_mode == 0:
        image = Image.open(img_path).convert("RGB")
        image = draw_all_boxes(image, label_path)
        _, col, _ = st.columns([1, 2, 1])
        with col:
            st.image(image, caption=f"{img_name} ({img_idx+1}/{len(pseudo_images)})")

        # === LOCKED LAYOUT ZONE ===
        layout_container = st.container()

        with layout_container:
            # Baris 1: tombol utama
            space, c1, c2, c3, space = st.columns([0.1, 1, 1, 1, 0.1], gap="medium")
            with c1:
                if st.button("💾 Save", key="save_btn"):
                    save_image(img_name, img_path, label_path, dataset_dir)
                    push_action("Y", img_name)
                    st.session_state.current_image_idx += 1
                    st.rerun()
            with c2:
                if st.button("⏭️ Skip", key="skip_btn"):
                    reject_image(img_name, img_path, rejected_dir)
                    push_action("N", img_name)
                    data.deletion_count += 1
                    st.session_state.current_image_idx += 1
                    st.rerun()
            with c3:
                if st.button("🔍 Per Bounding Box", key="perbox_btn"):
                    # st.session_state.mode = "per_box"
                    data.deletion_display_mode = 1
                    st.session_state.current_image_name = img_name
                    st.session_state.box_idx = 0
                    st.session_state.boxes_saved = []
                    st.rerun()

            # Baris 2: input fast forward
            # fast_zone = st.empty()
            with st.container():
                fastinput = st.columns([1])[0]
                # default_value = 10 if "fast_num" not in st.session_state else st.session_state.fast_num
                with fastinput:
                    data.deletion_skip_num = st.number_input("Jumlah Fast forward", min_value=1,
                                    step=1, key="fast_num", value=data.deletion_skip_num)
                save_count = skip_count = back_count = data.deletion_skip_num

            # Baris 3: tombol batch action
            with st.container():
                space, c4, c5, c6, space = st.columns([0.1, 1, 1, 1, 0.1], gap="medium")
                with c4:
                    if st.button(f"💾 Save {save_count}", key="save10_btn"):
                        for _ in range(save_count):
                            if st.session_state.current_image_idx >= len(pseudo_images): break
                            img_n = pseudo_images[st.session_state.current_image_idx]
                            ip = f"{pseudo_dir}/images/{img_n}"
                            lp = f"{pseudo_dir}/labels/{img_n.replace('.jpg', '.txt')}"
                            if os.path.exists(lp) and os.path.getsize(lp) > 0:
                                shutil.copy(ip, f"{dataset_dir}/images/{img_n}")
                                shutil.copy(lp, f"{dataset_dir}/labels/{img_n.replace('.jpg', '.txt')}")
                                st.session_state.action_stack.append(("Y", img_n))
                            else:
                                shutil.copy(ip, f"{rejected_dir}/{img_n}")
                                st.session_state.action_stack.append(("N", img_n))
                            st.session_state.current_image_idx += 1
                        st.rerun()

                with c5:
                    if st.button(f"⏭️ Skip {skip_count}", key="skip10_btn"):
                        data.deletion_count += skip_count
                        for _ in range(skip_count):
                            if st.session_state.current_image_idx >= len(pseudo_images): break
                            img_n = pseudo_images[st.session_state.current_image_idx]
                            ip = f"{pseudo_dir}/images/{img_n}"
                            shutil.copy(ip, f"{rejected_dir}/{img_n}")
                            st.session_state.action_stack.append(("N", img_n))
                            st.session_state.current_image_idx += 1
                        st.rerun()

                with c6:
                    if st.button(f"⬅️ Back {back_count}", key="back10_btn"):
                        back_n_frames(back_count, dataset_dir, rejected_dir)
                        st.session_state.current_image_idx = max(0, st.session_state.current_image_idx - back_count)
                        st.rerun()

    # ==== MODE PER-BOUNDING-BOX ====
    # elif st.session_state.mode == "per_box":
    elif data.deletion_display_mode == 1:
        img_name = st.session_state.current_image_name
        img_path = f"{pseudo_dir}/images/{img_name}"
        label_path = f"{pseudo_dir}/labels/{img_name.replace('.jpg', '.txt')}"

        with open(label_path, "r") as f:
            boxes = [b.strip() for b in f.readlines()]

        box_idx = st.session_state.box_idx


        if box_idx >= len(boxes):
            # selesai semua bounding box → tulis label baru berisi yang disimpan
            # BUAT SKIP KALAU GAADA BB
            label_save_path = f"{dataset_dir}/labels/{img_name.replace('.jpg', '.txt')}"
            with open(label_save_path, "w") as lf:
                for idx in st.session_state.boxes_saved:
                    lf.write(boxes[idx] + "\n")

            # salin image
            shutil.copy(img_path, f"{dataset_dir}/images/{img_name}")

            # reset mode ke image
            push_action("Y", img_name)
            # st.session_state.mode = "image"
            data.deletion_display_mode = 0
            st.session_state.current_image_idx += 1
            st.rerun()

        else:
            image = Image.open(img_path).convert("RGB")
            box_line = boxes[box_idx]
            image = draw_single_box(image, box_line)
            space, col, space = st.columns([1, 2, 1])
            with col:  
                st.image(image, caption=f"Box {box_idx+1}/{len(boxes)} on {img_name}")
            # tampilkan bounding box ke box_idx
            # draw_box(image, boxes[box_idx])
            # st.image(image, caption=f"Box {box_idx+1}/{len(boxes)}")

            # save_box, delete_box = st.columns(2)
            st.markdown("""
                <style>
                .block-container {
                    padding-top: 1rem;
                    padding-bottom: 1rem;
                }
                div[data-testid="stHorizontalBlock"] > div {
                    gap: 0.25rem !important;  /* jarak antar kolom */
                }
                </style>
            """, unsafe_allow_html=True)
            save_box, delete_box = st.columns([1, 1])  # default sama lebar, masih agak renggang
            # coba ganti jadi:
            spacer, save_box, delete_box = st.columns([0.1, 1, 1])  # tambah spacer kecil

            with save_box:
                if st.button("💾 Save Box"):
                    st.session_state.boxes_saved.append(box_idx)
                    st.session_state.box_idx += 1
                    st.rerun()

            with delete_box:
                if st.button("🗑️ Delete Box"):
                    # st.session_state.boxes_deleted.append(box_idx)
                    st.session_state.box_idx += 1
                    data.deletion_count += 1
                    st.rerun()

            # with next_box:
            #     if st.button("➡️ Next Box"):
            #         st.session_state.box_idx += 1
            #         st.rerun()
