from PIL import Image, ImageDraw, ImageFont
import os
import shutil
import streamlit as st


def push_action(action, img_name):
    """Simpan aksi ke stack untuk fitur back 10 frame"""
    if "action_stack" not in st.session_state:
        st.session_state.action_stack = []
    st.session_state.action_stack.append((action, img_name))

def save_image(img_name, img_path, label_path, dataset_dir):
    """Simpan image dan label ke dataset"""
    dst_img = f"{dataset_dir}/images/{img_name}"
    dst_lbl = f"{dataset_dir}/labels/{img_name.replace('.jpg', '.txt')}"
    
    os.makedirs(os.path.dirname(dst_img), exist_ok=True)
    os.makedirs(os.path.dirname(dst_lbl), exist_ok=True)
    
    # Salin file image dan label (jika label ada)
    shutil.copy2(img_path, dst_img)
    if os.path.exists(label_path):
        shutil.copy2(label_path, dst_lbl)

def reject_image(img_name, img_path, rejected_dir):
    """Tolak image dengan memindahkannya ke rejected_dir"""
    dst_path = f"{rejected_dir}/{img_name}"
    os.makedirs(rejected_dir, exist_ok=True)
    shutil.copy2(img_path, dst_path)

    # label_path = img_path.replace("/images/", "/labels/").replace(".jpg", ".txt")
    # if os.path.exists(label_path):
    #     label_dst = f"{rejected_dir}/{img_name.replace('.jpg', '.txt')}"
    #     shutil.copy2(label_path, label_dst)

def draw_single_box(image, box_line, scale = 0.4):
    image = image.copy()
    parts = box_line.strip().split()
    if len(parts) != 5:
        return image
    cls_id, x_c, y_c, w, h = map(float, parts)
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("arial.ttf", size=24)
    except:
        font = ImageFont.load_default()
    W, H = image.size
    x1 = (x_c - w/2) * W
    y1 = (y_c - h/2) * H
    x2 = (x_c + w/2) * W
    y2 = (y_c + h/2) * H
    draw.rectangle([x1, y1, x2, y2], outline=(255,69,0), width=6)
    draw.text((x1, max(0, y1-25)), f"Class {int(cls_id)}", fill=(255,69,0), font=font)
    if scale != 1.0:
        new_w = int(W * scale)
        new_h = int(H * scale)
        image = image.resize((new_w, new_h), Image.Resampling.LANCZOS)
    return image

def draw_all_boxes(image, label_path, scale = 0.4):
    image = image.copy()
    if not os.path.exists(label_path):
        return image
    with open(label_path, "r") as f:
        boxes = [b.strip().split() for b in f.readlines() if b.strip()]
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("arial.ttf", size=24)
    except:
        font = ImageFont.load_default()
    W, H = image.size
    for box in boxes:
        cls_id, x_c, y_c, w, h = map(float, box)
        x1 = (x_c - w/2) * W
        y1 = (y_c - h/2) * H
        x2 = (x_c + w/2) * W
        y2 = (y_c + h/2) * H
        draw.rectangle([x1, y1, x2, y2], outline=(173,255,47), width=5)
        draw.text((x1, max(0, y1-25)), f"Class {int(cls_id)}", fill=(173,255,47), font=font)
    
    # setelah menggambar semua, lakukan scaling jika < 1.0
    if scale != 1.0:
        new_w = int(W * scale)
        new_h = int(H * scale)
        image = image.resize((new_w, new_h), Image.Resampling.LANCZOS)
    return image

def back_n_frames(n, dataset_dir, rejected_dir):
    for _ in range(n):
        if not st.session_state.action_stack:
            break
        action, img_name = st.session_state.action_stack.pop()
        if action == "Y":
            for p in [
                f"{dataset_dir}/images/{img_name}",
                f"{dataset_dir}/labels/{img_name.replace('.jpg', '.txt')}"
            ]:
                if os.path.exists(p):
                    os.remove(p)
        elif action == "N":
            p = f"{rejected_dir}/{img_name}"
            if os.path.exists(p):
                os.remove(p)
        elif action == "y":
            for p in [
                f"{dataset_dir}/images/{img_name}",
                f"{dataset_dir}/labels/{img_name.replace('.jpg', '.txt')}"
            ]:
                shutil.copy2(f"{dataset_dir}/images/{img_name}",f"{rejected_dir}/{img_name}")
                if os.path.exists(p):
                    os.remove(p)
          