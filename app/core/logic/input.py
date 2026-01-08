from data import data
import streamlit as st
import os
def inputs():
    data.working_dir = st.text_input("Masukkan Path folder hasil iterasi :", "C:/Users/USER/Documents/a skripsi/workv1/self-training")
    # data.video_path = st.text_input("Masukkan Path video :", "C:/Users/USER/Documents/a skripsi/data/data2.mp4")
    data.video_path = st.text_input("Masukkan Path video :", "C:/Users/USER/Documents/a skripsi/riset/data_video_train/data2.mp4")
    data.model_path = st.text_input("Masukkan Path model awal :", "C:/Users/USER/Documents/a skripsi/riset/model_awal/best.pt")
    data.eval_path = st.text_input("Masukkan Path dataset evaluasi :",  "C:/Users/USER/Documents/a skripsi/riset/data_eval_final/test generalisasi")
    data.max_iteration = st.number_input(
        "Masukkan Max Iteration :",
        min_value=1,
        value=data.max_iteration,
        step=1,
        format="%d"
    )
    data.epochs_per_iteration = st.number_input(
        "Masukkan Epoch Setiap Iterasi :",
        min_value=1,
        value=data.epochs_per_iteration,
        step=1,
        format="%d"
    )
    # Toggle untuk mengaktifkan/menonaktifkan randomisasi data
    default_is_random = data.is_random
    data.is_random = st.checkbox("Acak data (is_random):", value=default_is_random)
    if st.button("Extract", key="extract-btn"):
        if os.path.exists(data.video_path):
            data.frame_dir = f"{data.working_dir}/frames_iter"
            st.session_state.ibba_stage = "frame extraction"
            data.is_done_extract = False
            st.rerun()
        else:
            st.warning("Video tidak ada")
    if st.button("Skip", key="skip-extract-btn"):
        data.frame_dir = f"{data.working_dir}/frames_iter"
        st.session_state.ibba_stage = "automatic annotation"
        # data.iteration = 5
        st.rerun()