import streamlit as st
import os, shutil, math
from data import data
from tab import train_awal, ibba, eval

# Fungsi untuk memuat CSS eksternal
def load_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

if "loaded" not in st.session_state:
    data.reset_data()
    mkdir_path = os.path.join(os.path.dirname(os.getcwd()), "self-training")
    if not os.path.exists(mkdir_path):
        os.makedirs(mkdir_path)
    st.session_state.loaded = True
# Muat file CSS
load_css("app.css")

# Set title
st.title("Aplikasi Deteksi Objek dengan Iterative Bounding Box Annotation (IBBA)")

# Inisialisasi session_state untuk menyimpan tab aktif
if "active_tab" not in st.session_state:
    st.session_state.active_tab = "Training Awal"
if "on_correction" not in st.session_state:
    st.session_state.on_correction = "false"
if "ibba_stage" not in st.session_state:
    st.session_state.ibba_stage = "inputs"


# Fungsi untuk mengubah tab aktif
def change_tab(tab_name):
    st.session_state.active_tab = tab_name

# Tombol navigasi tab
space, col1, col2, col3, space = st.columns([0.1, 1, 1, 1, 0.1])
with col1:
    if st.button("Training Awal", key="btn_training"):
        change_tab("Training Awal")
with col2:
    if st.button("IBBA", key="btn_ibba"):
        change_tab("IBBA")
with col3:
    if st.button("Est Corrections", key="btn_testing"):
        change_tab("Eval")
    
st.markdown("---")
# Tampilkan konten sesuai tab aktif
if st.session_state.active_tab == "Training Awal":
    train_awal.main()

if st.session_state.active_tab == "IBBA":
    ibba.main()

if st.session_state.active_tab == "Eval":
    if data.done_IBBA == False:
        st.warning("Silakan selesaikan proses IBBA terlebih dahulu sebelum melakukan Testing and Predicting.")
    else:
        eval.evaluate_estimation()