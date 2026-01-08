import streamlit as st
import os
import shutil
from ultralytics import YOLO
from data import data
from core.logic.input import inputs
from core.logic.frame_extraction import extract_frames, extract_frames_random
from core.logic.auto_anot import automatic_annotationation
from core.utils import show_auto_anotation, show_iteration_info, show_training_confirmation, show_evaluation_summary, simpan_evaluasi_otomatisasi, add_iteration, show_correction_confirmation
def show():
    st.info("INI IBBA")

def main():
    if data.iteration > data.max_iteration:
        st.session_state.ibba_stage = "idle"
    if st.session_state.ibba_stage == "idle":
        # show_iteration_info()
        st.info("SELESAI!")
        return

    if st.session_state.ibba_stage == "inputs":
        inputs()
        return

    show_iteration_info()

    if st.session_state.ibba_stage == "frame extraction":
        if data.is_random:
            extract_frames_random(data.video_path, data.frame_dir)
        else:
            extract_frames(data.video_path, data.frame_dir)
    
    if st.session_state.ibba_stage == "automatic annotation":
        # data.is_done_train = False
        show_auto_anotation()
        if not data.is_done_auto_annot:
            default_is_include_reject = data.is_include_reject_in_auto_annot
            data.is_include_reject_in_auto_annot = st.checkbox("Anotasi Reject:", value=default_is_include_reject)
            if st.button("Skip", key="btn-skip-automatic-anotation"):
                st.session_state.ibba_stage = "manual correction"
                st.rerun()
            if st.button("Auto Annotation", key="btn-auto-anotation"):
                reject_dir = f"{data.working_dir}/{data.iteration-1}-iter/reject"
                out_dir = f"{data.working_dir}/{data.iteration}-iter/pseudo"
                automatic_annotationation(
                    model_path=data.model_path,
                    frame_path=data.frame_dir,
                    reject_path=reject_dir,
                    out_dir=out_dir,
                    conf_thresh=0.5,
                    max_iter=data.max_iteration
                )
        else:
            st.session_state.ibba_stage = "manual correction"
        
    if st.session_state.ibba_stage == "manual correction":
        show_correction_confirmation()
        if not data.is_done_correction:
            if st.button("Skip", key="btn-skip-manual-correction"):
                st.session_state.ibba_stage = "training"
                st.rerun()
            if not data.is_on_correction:
                if st.button("Start Correction", key="btn-start-manual-correction"):
                    data.is_on_correction = True
                    st.rerun()
            else:
                if data.manual_corrections_mode == 0:
                    from core.logic.corrections.deletions import manual_correction
                    manual_correction()
                else:
                    from core.logic.corrections.addition import manual_addition
                    manual_addition()
        else:
            st.session_state.ibba_stage = "training"
    if st.session_state.ibba_stage == "training":
        show_training_confirmation()
        if not data.is_done_train:
            if st.button("Mulai Training"):
                from core.logic.train import train
                train()
                data.is_done_train = True
        if not data.is_done_train:
            if st.button("Skip Training"):
                yaml_path = f"{data.working_dir}/{data.iteration}-iter/datatrain_iter{data.iteration}/data.yaml"
                CLASS_NAME = "pelat"
                run_name = f"model{data.iteration}"
                project_path = f"{data.working_dir}/{data.iteration}-iter"
                new_model = YOLO(f"{data.working_dir}/{data.iteration}-iter/{run_name}/weights/best.pt")
                
                # === Evaluasi model (tampilkan gambar + metrik) ===
                precision, recall, map50, map5095, est_deletion, est_addition = show_evaluation_summary(data.working_dir, data.iteration, new_model, yaml_path, CLASS_NAME, run_name)
                simpan_evaluasi_otomatisasi(model_path=f"{project_path}/{run_name}/weights/best.pt", precision=precision, recall=recall, map50=map50, map5095=map5095, iteration=data.iteration, ril_deletion=data.deletion_count, ril_addition=data.addition_count, est_deletion=est_deletion, est_addition=est_addition)
                data.is_done_train = True
            
        if data.is_done_train:
            if st.button("Next (lanjutkan)"):
                st.session_state.ibba_stage = "automatic annotation"
                from core.utils import add_iteration
                add_iteration()
                st.rerun()