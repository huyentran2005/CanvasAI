import streamlit as st
from services import *
from PIL import Image 
import os
import numpy as np
import cv2
import onnxruntime as ort
from module.enhance.classical_enhancement import (
    enhance_by_clahe,
    enhance_by_retinex,
    histogram_equal,
)
from module.enhance.enlightengan import enhance_with_enlightengan
from module.segmentation.intelligent_scissors import clear_scissor_state, undo_scissor_step, redo_scissor_step

session = ort.InferenceSession(
    "model/dncnn-onnx-float/dncnn.onnx",
    providers=["CPUExecutionProvider"]
)

model_denoise = ort.InferenceSession(
    "model/real_esrgan_x4plus-onnx-float/real_esrgan_x4plus.onnx",
    providers= ["CPUExecutionProvider"]
)


def enhance_image(image):
    return enhance_with_enlightengan(image)

def enhance_image_by_algorithm(image, algorithm):
    image = image.convert("RGB")
    img_rgb = np.array(image)
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

    if algorithm == "HE":
        result_bgr, _, _ = histogram_equal(img_bgr)
    elif algorithm == "CLAHE":
        result_bgr, _, _ = enhance_by_clahe(img_bgr)
    elif algorithm == "Retinex":
        result_bgr = enhance_by_retinex(img_bgr, method="msrcr")
    elif algorithm == "EnlightenGAN":
        return enhance_with_enlightengan(image)
    else:
        return image

    result_rgb = cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(result_rgb)

def get_analysis_score(items, name):
    for item in items:
        if item.get("name") == name:
            try:
                return float(item.get("score"))
            except (TypeError, ValueError):
                return None
    return None

def auto_fix_by_analysis(image):
    analysis_results = st.session_state.get("analysis_results", [])
    brightness_score = get_analysis_score(analysis_results, "Cân bằng sáng")
    purity_score = get_analysis_score(analysis_results, "Độ trong của ảnh")

    if brightness_score is None or purity_score is None:
        return None, None

    if brightness_score <= purity_score:
        return enhance_image(image), "DeepEnhance"

    return denoise_image_v2(image), "DeepDenoise"


def denoise_image_v1(image):
    original_size = image.size 

    ycbcr = image.convert('YCbCr')
    y, cb, cr = ycbcr.split()

    y_resized = y.resize((256, 256))
    
    img_y = np.array(y_resized).astype(np.float32) / 255.0
    
    img_y = img_y.reshape(1, 1, 256, 256)

    input_name = session.get_inputs()[0].name
    output_y = session.run(None, {input_name: img_y})[0]

    output_y = np.squeeze(output_y)
    output_y = np.clip(output_y, 0, 1)
    output_y = (output_y * 255).astype(np.uint8)

    denoised_y_img = Image.fromarray(output_y).resize(original_size)

    final_ycbcr = Image.merge('YCbCr', (denoised_y_img, cb, cr))

    return final_ycbcr.convert('RGB')

def denoise_image_v2(image): 
    original_size = image.size
    image = image.convert("RGB")   
    # PIL -> array
    image = image.resize((128,128))
    img = np.array(image).astype(np.float32)
    img = img/255.0
    # HWC -> CHW
    img = np.transpose(img,(2,0,1))
    img = np.expand_dims(img, axis= 0).astype(np.float32)
    info = model_denoise.get_inputs()[0].name
    output = model_denoise.run(None,{info: img})[0]
    output = np.squeeze(output, axis= 0)
    output = np.transpose(output, (1,2,0))
    output = np.clip(output, 0,1)
    output = (output * 255).astype(np.uint8)
    return Image.fromarray(output).resize(original_size)

def ensure_history_state():
    if "undo_stack" not in st.session_state:
        st.session_state.undo_stack = []
    if "redo_stack" not in st.session_state:
        st.session_state.redo_stack = []
    if "last_upload_signature" not in st.session_state:
        st.session_state.last_upload_signature = None
    if "last_applied_algorithm" not in st.session_state:
        st.session_state.last_applied_algorithm = "Algorithm"

def apply_edit(next_image):
    if st.session_state.edited_img is not None:
        st.session_state.undo_stack.append(st.session_state.edited_img.copy())
    st.session_state.edited_img = next_image
    st.session_state.redo_stack.clear()

def undo_edit():
    if not st.session_state.undo_stack or st.session_state.edited_img is None:
        return
    st.session_state.redo_stack.append(st.session_state.edited_img.copy())
    st.session_state.edited_img = st.session_state.undo_stack.pop()

def redo_edit():
    if not st.session_state.redo_stack or st.session_state.edited_img is None:
        return
    st.session_state.undo_stack.append(st.session_state.edited_img.copy())
    st.session_state.edited_img = st.session_state.redo_stack.pop()

def display_left_panel():
    ensure_history_state()
    embed_css("adjust_panel.css")
    with st.container(key ="left_panel"):
        st.subheader("TOOLS", anchor= False)

        img = st.file_uploader(
            label= "UPLOAD IMAGE",
            type = ["png","jpg","jpeg"],
            accept_multiple_files= False,
            label_visibility="collapsed"
        )

        if img is not None:
            upload_signature = (img.name, img.size)
            uploaded_img = Image.open(img).copy()

            if st.session_state.last_upload_signature != upload_signature:
                st.session_state.original_img = uploaded_img
                st.session_state.edited_img = uploaded_img.copy()
                st.session_state.undo_stack.clear()
                st.session_state.redo_stack.clear()
                st.session_state.last_upload_signature = upload_signature
                st.session_state.last_applied_algorithm = "Algorithm"
                st.session_state.analysis_results = []
                clear_scissor_state()
                st.session_state.scissor_active = False

        if st.button("AUTO FIX",use_container_width=True, key = 'btn_ai'):
            if st.session_state.edited_img is not None:
                fixed_img, _ = auto_fix_by_analysis(st.session_state.edited_img)
                if fixed_img is None:
                    pass
                else:
                    apply_edit(fixed_img)
        if st.button("REMOVE OBJECT",  icon="🧽", use_container_width=True, key = 'btn_remove'):
            if (
                st.session_state.edited_img is not None
                and st.session_state.mask is not None
            ):
                from module.inpainting.Lama import run_inpaint

                result = run_inpaint(
                    st.session_state.edited_img,
                    st.session_state.mask
                )

                st.session_state.edited_img = result
                st.session_state.mask = None
                st.session_state.masked_img = None
                st.session_state.scissor_points = []
                st.session_state.scissor_segments = []
                st.session_state.scissor_last_click = None
                st.session_state.scissor_undo_stack = []
                st.session_state.scissor_redo_stack = []
                st.session_state.scissor_active = False
                st.session_state.scissor_completed = False
                st.session_state.text_prompt = None
                st.rerun()
                
        if st.button("DRAW MASK", icon="🎭", use_container_width=True, key ='btn_mask'):
            should_rerun = True
            prompt = (st.session_state.get("text_prompt") or "").strip()
            if st.session_state.edited_img is not None and prompt:
                from module.segmentation.text_remove import grounding_sam_pipeline

            
                mask = grounding_sam_pipeline(st.session_state.edited_img, prompt)

                clear_scissor_state()
                st.session_state.scissor_active = False
                if mask is None:
                    should_rerun = False
                else:
                    st.session_state.mask = mask
                    st.session_state.masked_img = Image.fromarray(mask)
                    st.session_state.scissor_completed = True
            else:
                st.session_state.scissor_active = True
                st.session_state.scissor_completed = False

        action = st.selectbox(
            "Algorithm",
            ["Algorithm","EnlightenGAN","HE","CLAHE","Retinex"],
            label_visibility= "collapsed"
        )
        if action == "Algorithm":
            st.session_state.last_applied_algorithm = "Algorithm"
        elif (
            st.session_state.edited_img is not None
            and st.session_state.last_applied_algorithm != action
        ):
            apply_edit(enhance_image_by_algorithm(st.session_state.edited_img, action))
            st.session_state.last_applied_algorithm = action

        if st.button("UNDO", icon="↩️", use_container_width=True, key ='btn_undo'):
            if st.session_state.scissor_active:
                undo_scissor_step()
            else:
                undo_edit()
            st.rerun()
        if st.button("REDO", icon="↪️", use_container_width=True, key ='btn_redo'):
            if st.session_state.scissor_active:
                redo_scissor_step()
            else:
                redo_edit()
            st.rerun()
