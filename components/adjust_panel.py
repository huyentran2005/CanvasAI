import streamlit as st
from services import *
from PIL import Image 
import os
import numpy as np
import tensorflow as tf
import cv2
import onnxruntime as ort
from module.enhance.classical_enhancement import (
    enhance_by_clahe,
    enhance_by_retinex,
    histogram_equal,
)

TF_MODEL_DIR = "model/low_light_image_enhancement"

session = ort.InferenceSession(
    "model/dncnn-onnx-float/dncnn.onnx",
    providers=["CPUExecutionProvider"]
)

model_denoise = ort.InferenceSession(
    "model/real_esrgan_x4plus-onnx-float/real_esrgan_x4plus.onnx",
    providers= ["CPUExecutionProvider"]
)


@st.cache_resource
def load_low_light_model():
    return tf.saved_model.load(TF_MODEL_DIR)


def enhance_image(image):
    model = load_low_light_model()
    image = image.convert("RGB")
    original_img = np.array(image).astype(np.float32) / 255.0
    
    # Resize to model input size for curve estimation
    h, w = original_img.shape[:2]
    img_resized = cv2.resize(original_img, (400, 400))
    
    img_tensor = tf.convert_to_tensor(np.expand_dims(img_resized, axis=0), dtype=tf.float32)
    signature = model.signatures["serving_default"]
    output = signature(img_tensor)
    curves = next(iter(output.values()))[0].numpy()  # Shape: (400, 400, 24)
    
    # Resize curves back to original image size
    curves_resized = cv2.resize(curves, (w, h))
    
    # Apply curves to original image - Zero-DCE with controlled enhancement strength
    # Scale curves to prevent oversaturation (divide by 4 for gentle enhancement)
    enhanced = original_img.copy()
    
    for i in range(8):
        curve_i = curves_resized[..., i * 3 : (i + 1) * 3].mean(axis=2, keepdims=True) / 4.0
        # Apply gentle exponential adjustment
        enhanced = enhanced * np.exp(-curve_i)
    
    enhanced = np.clip(enhanced, 0.0, 1.0)
    enhanced = (enhanced * 255.0).astype(np.uint8)
    return Image.fromarray(enhanced)

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

        if st.button("AUTO FIX",use_container_width=True, key = 'btn_ai'):
            if st.session_state.edited_img is not None:
                fixed_img, _ = auto_fix_by_analysis(st.session_state.edited_img)
                if fixed_img is None:
                    st.warning("Hãy phân tích ảnh bằng AI trước khi dùng AUTO FIX.")
                else:
                    apply_edit(fixed_img)
                    st.rerun()
        if st.button("REMOVE OBJECT",  icon="🧽", use_container_width=True, key = 'btn_remove'):
            pass
        if st.button("DRAW MASK", icon="🎭", use_container_width=True, key ='btn_mask'):
            pass

        action = st.selectbox(
            "Algorithm",
            ["Algorithm","HE","CLAHE","Retinex"],
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
            st.rerun()

        if st.button("UNDO", icon="↩️", use_container_width=True, key ='btn_undo'):
            undo_edit()
            st.rerun()
        if st.button("REDO", icon="↪️", use_container_width=True, key ='btn_redo'):
            redo_edit()
            st.rerun()
