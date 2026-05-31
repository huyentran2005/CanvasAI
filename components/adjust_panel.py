import streamlit as st
from services import *
from PIL import Image 
import os
import numpy as np
from enlighten_inference import EnlightenOnnxModel
import cv2
import onnxruntime as ort


session = ort.InferenceSession(
    "model/dncnn-onnx-float/dncnn.onnx",
    providers= ["CPUExecutionProvider"]
)

model_denoise = ort.InferenceSession(
    "model/real_esrgan_x4plus-onnx-float/real_esrgan_x4plus.onnx",
    providers= ["CPUExecutionProvider"]
)


@st.cache_resource
def load_model():
    return  EnlightenOnnxModel(providers=["CPUExecutionProvider"])

def enhance_image(image):
    model = load_model()

    image = image.convert("RGB")
    img_np = np.array(image)
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    resuilt = model.predict(img_bgr)
    resuilt = cv2.cvtColor(resuilt,cv2.COLOR_BGR2RGB)
    return Image.fromarray(resuilt)


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

        if st.button("AUTO FIX",use_container_width=True, key = 'btn_ai'):
            if st.session_state.edited_img is not None:
                apply_edit(denoise_image_v2(st.session_state.edited_img))
        if st.button("REMOVE OBJECT",  icon="🧽", use_container_width=True, key = 'btn_remove'):
            pass
        if st.button("DRAW MASK", icon="🎭", use_container_width=True, key ='btn_mask'):
            pass

        action = st.selectbox(
            "Algorithm",
            ["Algorithm","HE","CLAHE","Retinex"],
            label_visibility= "collapsed"
        )

        if st.button("UNDO", icon="↩️", use_container_width=True, key ='btn_undo'):
            undo_edit()
            st.rerun()
        if st.button("REDO", icon="↪️", use_container_width=True, key ='btn_redo'):
            redo_edit()
            st.rerun()
