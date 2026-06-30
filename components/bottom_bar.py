import streamlit as st
import io
from services import *
from module.segmentation.intelligent_scissors import clear_scissor_state

def display_bottom_bar(btn = ["⟲ Reset ","EXPORT PNG","EXPORT JPG"]):
    embed_css("bottom_bar.css")
    with st.container(key="bottom_bar_section"):
        image = st.session_state.edited_img

        col = []
        col = st.columns([5,0.8,0.8,0.8])
        with col[0]:
            put, nofi = st.columns([1.5,4])
            with put:
                text_prompt = st.text_input(
                    "Input",
                    value=st.session_state.get("text_prompt") or "",
                    label_visibility="collapsed"
                )
                st.session_state.text_prompt = text_prompt.strip()
        with col[1]:
            if st.button(btn[0]):
                if st.session_state.get("original_img") is not None:
                    st.session_state.edited_img = st.session_state.original_img.copy()
                    clear_scissor_state()
                    st.session_state.scissor_active = False

        with col[2]:
            if image is None:
                st.button(btn[1], disabled= True)
            else:
                buf = io.BytesIO()
                image.save(buf, format="PNG", compress_level=0)
                buf.seek(0)
                st.download_button(
                    label= btn[1],
                    data = buf,
                    file_name= "image.png",
                    mime= "image/png"
                )

        with col[3]:
            if image is None:
                st.button(btn[2], disabled= True)
            else:
                buf = io.BytesIO()
                image.convert("RGB").save(
                    buf,
                    format="JPEG",
                    quality=100,
                    subsampling=0,
                    optimize=True,
                )
                buf.seek(0)
                st.download_button(
                    label= btn[2],
                    data = buf,
                    file_name= "image.jpg",
                    mime= "image/jpeg"
                )
