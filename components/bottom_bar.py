import streamlit as st
import io
from services import *

def display_bottom_bar(btn = ["⟲ Reset ","EXPORT PNG","EXPORT JPG"]):
    embed_css("bottom_bar.css")
    with st.container(key="bottom_bar_section"):
        image = st.session_state.edited_img

        col = []
        col = st.columns([5,0.8,0.8,0.8])
        with col[1]:
            if st.button(btn[0]):
                if st.session_state.get("original_img") is not None:
                    st.session_state.edited_img = st.session_state.original_img.copy()
                    st.rerun()
                else:
                    pass

        with col[2]:
            if image is None:
                st.button(btn[1], disabled= True)
            else:
                buf = io.BytesIO()
                image.save(buf, format='PNG')
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
                image.convert("RGB").save(buf, format='JPEG')
                buf.seek(0)
                st.download_button(
                    label= btn[2],
                    data = buf,
                    file_name= "image.jpg",
                    mime= "image/jpeg"
                )
