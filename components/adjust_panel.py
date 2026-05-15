import streamlit as st
from services import *
from PIL import Image 
import os

def display_left_panel():
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
            img = Image.open(img).copy()

            st.session_state.original_img = img
            st.session_state.edited_img = img.copy()

        if st.button("AUTO FIX",use_container_width=True, key = 'btn_ai'):
            pass
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
            pass
        if st.button("REDO", icon="↪️", use_container_width=True, key ='btn_redo'):
            pass