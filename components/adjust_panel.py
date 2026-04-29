import streamlit as st
from services import *
import os

def display_left_panel():
    embed_css("adjust_panel.css")
    with st.container(key ="left_panel"):
        st.subheader("TOOLS", anchor= False)
        st.file_uploader(
            label= "UPLOAD IMAGE",
            type = ["pnj","jpg","jpeg"],
            accept_multiple_files= False,
            label_visibility="collapsed"
        )

        action = st.selectbox(
            "Algorithm",
            ["Algorithm","HE","CLAHE","Retinex"],
            label_visibility= "collapsed"
        )
        
        if st.button("AI ANALYZE",use_container_width=True, key = 'btn_ai'):
            pass
        if st.button("REMOVE OBJECT",  icon="🧽", use_container_width=True, key = 'btn_remove'):
            pass
        if st.button("DRAW MASK", icon="🎭", use_container_width=True, key ='btn_mask'):
            pass
        if st.button("UNDO", icon="↩️", use_container_width=True, key ='btn_undo'):
            pass
        if st.button("REDO", icon="↪️", use_container_width=True, key ='btn_redo'):
            pass