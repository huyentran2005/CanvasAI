import os
import base64
import streamlit as st

def embed_css(name =""):
    css_path = os.path.join(os.path.dirname(__file__),"assets","css",name)
    if os.path.exists(css_path):
        with open(css_path) as f:
            st.markdown(f"<style>{f.read()}</style>",unsafe_allow_html= True)
            f.close()


def get_base64(img_path):
    with open(img_path,"rb") as f:
        return base64.b64encode(f.read()).decode()


def init_default_state():
    defaults = {
        "edited_img": None,
        "original_img": None,
        "user": None,
        "masked_img": None,
        "mask": None,
        "analysis": False,
        "analysis_results": [],
        "object":None,
        "undo_stack": [],
        "redo_stack": [],
        "last_upload_signature": None,
        "scissor_points": [],
        "scissor_segments": [],
        "scissor_last_click": None,
        "scissor_undo_stack": [],
        "scissor_redo_stack": [],
        "scissor_active": False,
        "scissor_completed": False,
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value
