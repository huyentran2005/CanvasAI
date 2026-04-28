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
