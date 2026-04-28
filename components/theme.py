import streamlit as st
from .auth import *
from services import *
import os


def set_background():
    img_path = os.path.join(os.path.dirname(__file__),"..","assets","favicon","wallhaven.png")
    if os.path.exists(img_path):
        img_base64 = get_base64(img_path)
        st.markdown(f"""
            <style>
                [data-testid="stAppViewContainer"] {{
                    background-image: 
                        linear-gradient( 
                            rgba(0, 0, 0, 0.45),
                            rgba(0, 0, 0, 0.45) 
                        ),
                        url("data:image/png;base64,{img_base64}"); 
                    background-size: cover;
                    background-position: center;
                    background-repeat: no-repeat;
                    background-attachment: fixed; 
                }} 
            </style>
        """, unsafe_allow_html= True)
    embed_css("theme.css")
    st.title("Create Stunning Images with Just a Prompt",anchor = False)
    