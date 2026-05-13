import streamlit as st
from services import * 
from PIL import Image
from streamlit_image_zoom import image_zoom
import io


def display_img():
    embed_css("main_canvas.css")
    image = st.session_state.edited_img
    
    with st.container(key='main_canvas'):
        if image is not None:
            st.image(image)
        else:
            temp =  Image.open("assets/favicon/empty.jpg")
            st.image(temp)
        