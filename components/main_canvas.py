import streamlit as st
from services import * 
from PIL import Image
from streamlit_image_zoom import image_zoom
import io
from module.segmentation.intelligent_scissors import (
    draw_mask,
    ensure_scissor_state,
    fit_image_for_display,
)


@st.fragment
def display_img():
    embed_css("main_canvas.css")
    image = st.session_state.edited_img
    
    with st.container(key='main_canvas'):
        if image is not None:
            ensure_scissor_state()
            if st.session_state.scissor_active:
                draw_mask()
            else:
                display_img, _ = fit_image_for_display(image)
                st.image(display_img)
        else:
            temp =  Image.open("assets/favicon/empty.jpg")
            st.image(temp)
        
