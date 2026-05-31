import streamlit as st
import requests
import os
from components import header, auth, theme, main_canvas, bottom_bar,adjust_panel, ai_panel
from services import *

# set favicon
st.set_page_config(
    page_title = "CanvasAI",
    page_icon ="assets/favicon/painter.png",
    layout = "wide"
)


init_default_state()
header.show_header()

user = auth.get_user()

if user:
    theme.set_background()
else:
    col = []
    col = st.columns([1,3.5,1.5],gap ='medium')
    with col[0]:
        adjust_panel.display_left_panel()
    with col[1]:
        main_canvas.display_img()
    with col[2]:
        ai_panel.display_right_panel()
    
    bottom_bar.display_bottom_bar()