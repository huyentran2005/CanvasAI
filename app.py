import streamlit as st
import requests
import os
from components import header, auth, theme, main_canvas, bottom_bar
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

if not user:
    theme.set_background()
else:
    col = []
    col = st.columns([1,3.5,1.5],gap ='medium')
    with col[0]:
        pass
    with col[1]:
        pass
    with col[2]:
        pass

    bottom_bar.display_bottom_bar()