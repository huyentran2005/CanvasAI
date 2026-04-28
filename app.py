import streamlit as st
import requests
import os
from components import header 


# set favicon
st.set_page_config(
    page_title = "CanvasAI",
    page_icon ="assets/favicon/painter.png",
    layout = "wide"
)

header.show_header()

