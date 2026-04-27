import streamlit as st
import requests
import os
from components import header 
from components.auth import *


# set favicon
st.set_page_config(
    page_title = "CanvasAI",
    page_icon ="assets/favicon/painter.png",
    layout = "wide"
)

header.show_header()

st.title("Demo Hệ thống Xử lý Thông minh")

x = st.number_input("Nhập số", value=1)

if st.button("Tính"):
    pass

if process_login_callback():
    st.rerun()
user = get_user()
if user:
    name = get_display_name(user)
    st.success(f"Xin chào {name} 👋")
# Giao diện hiển thị
if user:
    st.success(f"Đã đăng nhập thành công! Email: {user.email}")
    if st.button("Đăng xuất"):
        supabase.auth.sign_out()
        st.rerun()
else:
    st.title("Hệ thống CanvasAI")
    st.warning("Bạn chưa đăng nhập.")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Đăng nhập bằng Google"):
            login("google")
    with col2:
        if st.button("Đăng nhập bằng Facebook"):
            login("facebook")