import streamlit as st
import os

def show_header():
    css_path = os.path.join(os.path.dirname(__file__),"..","assets","css","header.css")
    if os.path.exists(css_path):
        with open(css_path) as f:
            st.markdown(f"<style>{f.read()}</style>",unsafe_allow_html= True)
            f.close()
            
    st.markdown("""
        <div class = "nav-container">
            <div class="logo"> CanvasAI </div>
            <div class ="nav_link"> 
                <a href ="#"> Hướng dẫn</a> 
                <a href = "#">Đăng nhập</a>
            </div>
        </div>
    """, unsafe_allow_html= True)