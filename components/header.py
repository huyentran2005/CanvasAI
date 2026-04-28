import streamlit as st
from .auth import *
import os

@st.dialog('Đăng nhập')
def show_modal( msg = ['Google','Facebook']):
    if st.button(msg[0], key ='google', use_container_width= True):
        login('google')
    if st.button(msg[1], key ='facebook', use_container_width= True):
        login('facebook')


def process_login_logout(col_user):
    if process_login_callback():
        st.rerun()
    user = get_user()
    with col_user:
        if user:
            col1, col2 = st.columns([1.7,1.3])
            name = get_display_name(user)
            with col1:
                st.markdown(
                    f'<p class="user-badge">Xin chào, {name} 👋</p>',
                    unsafe_allow_html = True
                )
            with col2:
                if st.button("Đăng xuất"):
                    logout()
                    st.rerun()
        else:
            _, btn_col = st.columns([1,1])
            if st.button("Đăng nhập"):
                st.session_state.show = True
    if st.session_state.get('show', False):
        show_modal()


def show_header():
    css_path = os.path.join(os.path.dirname(__file__),"..","assets","css","header.css")
    if os.path.exists(css_path):
        with open(css_path) as f:
            st.markdown(f"<style>{f.read()}</style>",unsafe_allow_html= True)
            f.close()

    # =======Header==========
    col_logo ,col_space, col_user = st.columns([2,5,2.7])
    
    # Logo 
    with col_logo:
        st.markdown("""
            <div class="logo">CanvasAI</div>
        """, unsafe_allow_html= True)

    # Sign in /Sign out
    process_login_logout(col_user)


   
