import streamlit as st
import requests
import os
import components.header as header



# set favicon
st.set_page_config(
    page_title = "CanvasAI",
    page_icon ="frontend/assets/favicon/painter.png",
    layout = "wide"
)

header.show_header()

st.title("Demo Hệ thống Xử lý Thông minh")

x = st.number_input("Nhập số", value=1)

if st.button("Tính"):
    try:
        # Gửi request đến địa chỉ của FastAPI
        res = requests.post(
            "http://127.0.0.1:8000/predict",
            json={"x": x}
        )

        if res.status_code == 200:
            data = res.json()
            st.success(f"Kết quả từ Backend: {data['result']}")
        else:
            st.error(f"Lỗi hệ thống: Mã lỗi {res.status_code}")

    except Exception as e:
        st.error(f"Không kết nối được backend. Chi tiết: {e}")