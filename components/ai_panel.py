import streamlit as st
from services import *
from openai import OpenAI
from PIL import Image 
import io
import os
import time
import json
import base64
import google.generativeai as genai


genai.configure(api_key=st.secrets["GEMINI_API_KEY"])

def analysis_item(title, level, percent, color_class):
    st.markdown(f"""
    <div class="analysis-item">
        <div class="item-header">
            <span>{title}</span>
            <span class="badge {color_class}">{level}</span>
        </div>
        <div class="progress-bar">
            <div class="progress-fill {color_class}" style="width:{percent}%"></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

def analysis_img_with_gemini(image, extention = "PNG" ):
    extention = (extention or "PNG").upper()
    if extention == "JPG":
        extention = "JPEG"

    model = genai.GenerativeModel("models/gemini-2.5-flash")
    buf = io.BytesIO()
    image.save(buf, format = extention)
    img_bytes = buf.getvalue()
    prompt = """
        Bạn là chuyên gia phân tích chất lượng hình ảnh. 
        Hãy chấm điểm ảnh từ 0-100 (100 là hoàn hảo nhất) theo 4 chỉ số chất lượng sau:

        1. Cân bằng sáng (Brightness): 100 là ánh sáng hài hòa. Điểm thấp nếu ảnh quá tối hoặc cháy sáng.
        2. Độ trong của ảnh (Purity/Noise): 100 là ảnh cực sạch, không có nhiễu hạt. Điểm thấp nếu ảnh bị nhiễu (noise) nặng.
        3. Độ chuẩn màu (Color Accuracy): 100 là màu sắc trung thực. Điểm thấp nếu màu bị ám hoặc lệch tông.
        4. Độ chi tiết (Sharpness): 100 là ảnh sắc nét. Điểm thấp nếu ảnh bị mờ nhòe.

        Trả về JSON đúng format:
        {
          "items": [
            {
              "name": "Cân bằng sáng",
              "level": "Tốt/Trung bình/Kém",
              "score": 0-100,
              "severity": "low/medium/high"
            },
            {
              "name": "Độ trong của ảnh",
              "level": "Rất sạch/Hơi nhiễu/Nhiễu nặng",
              "score": 0-100,
              "severity": "low/medium/high"
            },
            {
              "name": "Độ chuẩn màu",
              "level": "Chính xác/Sai lệch nhẹ/Sai lệch nặng",
              "score": 0-100,
              "severity": "low/medium/high"
            },
            {
              "name": "Độ chi tiết",
              "level": "Sắc nét/Khá rõ/Mờ nhòe",
              "score": 0-100,
              "severity": "low/medium/high"
            }
          ]
        }
        Lưu ý: Chỉ trả về JSON.
    """
    response = model.generate_content(
        contents=[
            prompt,
            {
                "mime_type": f"image/{extention.lower()}",
                "data": img_bytes
            }
        ],
        generation_config={"response_mime_type": "application/json"}
    )
    raw_text = response.text.strip()
    if raw_text.startswith("```"):
        raw_text = raw_text.replace("```json", "").replace("```", "").strip()
            
    return json.loads(raw_text)["items"]

def display_right_panel():
    embed_css("ai_panel.css")
    with st.container(key ="right_panel"):
        st.subheader("AI ASSISTANT", anchor= False)

        resuilt = [
                    {
                        "name": "Cân bằng sáng",
                        "level": "None",
                        "score": 0,
                        "severity": "None"
                    },
                    {
                        "name": "Độ trong của ảnh",
                        "level": "None",
                        "score": 0,
                        "severity": "None"
                    },
                    {
                        "name": "Độ chuẩn màu",
                        "level": "None",
                        "score": 0,
                        "severity": "None"
                    },
                    {
                        "name": "Độ nét",
                        "level": "None",
                        "score": 0,
                        "severity": "None"
                    },
        ]
        with st.container(key="container"):
            image_edited = None
            if st.button("PHÂN TÍCH ẢNH",icon="🔍",use_container_width=True, key = 'btn-analysis'):
                image_edited = st.session_state.edited_img
                with st.spinner("⏳AI đang phân tích ảnh ..."):
                    if image_edited:
                            try:
                                resuilt = analysis_img_with_gemini(image_edited, image_edited.format)
                            except Exception as e:
                                error_placeholder = st.empty()
                                if "429" in str(e):
                                    error_placeholder.error("⚠️ Bạn đã dùng hết lượt phân tích miễn phí!")
                                else:
                                    print(f"error: {e}")
                                time.sleep(3)
                                error_placeholder.empty()


                with st.container(key ="analysis-card"):
                    for r in resuilt:
                        analysis_item(r["name"], r["level"], r["score"], r["severity"])
            

            with st.container(key="masked"):
                st.caption("DETECTION MASK")
                with st.container(key="masked_img"):
                    if not st.session_state.masked_img:
                        mask_path = "assets/favicon/mask.jpg"
                        st.session_state.masked_img = Image.open(mask_path)
                    
                    st.image(st.session_state.masked_img)


                