import streamlit as st
from services import *
from openai import OpenAI
from PIL import Image 
import io
import os
import time
import json
import base64
import cv2
import numpy as np
import google.generativeai as genai
from module.segmentation.intelligent_scissors import display_mask_panel


genai.configure(api_key=st.secrets["GEMINI_API_KEY"])


def _score_to_level(score, labels):
    if score >= 75:
        return labels[0], "low"
    if score >= 45:
        return labels[1], "medium"
    return labels[2], "high"


def analysis_img_by_computation(image):
    image = image.convert("RGB")
    img = np.array(image).astype(np.float32)
    gray = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32)

    brightness = float(gray.mean())
    brightness_score = float(np.clip(100 - abs(brightness - 128.0) / 128.0 * 100.0, 0, 100))

    lap_var = float(cv2.Laplacian(gray, cv2.CV_32F).var())
    sharpness_score = float(np.clip((lap_var / 1800.0) * 100.0, 0, 100))

    rgb_means = img.mean(axis=(0, 1))
    mean_rgb = float(np.mean(rgb_means) + 1e-6)
    channel_delta = float(np.mean(np.abs(rgb_means - mean_rgb)) / mean_rgb)
    color_score = float(np.clip(100 - channel_delta * 220.0, 0, 100))

    blur = cv2.GaussianBlur(gray, (0, 0), 3)
    high_freq = gray - blur
    noise_level = float(np.std(high_freq))
    purity_score = float(np.clip(100 - (noise_level / 35.0) * 100.0, 0, 100))

    brightness_level, brightness_severity = _score_to_level(
        brightness_score, ("Tốt", "Trung bình", "Kém")
    )
    purity_level, purity_severity = _score_to_level(
        purity_score, ("Rất sạch", "Hơi nhiễu", "Nhiễu nặng")
    )
    color_level, color_severity = _score_to_level(
        color_score, ("Chính xác", "Sai lệch nhẹ", "Sai lệch nặng")
    )
    sharpness_level, sharpness_severity = _score_to_level(
        sharpness_score, ("Sắc nét", "Khá rõ", "Mờ nhòe")
    )

    return [
        {
            "name": "Cân bằng sáng",
            "level": brightness_level,
            "score": round(brightness_score, 2),
            "severity": brightness_severity,
        },
        {
            "name": "Độ trong của ảnh",
            "level": purity_level,
            "score": round(purity_score, 2),
            "severity": purity_severity,
        },
        {
            "name": "Độ chuẩn màu",
            "level": color_level,
            "score": round(color_score, 2),
            "severity": color_severity,
        },
        {
            "name": "Độ nét",
            "level": sharpness_level,
            "score": round(sharpness_score, 2),
            "severity": sharpness_severity,
        },
    ]

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
    try:
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
    except Exception as e:
        if "429" in str(e):
            return analysis_img_by_computation(image)
        raise

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
        if st.session_state.analysis_results:
            resuilt = st.session_state.analysis_results

        with st.container(key="container"):
            image_edited = None
            if st.button("PHÂN TÍCH ẢNH",icon="🔍",use_container_width=True, key = 'btn-analysis'):
                image_edited = st.session_state.edited_img
                with st.spinner("⏳AI đang phân tích ảnh ..."):
                    if image_edited:
                            try:
                                resuilt = analysis_img_with_gemini(image_edited, image_edited.format)
                                st.session_state.analysis_results = resuilt
                            except Exception as e:
                                error_placeholder = st.empty()
                                print(f"error: {e}")
                                time.sleep(3)
                                error_placeholder.empty()


            with st.container(key ="analysis-card"):
                for r in resuilt:
                    analysis_item(r["name"], r["level"], r["score"], r["severity"])
            with st.container(key="masked"):
                display_mask_panel()

                
