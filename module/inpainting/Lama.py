import io
import requests
import onnxruntime
import numpy as np
from PIL import Image
import streamlit as st



MODEL_PATH = "model/lama/lama_fp32.onnx"

def open_image(source) -> Image.Image:
    if isinstance(source, Image.Image):
        return source
    if isinstance(source, np.ndarray):
        arr = source
        # Bool array (True/False) → 0/255
        if arr.dtype == bool:
            arr = arr.astype(np.uint8) * 255
        # Float [0,1] → 0/255
        elif arr.dtype in (np.float32, np.float64):
            arr = (np.clip(arr, 0, 1) * 255).astype(np.uint8)
        # Squeeze chiều thừa: (H,W,1) hoặc (1,H,W) → (H,W)
        if arr.ndim == 3 and arr.shape[0] == 1:
            arr = arr[0]
        elif arr.ndim == 3 and arr.shape[2] == 1:
            arr = arr[:, :, 0]
        return Image.fromarray(arr)
    if isinstance(source, str):
        if source.startswith("http"):
            r = requests.get(source, timeout=15)
            r.raise_for_status()
            return Image.open(io.BytesIO(r.content))
        return Image.open(source)
    raise TypeError(f"Không hỗ trợ: {type(source)}")


def to_chw_float(image):
    arr = np.array(image) if isinstance(image, Image.Image) else image.copy()
    if arr.ndim == 2:
        arr = arr[np.newaxis,...]
    else:
        arr = np.transpose(arr,(2,0,1))
    
    return arr.astype(np.float32) / 255.0

def get_model_input_size(session):
    """Đọc kích thước input cố định từ ONNX model (vd: 512x512)."""
    shape = session.get_inputs()[0].shape
    h, w  = shape[2], shape[3]
    h = h if isinstance(h, int) else 512
    w = w if isinstance(w, int) else 512
    return h,w

def prepare_inputs(session, image, mask):
    """
    Chuẩn bị tensor đầu vào:
    - Resize ảnh và mask về đúng kích thước model yêu cầu
    - Giữ lại kích thước gốc để crop lại sau
    """
    model_h, model_w = get_model_input_size(session)
    
    img_pil = open_image(image).convert('RGB')
    msk_pil = open_image(mask).convert('L')
    original_size = img_pil.size

    img_resized = img_pil.resize((model_w, model_h), Image.LANCZOS)
    msk_resized = msk_pil.resize((model_w, model_h), Image.NEAREST)

    img_arr = to_chw_float(img_resized)
    msk_arr = to_chw_float(msk_resized)
    msk_arr = (msk_arr > 0).astype(np.float32)

    return (
        img_arr[np.newaxis], # [1,3,H,W]
        msk_arr[np.newaxis], # [1,1,H,W]
        original_size
    )

@st.cache_resource
def load_model(model_path = MODEL_PATH, use_gpu = False):
    providers = (
        ['CUDAExecutionProvider', 'CPUExecutionProvider']
        if use_gpu else ['CPUExecutionProvider']
    )

    sess = onnxruntime.InferenceSession(model_path, providers=providers)
    h,w = get_model_input_size(sess)
    print(f'Model loaded | Provider: {sess.get_providers()[0]}')
    print(f'Input size cố định: {w}×{h}')
    return sess

def run_inpaint(image, mask):
    session = load_model(MODEL_PATH, use_gpu= False)
    if isinstance(mask, np.ndarray):
        Image.fromarray(mask).save("debug_mask.png")
    if isinstance(image, np.ndarray):
        Image.fromarray(mask).save("debug_image.png")
    img, msk, orig_size = prepare_inputs(session, image, mask)
    print(f"[img tensor]  shape={img.shape}, min={img.min():.3f}, max={img.max():.3f}")
    print(f"[msk tensor]  shape={msk.shape}, min={msk.min():.3f}, max={msk.max():.3f}, "
          f"nonzero={np.count_nonzero(msk)}")

    # Lưu mask tensor để kiểm tra vùng model thấy
    msk_vis = (msk[0, 0] * 255).astype(np.uint8)
    Image.fromarray(msk_vis).save("debug_mask_tensor.png")
    outputs = session.run(None, {
        'image': img.astype(np.float32),
        'mask' : msk.astype(np.float32),
    })
    result = outputs[0][0]                        # [3, H, W]
    result = np.transpose(result, (1, 2, 0))      # [H, W, 3]
    result = np.clip(result, 0, 255).astype(np.uint8)
    Image.fromarray(result).save("debug_output_raw.png")

    result_pil = Image.fromarray(result)
    # Resize kết quả về kích thước ảnh gốc
    result_pil = result_pil.resize(orig_size, Image.LANCZOS)
    result_pil.save("result.png")
    print("thanh cong")
    return result_pil
