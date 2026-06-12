import numpy as np
import streamlit as st
import torch
from PIL import Image
import os
from pathlib import Path

# Đặt thư mục cache HuggingFace vào dự án
HF_CACHE = Path(__file__).parent.parent.parent / "model_cache"
os.environ["HF_HOME"] = str(HF_CACHE)
os.environ["TRANSFORMERS_CACHE"] = str(HF_CACHE)

try:
    _torchvision_lib = torch.library.Library("torchvision", "DEF")
    _torchvision_lib.define("nms(Tensor dets, Tensor scores, float iou_threshold) -> Tensor")
except RuntimeError:
    _torchvision_lib = None

from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
from segment_anything import (
    sam_model_registry,
    SamPredictor
)


SAM_CHECKPOINT = "model/sam/sam_vit_b_01ec64.pth"
GROUNDING_DINO_MODEL_ID = "IDEA-Research/grounding-dino-tiny"
GROUNDING_DINO_LOCAL_DIR = HF_CACHE / "IDEA-Research" / "grounding-dino-tiny"


@st.cache_resource(show_spinner=False)
def load_grounding_sam():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_source = GROUNDING_DINO_LOCAL_DIR if GROUNDING_DINO_LOCAL_DIR.exists() else GROUNDING_DINO_MODEL_ID

    processor = AutoProcessor.from_pretrained(
        model_source,
        cache_dir=str(HF_CACHE),
        trust_remote_code=True
    )
    model = AutoModelForZeroShotObjectDetection.from_pretrained(
        model_source,
        cache_dir=str(HF_CACHE),
        trust_remote_code=True
    )
    model.to(device)
    model.eval()

    sam = sam_model_registry["vit_b"](checkpoint=SAM_CHECKPOINT)
    sam.to(device=device)
    predictor = SamPredictor(sam)

    return processor, model, predictor, device


def normalize_prompt(prompt):
    prompt = (prompt or "").strip()
    if prompt and not prompt.endswith("."):
        prompt = f"{prompt}."
    return prompt


def detect_box(image_pil, prompt, box_threshold=0.35, text_threshold=0.25):
    prompt = normalize_prompt(prompt)
    if not prompt or image_pil is None:
        return None
    
    processor, model, _, device = load_grounding_sam()
    image_pil = image_pil.convert("RGB")

    inputs = processor(
        images=image_pil,
        text=prompt,
        return_tensors="pt"
    )
    inputs = {key: value.to(device) for key, value in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
    
    results = processor.post_process_grounded_object_detection(
        outputs,
        inputs["input_ids"],
        target_sizes=[image_pil.size[::-1]]
    )

    boxes = results[0]["boxes"]
    scores = results[0]["scores"]
    
    # Lọc theo threshold
    valid_mask = scores >= text_threshold
    boxes = boxes[valid_mask]
    
    if len(boxes) == 0:
        return None
    
    return boxes.detach().cpu().numpy()
    
def segment_with_sam(image_rgb, box):
    _, _, predictor, _ = load_grounding_sam()
    predictor.set_image(image_rgb)

    masks, scores, logits = (
        predictor.predict(box=box, multimask_output=False)
    )

    return masks[0]

def grounding_sam_pipeline(image_pil=None, prompt=None):
    if image_pil is None:
        image_pil = st.session_state.get("edited_img")

    if prompt is None:
        prompt = st.session_state.get("text_prompt")

    if image_pil is None or not (prompt or "").strip():
        return None

    boxes = detect_box(image_pil, prompt)
    if boxes is None:
        return None

    image_rgb = np.array(image_pil.convert("RGB"))
    mask = np.zeros(image_rgb.shape[:2], dtype=np.uint8)

    for box in boxes:
        box = box.astype(np.int32)
        object_mask = (
            segment_with_sam(
                image_rgb,
                box
            )
        )
        mask[object_mask == True] = 255
    
    return mask


def grounding_sam_pipline():
    return grounding_sam_pipeline()
