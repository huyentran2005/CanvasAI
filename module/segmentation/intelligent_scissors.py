import cv2
import math
import numpy as np
import streamlit as st
from PIL import Image
from streamlit_image_coordinates import streamlit_image_coordinates

SCISSOR_MAX_WIDTH = 760
SCISSOR_MAX_HEIGHT = 600


def is_close_to_first(point, first_point, threshold=10):
    distance = math.sqrt(
        (point[0] - first_point[0]) ** 2 + (point[1] - first_point[1]) ** 2
    )
    return distance < threshold


def ensure_scissor_state():
    defaults = {
        "scissor_points": [],
        "scissor_segments": [],
        "scissor_last_click": None,
        "scissor_undo_stack": [],
        "scissor_redo_stack": [],
        "mask": None,
        "masked_img": None,
        "scissor_active": False,
        "scissor_completed": False,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def clear_scissor_state(clear_preview=True):
    st.session_state.scissor_points = []
    st.session_state.scissor_segments = []
    st.session_state.scissor_last_click = None
    st.session_state.scissor_undo_stack = []
    st.session_state.scissor_redo_stack = []
    st.session_state.mask = None
    st.session_state.scissor_completed = False
    if clear_preview:
        st.session_state.masked_img = None


def capture_scissor_snapshot():
    return {
        "points": st.session_state.scissor_points.copy(),
        "segments": st.session_state.scissor_segments.copy(),
        "last_click": st.session_state.scissor_last_click,
        "mask": None if st.session_state.mask is None else st.session_state.mask.copy(),
        "completed": st.session_state.scissor_completed,
    }


def restore_scissor_snapshot(snapshot):
    st.session_state.scissor_points = snapshot["points"].copy()
    st.session_state.scissor_segments = snapshot["segments"].copy()
    st.session_state.scissor_last_click = snapshot["last_click"]
    st.session_state.mask = None if snapshot["mask"] is None else snapshot["mask"].copy()
    st.session_state.scissor_completed = snapshot["completed"]
    if st.session_state.mask is None:
        st.session_state.masked_img = None
    else:
        st.session_state.masked_img = Image.fromarray(st.session_state.mask)


def push_scissor_history():
    st.session_state.scissor_undo_stack.append(capture_scissor_snapshot())
    st.session_state.scissor_redo_stack.clear()


def undo_scissor_step():
    if not st.session_state.scissor_undo_stack:
        return
    st.session_state.scissor_redo_stack.append(capture_scissor_snapshot())
    snapshot = st.session_state.scissor_undo_stack.pop()
    restore_scissor_snapshot(snapshot)


def redo_scissor_step():
    if not st.session_state.scissor_redo_stack:
        return
    st.session_state.scissor_undo_stack.append(capture_scissor_snapshot())
    snapshot = st.session_state.scissor_redo_stack.pop()
    restore_scissor_snapshot(snapshot)


def build_scissor_image(image):
    rgb = np.array(image.convert("RGB"))
    preview_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    if st.session_state.scissor_completed and len(st.session_state.scissor_points) >= 2:
        cv2.line(
            preview_bgr,
            st.session_state.scissor_points[0],
            st.session_state.scissor_points[-1],
            (0, 255, 0),
            2,
        )

    for seg in st.session_state.scissor_segments:
        if len(seg) > 1:
            cv2.polylines(preview_bgr, [seg], False, (0, 255, 0), 2)

    for idx, p in enumerate(st.session_state.scissor_points):
        if idx == 0:
            cv2.circle(preview_bgr, p, 5, (0, 0, 255), -1)
        else:
            cv2.circle(preview_bgr, p, 3, (0, 0, 0), -1)

    return cv2.cvtColor(preview_bgr, cv2.COLOR_BGR2RGB)


def fit_image_for_display(image, max_width=SCISSOR_MAX_WIDTH, max_height=SCISSOR_MAX_HEIGHT):
    width, height = image.size
    scale = min(max_width / width, max_height / height, 1.0)
    if scale < 1.0:
        display_size = (max(1, int(width * scale)), max(1, int(height * scale)))
        return image.resize(display_size, Image.LANCZOS), scale
    return image, 1.0


def build_mask_from_state(image):
    if len(st.session_state.scissor_points) < 3:
        return None

    scissor = cv2.segmentation_IntelligentScissorsMB()
    image_rgb = np.array(image.convert("RGB"))
    scissor.applyImage(image_rgb)

    first_point = st.session_state.scissor_points[0]
    scissor.buildMap(st.session_state.scissor_points[-1])
    closing = np.array(scissor.getContour(first_point), dtype=np.int32)

    all_segments = st.session_state.scissor_segments.copy()
    if len(closing) > 0:
        all_segments.append(closing)

    if not all_segments:
        return None

    polygon = np.concatenate(all_segments, axis=0)
    mask = np.zeros(image_rgb.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [polygon], 255)
    return mask


def handle_scissor_click(image, clicked):
    if clicked is None:
        return

    current_click = (clicked["x"], clicked["y"])
    if current_click == st.session_state.scissor_last_click:
        return

    push_scissor_history()
    st.session_state.scissor_last_click = current_click

    if (
        len(st.session_state.scissor_points) >= 3
        and is_close_to_first(current_click, st.session_state.scissor_points[0])
    ):
        mask = build_mask_from_state(image)
        if mask is not None:
            st.session_state.mask = mask
            st.session_state.scissor_completed = True
            st.session_state.masked_img = Image.fromarray(mask)
        st.rerun()
        return

    if st.session_state.scissor_completed:
        return

    st.session_state.scissor_points.append(current_click)

    if len(st.session_state.scissor_points) >= 2:
        scissor = cv2.segmentation_IntelligentScissorsMB()
        image_rgb = np.array(image.convert("RGB"))
        scissor.applyImage(image_rgb)

        start = st.session_state.scissor_points[-2]
        end = st.session_state.scissor_points[-1]
        scissor.buildMap(start)

        contour = np.array(scissor.getContour(end), dtype=np.int32)
        if len(contour) > 0:
            st.session_state.scissor_segments.append(contour)

    st.session_state.mask = None
    st.rerun()


def draw_mask():
    if st.session_state.edited_img is None:
        return

    ensure_scissor_state()
    image = st.session_state.edited_img
    preview_rgb = build_scissor_image(image)
    display_preview, scale = fit_image_for_display(Image.fromarray(preview_rgb))

    centered_col = st.columns([1, 6, 1])
    with centered_col[1]:
        clicked = streamlit_image_coordinates(
            display_preview,
            width=display_preview.width,
            height=display_preview.height,
            use_column_width="never",
            key="scissor_image",
        )
    if clicked is not None and scale != 1.0:
        clicked = {
            "x": int(clicked["x"] / scale),
            "y": int(clicked["y"] / scale),
        }

    handle_scissor_click(image, clicked)


def display_mask_panel():
    if st.session_state.edited_img is None:
        return

    ensure_scissor_state()

    st.caption("MASK PREVIEW")
    if st.session_state.masked_img is None:
        placeholder = Image.open("assets/favicon/mask.jpg")
        st.session_state.masked_img = placeholder


    st.image(st.session_state.masked_img, use_container_width=True)
