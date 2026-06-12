import numpy as np
import streamlit as st
import torch
import torch.nn as nn
from PIL import Image


ZERO_DCE_CHECKPOINT = "model/zero_dce/zero_dce_best.pth"


class DCENet(nn.Module):
    def __init__(self, num_iterations=8):
        super().__init__()
        self.num_iterations = num_iterations

        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 32, 3, padding=1)
        self.conv4 = nn.Conv2d(32, 32, 3, padding=1)
        self.conv5 = nn.Conv2d(32, 32, 3, padding=1)
        self.conv6 = nn.Conv2d(32, 32, 3, padding=1)
        self.conv7 = nn.Conv2d(32, 3 * num_iterations, 3, padding=1)

        self.relu = nn.ReLU(inplace=True)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x1 = self.relu(self.conv1(x))
        x2 = self.relu(self.conv2(x1))
        x3 = self.relu(self.conv3(x2))
        x4 = self.relu(self.conv4(x3))
        x5 = self.relu(self.conv5(x4))
        x6 = self.relu(self.conv6(x5))
        curves = self.tanh(self.conv7(x6))

        enhanced = x
        for idx in range(self.num_iterations):
            curve_params = curves[:, idx * 3 : (idx + 1) * 3, :, :]
            enhanced = enhanced + curve_params * enhanced * (1 - enhanced)

        return enhanced, curves


@st.cache_resource
def load_zero_dce_model(checkpoint_path=ZERO_DCE_CHECKPOINT):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DCENet(num_iterations=8).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, device


def enhance_with_zero_dce(image):
    model, device = load_zero_dce_model()
    image = image.convert("RGB")
    img_rgb = np.array(image).astype(np.float32) / 255.0
    img_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).unsqueeze(0).to(device)

    with torch.no_grad():
        enhanced, _ = model(img_tensor)
        enhanced = torch.clamp(enhanced, 0, 1)

    output = enhanced.squeeze(0).permute(1, 2, 0).cpu().numpy()
    output = (output * 255.0).astype(np.uint8)
    return Image.fromarray(output)


# ----------------------------------------------------------------------
# TensorFlow Zero-DCE (commented out for reference)
#
# import cv2
# import tensorflow as tf
#
# TF_MODEL_DIR = "model/low_light_image_enhancement"
#
#
# @st.cache_resource
# def load_low_light_model():
#     return tf.saved_model.load(TF_MODEL_DIR)
#
#
# def enhance_with_zero_dce(image):
#     model = load_low_light_model()
#     image = image.convert("RGB")
#     original_img = np.array(image).astype(np.float32) / 255.0
#
#     h, w = original_img.shape[:2]
#     img_resized = cv2.resize(original_img, (400, 400))
#
#     img_tensor = tf.convert_to_tensor(np.expand_dims(img_resized, axis=0), dtype=tf.float32)
#     signature = model.signatures["serving_default"]
#     output = signature(img_tensor)
#     curves = next(iter(output.values()))[0].numpy()
#
#     curves_resized = cv2.resize(curves, (w, h))
#     enhanced = original_img.copy()
#
#     for i in range(8):
#         curve_i = curves_resized[..., i * 3 : (i + 1) * 3].mean(axis=2, keepdims=True) / 4.0
#         enhanced = enhanced * np.exp(-curve_i)
#
#     enhanced = np.clip(enhanced, 0.0, 1.0)
#     enhanced = (enhanced * 255.0).astype(np.uint8)
#     return Image.fromarray(enhanced)
# ----------------------------------------------------------------------
