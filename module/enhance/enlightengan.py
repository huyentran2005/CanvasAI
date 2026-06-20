import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import streamlit as st
import torch
from PIL import Image
from functools import lru_cache


ROOT_DIR = Path(__file__).resolve().parents[2]
ENLIGHTENGAN_DIR = ROOT_DIR / "model" / "EnlightenGAN"
CHECKPOINT_PATH = ENLIGHTENGAN_DIR / "checkpoints" / "enlightening" / "enlightening.pth"


def _add_enlightengan_to_path():
    repo_path = str(ENLIGHTENGAN_DIR)
    if repo_path not in sys.path:
        sys.path.insert(0, repo_path)


def _build_options():
    return SimpleNamespace(
        self_attention=True,
        syn_norm=False,
        use_norm=1,
        use_avgpool=0,
        tanh=False,
        times_residual=True,
        linear_add=False,
        latent_threshold=False,
        latent_norm=False,
        linear=False,
        skip=1,
    )


def _load_enlightengan_model(checkpoint_path=str(CHECKPOINT_PATH)):
    _add_enlightengan_to_path()
    from models.networks import Unet_resize_conv

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    opt = _build_options()
    model = Unet_resize_conv(opt, skip=True).to(device)

    state_dict = torch.load(checkpoint_path, map_location=device)
    clean_state_dict = {
        key.replace("module.", "", 1): value for key, value in state_dict.items()
    }
    model.load_state_dict(clean_state_dict)
    model.eval()
    return model, device


@lru_cache(maxsize=1)
def load_enlightengan_model_cli(checkpoint_path=str(CHECKPOINT_PATH)):
    return _load_enlightengan_model(checkpoint_path)


@st.cache_resource
def load_enlightengan_model(checkpoint_path=str(CHECKPOINT_PATH)):
    return _load_enlightengan_model(checkpoint_path)


def _image_to_tensor(image, device):
    image = image.convert("RGB")
    img = np.asarray(image).astype(np.float32) / 255.0
    img = img * 2.0 - 1.0
    tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)
    return tensor.to(device)


def _attention_from_tensor(tensor):
    r = tensor[:, 0:1] + 1.0
    g = tensor[:, 1:2] + 1.0
    b = tensor[:, 2:3] + 1.0
    return 1.0 - (0.299 * r + 0.587 * g + 0.114 * b) / 2.0


def _tensor_to_image(tensor, original_size):
    array = tensor.squeeze(0).detach().cpu().float().numpy()
    array = (np.transpose(array, (1, 2, 0)) + 1.0) / 2.0 * 255.0
    array = np.clip(array, 0, 255).astype(np.uint8)
    output = Image.fromarray(array, mode="RGB")

    if output.size != original_size:
        output = output.resize(original_size, Image.Resampling.LANCZOS)
    return output


def enhance_with_enlightengan(image, use_streamlit_cache=True):
    original_size = image.size
    if use_streamlit_cache:
        model, device = load_enlightengan_model()
    else:
        model, device = load_enlightengan_model_cli()
    input_tensor = _image_to_tensor(image, device)
    attention = _attention_from_tensor(input_tensor)

    with torch.inference_mode():
        output, _ = model(input_tensor, attention)

    return _tensor_to_image(output, original_size)
