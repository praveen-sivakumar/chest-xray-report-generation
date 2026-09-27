import os
import io
import time
import json
import random
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
from PIL import Image

import torch
import torch.nn.functional as F
from torchvision import transforms

# -------------------------
# CONFIG: edit these paths
# -------------------------
REPO_DIR = Path("/home/psivakumar/thesis/code/R2GenCMN")  # your R2GenCMN repo
CKPT_PATH = Path("/home/psivakumar/thesis/code/R2GenCMN/outputs/iu_xray_r2gencmn_baseline_5ep/current_checkpoint.pth")
OUTPUT_DIR = Path("/home/psivakumar/thesis/notebooks/streamlit_outputs_r2gencmn")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Add repo to import path
import sys
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

# Repo imports
from models.models import BaseCMNModel
from modules.tokenizers import Tokenizer


# -------------------------
# Helper: safe args object
# -------------------------
@dataclass
class Args:
    dataset_name: str = "iu_xray"
    ann_path: str = ""  # not needed in streamlit
    image_dir: str = "" # not needed in streamlit

    # tokenizer
    threshold: int = 3
    max_seq_length: int = 80

    # vision extractor
    visual_extractor: str = "resnet101"
    visual_extractor_pretrained: bool = True

    # base model dims (must match checkpoint!)
    # These are typical R2GenCMN defaults. If your checkpoint was trained with different dims,
    # we will auto-detect some of them from the checkpoint.
    d_model: int = 512
    d_ff: int = 2048
    num_layers: int = 3
    num_heads: int = 8
    dropout: float = 0.1
    drop_prob_lm: float = 0.1
    d_vf: int = 2048
    use_bn: int = 0

    # CMN memory
    topk: int = 32
    cmm_size: int = 2048
    cmm_dim: int = 512

    # special tokens (we will override)
    bos_idx: int = 0
    eos_idx: int = 0
    pad_idx: int = 0


# -------------------------
# Grad-CAM Engine
# -------------------------
class GradCAM:
    def __init__(self, target_layer):
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.h1 = target_layer.register_forward_hook(self._forward_hook)
        self.h2 = target_layer.register_full_backward_hook(self._backward_hook)

    def _forward_hook(self, module, inputs, output):
        self.activations = output.detach()

    def _backward_hook(self, module, grad_input, grad_output):
        # grad_output is tuple; take first
        self.gradients = grad_output[0].detach()

    def remove(self):
        self.h1.remove()
        self.h2.remove()

    def __call__(self, score):
        """
        score: scalar tensor (backprop target)
        """
        score.backward(retain_graph=True)
        grads = self.gradients          # (B,C,H,W)
        acts  = self.activations        # (B,C,H,W)

        weights = grads.mean(dim=(2, 3), keepdim=True)  # (B,C,1,1)
        cam = (weights * acts).sum(dim=1, keepdim=True) # (B,1,H,W)
        cam = F.relu(cam)

        cam = cam.squeeze(1)  # (B,H,W)
        cam_min = cam.flatten(1).min(dim=1)[0].view(-1, 1, 1)
        cam_max = cam.flatten(1).max(dim=1)[0].view(-1, 1, 1)
        cam = (cam - cam_min) / (cam_max - cam_min + 1e-6)
        return cam


# -------------------------
# Image transforms
# -------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

inv_normalize = transforms.Normalize(
    mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
    std=[1/0.229, 1/0.224, 1/0.225]
)


def pil_to_tensor(pil_img: Image.Image) -> torch.Tensor:
    return transform(pil_img.convert("RGB"))  # (3,224,224)


def tensor_to_uint8_img(t: torch.Tensor) -> np.ndarray:
    """
    t: (3,224,224) normalized
    returns uint8 HxWx3
    """
    x = inv_normalize(t).clamp(0, 1)
    x = (x.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    return x


def overlay_cam(rgb: np.ndarray, cam: np.ndarray, alpha=0.45) -> np.ndarray:
    """
    rgb: HxWx3 uint8
    cam: hxw float [0,1] (any resolution, e.g., 7x7 or 14x14)
    """
    H, W = rgb.shape[:2]

    # Resize CAM to image size
    cam_resized = cv2.resize(cam, (W, H), interpolation=cv2.INTER_CUBIC)
    cam_resized = np.clip(cam_resized, 0.0, 1.0)

    heatmap = (cam_resized * 255).astype(np.uint8)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    out = (rgb.astype(np.float32) * (1 - alpha) + heatmap.astype(np.float32) * alpha)
    out = np.clip(out, 0, 255).astype(np.uint8)
    return out


# -------------------------
# Load model (cached)
# -------------------------
@st.cache_resource
def load_model_and_tokenizer():
    args = Args()

    # Build tokenizer (needs ann_path for vocab build in some repos; yours already works without)
    # If your Tokenizer requires a valid ann_path, point it to your IU annotation json.
    # Example: args.ann_path = "/home/psivakumar/thesis/data/iu_xray/annotation_r2gencmn_2view_r2gencmnformat.json"
    # (only needed for vocab build)
    # We'll set it if file exists:
    default_ann = "/home/psivakumar/thesis/data/iu_xray/annotation_r2gencmn_2view_r2gencmnformat.json"
    if os.path.exists(default_ann):
        args.ann_path = default_ann

    tokenizer = Tokenizer(args)

    # Special tokens: IU tokenizer doesn’t include BOS/EOS explicitly -> set to PAD (0)
    tokenizer.pad_idx = 0
    tokenizer.bos_idx = 0
    tokenizer.eos_idx = 0
    tokenizer.unk_idx = tokenizer.token2idx.get("<unk>", 0)

    # Also set args token ids (model uses args.*)
    args.pad_idx = tokenizer.pad_idx
    args.bos_idx = tokenizer.bos_idx
    args.eos_idx = tokenizer.eos_idx

    # Load checkpoint first (to auto-detect shape-critical args like d_ff/cmm_size)
    ckpt = torch.load(str(CKPT_PATH), map_location="cpu")
    sd = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt

    # Auto-detect d_ff if possible from state_dict
    # encoder_decoder.model.encoder.layers.0.feed_forward.w_1.weight shape: (d_ff, d_model)
    for k, v in sd.items():
        if "feed_forward.w_1.weight" in k and v.ndim == 2:
            args.d_ff = int(v.shape[0])
            args.d_model = int(v.shape[1])
            break

    # Auto-detect memory_matrix shape
    if "encoder_decoder.memory_matrix" in sd:
        args.cmm_size = int(sd["encoder_decoder.memory_matrix"].shape[0])
        args.cmm_dim  = int(sd["encoder_decoder.memory_matrix"].shape[1])

    model = BaseCMNModel(args, tokenizer).to(DEVICE)

    # Strict=False but remove mismatched keys safely
    model_sd = model.state_dict()
    filtered = {}
    mismatched = []
    for k, v in sd.items():
        if k in model_sd and model_sd[k].shape == v.shape:
            filtered[k] = v
        else:
            if k in model_sd:
                mismatched.append((k, tuple(v.shape), tuple(model_sd[k].shape)))

    missing, unexpected = model.load_state_dict(filtered, strict=False)
    model.eval()

    return model, tokenizer, args, mismatched


# -------------------------
# Inference helper
# -------------------------
def generate_report(model, tokenizer, images_2view: torch.Tensor) -> str:
    """
    images_2view: (2,3,224,224)
    """
    images_batch = images_2view.unsqueeze(0).to(DEVICE)  # (1,2,3,224,224)
    with torch.no_grad():
        out = model(images_batch, mode="sample")
        # some repos return (seq, logprobs) etc.
        if isinstance(out, (tuple, list)):
            out = out[0]
        pred = tokenizer.decode_batch(out.cpu().numpy())[0]
    return pred


def compute_cam_for_view(model, images_2view: torch.Tensor, view_index: int):
    """
    Produces GradCAM for a single view using the visual extractor backbone.
    We backprop from a simple scalar target based on extracted features (mean activation).
    """
    # We want a conv layer inside visual_extractor.model (Sequential)
    # Common: model.visual_extractor.model[-1] is layer4 sequential, take its last block conv.
    ve = model.visual_extractor.model

    # Best practical target: last stage first conv
    # (you already found this works): ve[7][0].conv1 or ve[7][0].downsample[0] depending.
    # We'll try a robust approach:
    target_layer = None
    try:
        # last stage sequential
        stage4 = ve[7]
        # first block of stage4 is a Bottleneck; choose its conv3 for most meaningful map
        target_layer = stage4[0].conv3
    except Exception:
        # fallback: first conv we can find in last 2 modules
        for m in reversed(list(ve.modules())):
            if isinstance(m, torch.nn.Conv2d):
                target_layer = m
                break

    cam_engine = GradCAM(target_layer)

    # Prepare view image
    x = images_2view[view_index].unsqueeze(0).to(DEVICE)
    x.requires_grad_(True)

    # Forward through visual extractor
    feats, _ = model.visual_extractor(x)  # repo returns (att_feats, fc_feats) or similar
    # Make scalar score (mean feature activation) for backprop
    score = feats.mean()

    cam = cam_engine(score)[0].detach().cpu().numpy()  # (H,W) normalized
    cam_engine.remove()
    return cam, target_layer


# -------------------------
# Streamlit UI
# -------------------------
st.set_page_config(page_title="CXR Report Generator (R2GenCMN + Grad-CAM)", layout="wide")
st.title("Chest X-ray Report Generation (R2GenCMN) + Grad-CAM + Human-in-the-loop Editing")

st.markdown(
    """
**Workflow**
1) Upload **Frontal** and **Lateral** CXR images  
2) Generate AI report (R2GenCMN)  
3) View **Grad-CAM** for both views  
4) Edit the report (clinician-in-the-loop) and export
"""
)

with st.expander("Model info / Debug", expanded=False):
    st.write("Repo:", str(REPO_DIR))
    st.write("Checkpoint:", str(CKPT_PATH))
    st.write("Device:", DEVICE)

model, tokenizer, args, mismatched = load_model_and_tokenizer()

with st.expander("Checkpoint ↔ Model compatibility", expanded=False):
    st.write(f"Mismatched keys ignored: {len(mismatched)}")
    if len(mismatched) > 0:
        st.write(mismatched[:10])

colA, colB = st.columns(2)
with colA:
    frontal_file = st.file_uploader("Upload **Frontal** image (png/jpg)", type=["png", "jpg", "jpeg"], key="frontal")
with colB:
    lateral_file = st.file_uploader("Upload **Lateral** image (png/jpg)", type=["png", "jpg", "jpeg"], key="lateral")

run_btn = st.button("Generate report + Grad-CAM")

if run_btn:
    if frontal_file is None or lateral_file is None:
        st.error("Please upload BOTH frontal and lateral images.")
        st.stop()

    # Load PIL images
    frontal_pil = Image.open(io.BytesIO(frontal_file.read())).convert("RGB")
    lateral_pil = Image.open(io.BytesIO(lateral_file.read())).convert("RGB")

    # Transform
    frontal_t = pil_to_tensor(frontal_pil)
    lateral_t = pil_to_tensor(lateral_pil)

    images_2view = torch.stack([frontal_t, lateral_t], dim=0)  # (2,3,224,224)

    # Generate report
    with st.spinner("Generating report..."):
        pred_report = generate_report(model, tokenizer, images_2view)

    st.subheader("Generated Report (editable)")
    edited_report = st.text_area(
        "Clinician can edit the AI report here:",
        value=pred_report,
        height=220
    )

    # Grad-CAM
    with st.spinner("Computing Grad-CAM (frontal + lateral)..."):
        cam_f, layer_f = compute_cam_for_view(model, images_2view, view_index=0)
        cam_l, layer_l = compute_cam_for_view(model, images_2view, view_index=1)

    # Visualize
    frontal_rgb = tensor_to_uint8_img(frontal_t)
    lateral_rgb = tensor_to_uint8_img(lateral_t)

    frontal_overlay = overlay_cam(frontal_rgb, cam_f)
    lateral_overlay = overlay_cam(lateral_rgb, cam_l)

    st.subheader("Explainability (Grad-CAM)")
    st.caption(f"Frontal target layer: {layer_f} | Lateral target layer: {layer_l}")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Frontal**")
        st.image(frontal_rgb, caption="Original", use_container_width=True)
        st.image(frontal_overlay, caption="Grad-CAM overlay", use_container_width=True)
    with c2:
        st.markdown("**Lateral**")
        st.image(lateral_rgb, caption="Original", use_container_width=True)
        st.image(lateral_overlay, caption="Grad-CAM overlay", use_container_width=True)

    # Save outputs
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_base = OUTPUT_DIR / f"case_{ts}"

    Image.fromarray(frontal_overlay).save(str(out_base) + "_frontal_cam.png")
    Image.fromarray(lateral_overlay).save(str(out_base) + "_lateral_cam.png")

    with open(str(out_base) + "_report.txt", "w", encoding="utf-8") as f:
        f.write("=== AI Report ===\n")
        f.write(pred_report.strip() + "\n\n")
        f.write("=== Edited Report (HITL) ===\n")
        f.write(edited_report.strip() + "\n")

    st.success(f"Saved outputs to: {OUTPUT_DIR}")