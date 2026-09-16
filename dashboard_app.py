# Requires one extra package on top of this project's existing venv:
#   uv pip install streamlit
# Run with (NOT `python dashboard_app.py`):
#   streamlit run dashboard_app.py
#
# Theming note: this file's CSS handles cosmetic polish (cards, banner, buttons), but the
# file uploader / selectbox / native tables follow Streamlit's own theme engine, which page
# CSS can't reliably reach. The accompanying .streamlit/config.toml (same folder as this
# file) is what actually makes those widgets light instead of the default dark theme — it
# is not optional CSS polish, the app looks broken without it.

import json
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont

from citrus_model import CitrusNet
from citrus_transfer import build_model, IMAGENET_MEAN, IMAGENET_STD
from citrus_common import load_normalization_stats, get_transforms

# --------------------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
CHECKPOINT_DIR = BASE_DIR / "checkpoints"
METRICS_CSV_PATH = BASE_DIR / "model_comparison_metrics.csv"
NORM_STATS_PATH = BASE_DIR / "normalization_stats.json"
RUNS_DIR = BASE_DIR / "dashboard_runs"

CLASS_TO_IDX = {"aphids": 0, "gummosis": 1, "healthy": 2, "leaf_minnor": 3}
IDX_TO_CLASS = {v: k for k, v in CLASS_TO_IDX.items()}
DISPLAY_NAMES = {
    "aphids": "Aphids",
    "gummosis": "Gummosis",
    "healthy": "Healthy",
    "leaf_minnor": "Leaf Miner",
}
CLASS_ORDER = ["aphids", "gummosis", "healthy", "leaf_minnor"]  # fixed order, matches idx 0-3
CLASS_ICONS = {"aphids": "🐜", "gummosis": "🩹", "healthy": "✅", "leaf_minnor": "🕳️"}
# Distinct from MODEL_COLORS below — this palette is keyed by *diagnosis*, used on the
# annotated-image badge, independent of which model produced the verdict.
CLASS_ACCENT_COLORS = {
    "aphids": "#C62828",
    "gummosis": "#8D6E63",
    "healthy": "#2E7D32",
    "leaf_minnor": "#EF6C00",
}

ARCH_ORDER = ["citrusnet", "resnet50", "efficientnet_b0", "densenet121"]
ARCH_DISPLAY_NAMES = {
    "citrusnet": "CitrusNet",
    "resnet50": "ResNet50",
    "efficientnet_b0": "EfficientNet-B0",
    "densenet121": "DenseNet121",
}
ARCH_CHECKPOINTS = {
    "citrusnet": "baseline_best.pt",
    "resnet50": "tl_resnet50_phase2_best.pt",
    "efficientnet_b0": "tl_efficientnet_b0_phase2_best.pt",
    "densenet121": "tl_densenet121_phase2_best.pt",
}
# Consistent per-model color used across verdict cards, the leaderboard, and the radar chart.
MODEL_COLORS = {
    "citrusnet": "#2E7D32",       # green
    "resnet50": "#EF6C00",        # orange
    "efficientnet_b0": "#1565C0", # blue
    "densenet121": "#8E24AA",     # purple
}

# Design tokens — single source of truth for both the injected CSS and the matplotlib
# figures, so charts don't look like they were pasted on top of a differently-colored page.
APP_BG = "#FFFBF5"
APP_SURFACE = "#FFFFFF"
APP_SIDEBAR_BG = "#FFF3E0"
APP_TEXT = "#3E2723"
APP_MUTED_TEXT = "#8D6E63"
APP_BORDER = "#EAD9C4"
APP_GREEN = "#2E7D32"
APP_ORANGE = "#EF6C00"
APP_GOLD = "#F9A825"

TRUE_CLASS_OPTIONS = ["Unknown", "Aphids", "Gummosis", "Healthy", "Leaf Miner"]
TRUE_CLASS_TO_INTERNAL = {
    "Aphids": "aphids",
    "Gummosis": "gummosis",
    "Healthy": "healthy",
    "Leaf Miner": "leaf_minnor",
}


# --------------------------------------------------------------------------------------
# Cached loaders
# --------------------------------------------------------------------------------------

@st.cache_resource
def load_all_models():
    """Build all four models, load their trained weights, and hand back the Grad-CAM
    target layer for each. Cached so this only runs once per app session."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models = {}

    citrusnet = CitrusNet(num_classes=4)
    citrusnet.load_state_dict(
        torch.load(CHECKPOINT_DIR / ARCH_CHECKPOINTS["citrusnet"], map_location=device)
    )
    models["citrusnet"] = {"model": citrusnet, "target_layer": citrusnet.block4}

    resnet50 = build_model("resnet50", num_classes=4)
    resnet50.load_state_dict(
        torch.load(CHECKPOINT_DIR / ARCH_CHECKPOINTS["resnet50"], map_location=device)
    )
    models["resnet50"] = {"model": resnet50, "target_layer": resnet50.layer4}

    efficientnet_b0 = build_model("efficientnet_b0", num_classes=4)
    efficientnet_b0.load_state_dict(
        torch.load(CHECKPOINT_DIR / ARCH_CHECKPOINTS["efficientnet_b0"], map_location=device)
    )
    # Last MBConv block before pooling is the standard Grad-CAM target for EfficientNet.
    models["efficientnet_b0"] = {"model": efficientnet_b0, "target_layer": efficientnet_b0.features[-1]}

    densenet121 = build_model("densenet121", num_classes=4)
    densenet121.load_state_dict(
        torch.load(CHECKPOINT_DIR / ARCH_CHECKPOINTS["densenet121"], map_location=device)
    )
    # Full feature Sequential, hooked before the final ReLU/pool — standard for DenseNet.
    models["densenet121"] = {"model": densenet121, "target_layer": densenet121.features}

    for arch_key in ARCH_ORDER:
        model = models[arch_key]["model"]
        model.eval().to(device)
        # Freeze parameters: Grad-CAM only needs gradients flowing back from the input,
        # not gradients w.r.t. the weights, so this saves memory/time on every upload.
        for p in model.parameters():
            p.requires_grad_(False)

    return models, device


@st.cache_data
def load_metrics():
    return pd.read_csv(METRICS_CSV_PATH)


@st.cache_data
def load_norm_stats():
    return load_normalization_stats(NORM_STATS_PATH)


def build_transforms():
    citrus_mean, citrus_std = load_norm_stats()
    citrus_transform = get_transforms(augment=False, mean=citrus_mean, std=citrus_std, size=224)
    imagenet_transform = get_transforms(augment=False, mean=IMAGENET_MEAN, std=IMAGENET_STD, size=224)
    return {
        "citrusnet": citrus_transform,
        "resnet50": imagenet_transform,
        "efficientnet_b0": imagenet_transform,
        "densenet121": imagenet_transform,
    }


# --------------------------------------------------------------------------------------
# Grad-CAM
# --------------------------------------------------------------------------------------

class GradCAM:
    """Manual forward/backward-hook Grad-CAM, parameterized only by which module is
    hooked as the target layer. Works identically for all four architectures.

    Deliberately hooks the *tensor* (via output.register_hook) rather than using
    register_full_backward_hook on the module: several torchvision forward() methods
    (DenseNet in particular: `out = F.relu(features, inplace=True)` right after
    `self.features(x)`) mutate the target layer's output in-place immediately after it's
    returned. register_full_backward_hook wraps that output in a view that autograd
    forbids modifying in-place, which raises a RuntimeError. A plain tensor hook has no
    such restriction. We also clone the activation at capture time so that later in-place
    mutation of the original tensor (as DenseNet does) can't retroactively change the
    values we use for the CAM.
    """

    def __init__(self, model, target_layer):
        self.model = model
        self.activations = None
        self.gradients = None
        self._fwd_handle = target_layer.register_forward_hook(self._forward_hook)

    def _forward_hook(self, module, inputs, output):
        self.activations = output.detach().clone()
        output.register_hook(self._save_gradient)

    def _save_gradient(self, grad):
        self.gradients = grad.detach().clone()

    def compute(self, output, class_idx):
        """Backprop from the already-computed forward `output` for `class_idx`, using the
        activations/gradients captured by the hooks, and return a 224x224 [0,1] heatmap."""
        self.model.zero_grad(set_to_none=True)
        output[:, class_idx].sum().backward()

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((weights * self.activations).sum(dim=1, keepdim=True))
        cam = cam.squeeze().cpu().numpy().astype(np.float32)
        cam = cv2.resize(cam, (224, 224))
        cam -= cam.min()
        peak = cam.max()
        if peak > 1e-8:
            cam /= peak
        return cam

    def remove(self):
        self._fwd_handle.remove()


def run_model(model, target_layer, image, transform, device):
    """Preprocess, run a single gradient-enabled forward pass (timed), compute softmax
    predictions, then reuse that same forward pass for Grad-CAM (no second forward)."""
    input_tensor = transform(image).unsqueeze(0).to(device)
    input_tensor.requires_grad_(True)

    gradcam = GradCAM(model, target_layer)

    if device.type == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    output = model(input_tensor)
    if device.type == "cuda":
        torch.cuda.synchronize()
    latency_ms = (time.perf_counter() - start) * 1000

    probs = F.softmax(output, dim=1)
    pred_idx = int(torch.argmax(probs, dim=1).item())
    confidence = float(probs[0, pred_idx].item())

    heatmap = gradcam.compute(output, pred_idx)
    gradcam.remove()

    return {
        "pred_idx": pred_idx,
        "pred_label": IDX_TO_CLASS[pred_idx],
        "pred_display": DISPLAY_NAMES[IDX_TO_CLASS[pred_idx]],
        "confidence": confidence,
        "probs": probs.detach().cpu().numpy().flatten(),
        "latency_ms": latency_ms,
        "heatmap": heatmap,
    }


# --------------------------------------------------------------------------------------
# Small formatting helpers
# --------------------------------------------------------------------------------------

def format_param_count(n):
    n = float(n)
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    return f"{n / 1_000:.0f}K"


def to_accuracy_percent(value):
    """metrics CSV may store accuracy as a 0-1 fraction or an already-scaled percentage."""
    value = float(value)
    return value * 100 if value <= 1.0 else value


def _load_display_font(size):
    # Reuse matplotlib's bundled DejaVu Sans Bold rather than depending on a system font
    # being present on the deployment machine — matplotlib is already a hard dependency.
    try:
        font_path = Path(matplotlib.get_data_path()) / "fonts" / "ttf" / "DejaVuSans-Bold.ttf"
        return ImageFont.truetype(str(font_path), size)
    except Exception:
        return ImageFont.load_default()


def annotate_image_with_verdict(image, ensemble, results):
    """Return a copy of the uploaded image with the ensemble verdict and its confidence
    burned into a badge in the corner, so the headline result travels with the photo.

    Deliberately does NOT draw a CLASS_ICONS emoji here (unlike the HTML cards elsewhere,
    which the browser renders fine). DejaVu Sans Bold — the font used for PIL text drawing
    below — has no emoji glyphs, so an emoji character in this text produces a "missing
    glyph" tofu box instead of an icon. A plain drawn circle sidesteps the font entirely.
    """
    base = image.convert("RGBA")
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    avg_probs = np.mean([results[a]["probs"] for a in ARCH_ORDER], axis=0)
    soft_vote_class = IDX_TO_CLASS[ensemble["soft_vote_idx"]]
    confidence_pct = float(avg_probs[ensemble["soft_vote_idx"]]) * 100
    label_text = f"{ensemble['soft_vote_label']} · {confidence_pct:.1f}% confidence"

    font_size = max(16, base.width // 24)
    font = _load_display_font(font_size)

    padding_x, padding_y = 18, 12
    dot_d = font_size
    dot_gap = 10

    text_bbox = draw.textbbox((0, 0), label_text, font=font)
    text_w, text_h = text_bbox[2] - text_bbox[0], text_bbox[3] - text_bbox[1]
    box_w = text_w + padding_x * 2 + dot_d + dot_gap
    box_h = max(text_h, dot_d) + padding_y * 2
    margin = 16
    box = (margin, base.height - box_h - margin, margin + box_w, base.height - margin)

    accent_hex = CLASS_ACCENT_COLORS[soft_vote_class].lstrip("#")
    accent_rgb = tuple(int(accent_hex[i:i + 2], 16) for i in (0, 2, 4))
    draw.rounded_rectangle(box, radius=14, fill=accent_rgb + (225,))

    dot_cx = box[0] + padding_x + dot_d / 2
    dot_cy = (box[1] + box[3]) / 2
    draw.ellipse(
        [dot_cx - dot_d / 2, dot_cy - dot_d / 2, dot_cx + dot_d / 2, dot_cy + dot_d / 2],
        fill=(255, 255, 255, 255),
    )

    text_x = dot_cx + dot_d / 2 + dot_gap
    text_y = box[1] + padding_y - text_bbox[1]
    draw.text((text_x, text_y), label_text, font=font, fill=(255, 255, 255, 255))

    return Image.alpha_composite(base, overlay).convert("RGB")


# --------------------------------------------------------------------------------------
# Matplotlib theming (shared by the radar chart and the Grad-CAM grid)
# --------------------------------------------------------------------------------------

def apply_matplotlib_theme(fig, axes):
    """Match a matplotlib figure to the app's cream/brown palette instead of leaving
    Streamlit's default plain-white canvas, so charts feel embedded in the page rather
    than pasted on top of it. Call this right after creating the figure/axes — any
    per-title or per-legend colors should be set afterward at the call site."""
    fig.patch.set_facecolor(APP_BG)
    for ax in np.atleast_1d(axes).flatten():
        ax.set_facecolor(APP_SURFACE)
        ax.tick_params(colors=APP_MUTED_TEXT)
        for spine in ax.spines.values():
            spine.set_color(APP_BORDER)


# --------------------------------------------------------------------------------------
# Page setup / CSS
# --------------------------------------------------------------------------------------

def setup_page():
    st.set_page_config(page_title="Citrus Leaf Doctor", page_icon="🍊", layout="wide")
    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@600;700&family=Inter:wght@400;500;600&display=swap');

        html, body, [class*="css"] {{ font-family: 'Inter', sans-serif; }}

        .stApp {{ background-color: {APP_BG} !important; }}
        .stApp, .stApp p, .stApp span, .stApp label, .stApp li,
        section[data-testid="stSidebar"] * {{ color: {APP_TEXT} !important; }}
        h1, h2, h3, h4 {{ font-family: 'Poppins', sans-serif; color: {APP_GREEN} !important; }}

        header[data-testid="stHeader"] {{ background-color: transparent !important; }}

        section[data-testid="stSidebar"] {{ background-color: {APP_SIDEBAR_BG} !important; border-right: 1px solid {APP_BORDER}; }}
        section[data-testid="stSidebar"] hr {{ border-color: {APP_BORDER} !important; }}

        /* Widget backgrounds below are a light CSS polish only — the actual fix for the
           file uploader / selectbox / dataframe rendering dark is the accompanying
           .streamlit/config.toml theme, which page CSS alone cannot reliably override. */
        div[data-testid="stFileUploaderDropzone"] {{
            background-color: {APP_SURFACE} !important;
            border: 1.5px dashed {APP_ORANGE} !important;
            border-radius: 12px !important;
        }}
        div[data-testid="stFileUploaderDropzone"] section {{ background-color: transparent !important; }}

        div[data-baseweb="select"] > div {{
            background-color: {APP_SURFACE} !important;
            border-radius: 10px !important;
            border-color: {APP_BORDER} !important;
        }}

        .stButton button, .stDownloadButton button {{
            background-color: {APP_ORANGE} !important;
            color: #FFFFFF !important;
            border: none !important;
            border-radius: 10px !important;
            font-weight: 600 !important;
        }}
        .stButton button:hover, .stDownloadButton button:hover {{ background-color: #D84315 !important; }}

        [data-baseweb="tab-highlight"] {{ background-color: {APP_ORANGE} !important; }}
        [data-baseweb="tab"] {{ color: {APP_MUTED_TEXT} !important; }}
        [aria-selected="true"][data-baseweb="tab"] {{ color: {APP_ORANGE} !important; font-weight: 700 !important; }}

        .hero-banner {{
            background: linear-gradient(120deg, {APP_ORANGE} 0%, {APP_GOLD} 55%, {APP_GREEN} 100%);
            border-radius: 20px;
            padding: 22px 28px;
            margin-bottom: 18px;
            box-shadow: 0 6px 18px rgba(0,0,0,0.15);
        }}
        .hero-banner h1 {{ font-family: 'Poppins', sans-serif; color: #FFFFFF !important; margin: 0; font-size: 2.1rem; }}
        .hero-banner p {{ color: #FFF8E1 !important; margin: 4px 0 0 0; font-size: 0.95rem; }}

        .verdict-card {{
            border-radius: 16px;
            padding: 16px 12px;
            background-color: {APP_SURFACE};
            border: 2px solid {APP_BORDER};
            box-shadow: 0 2px 10px rgba(62,39,35,0.08);
            text-align: center;
            margin-bottom: 10px;
            transition: transform 0.15s ease;
        }}
        .verdict-card:hover {{ transform: translateY(-2px); }}
        .verdict-card.correct {{ border-color: #43A047; background-color: #F1F8E9; }}
        .verdict-card.incorrect {{ border-color: #E53935; background-color: #FFEBEE; }}
        .verdict-card .model-name {{ font-weight: 700; margin-bottom: 6px; text-transform: uppercase; font-size: 0.8rem; letter-spacing: 0.04em; }}
        .verdict-card .pred-label {{ font-size: 1.3rem; font-weight: 700; margin: 2px 0; color: #2B2B2B !important; }}
        .verdict-card .meta {{ color: {APP_MUTED_TEXT} !important; font-size: 0.85rem; margin: 2px 0; }}

        .leaderboard-list {{ display: flex; flex-direction: column; gap: 6px; }}
        .leaderboard-row {{
            background-color: {APP_SURFACE};
            border-left: 4px solid {APP_ORANGE};
            border-radius: 8px;
            padding: 6px 10px;
            display: flex;
            flex-wrap: wrap;
            align-items: baseline;
            gap: 6px 10px;
            box-shadow: 0 1px 4px rgba(62,39,35,0.06);
        }}
        .leaderboard-row .lb-rank {{ font-size: 0.95rem; }}
        .leaderboard-row .lb-name {{ font-weight: 700 !important; font-size: 0.9rem; }}
        .leaderboard-row .lb-acc {{ font-weight: 700 !important; margin-left: auto; }}
        .leaderboard-row .lb-meta {{ width: 100%; color: {APP_MUTED_TEXT} !important; font-size: 0.75rem; }}

        .trust-row {{ padding: 8px 10px; border-radius: 10px; background-color: #FFF8E1; margin-bottom: 6px; }}
        .trust-row * {{ color: {APP_TEXT} !important; }}
        .trust-row-header {{ display: flex; justify-content: space-between; margin-bottom: 4px; }}
        .trust-bar-track {{ background-color: #F0E1C6; border-radius: 6px; height: 10px; overflow: hidden; }}
        .trust-bar-fill {{ height: 100%; border-radius: 6px; transition: width 0.3s ease; }}

        .agreement-track {{ display: flex; gap: 6px; margin: 8px 0; }}
        .agreement-seg {{ flex: 1; height: 22px; border-radius: 6px; }}

        .footer-disclaimer {{
            color: #9E9E9E !important; font-size: 0.8rem; text-align: center;
            margin-top: 2.5rem; border-top: 1px solid #EEE; padding-top: 0.8rem;
        }}
        .footer-disclaimer * {{ color: #9E9E9E !important; }}
        </style>
        <div class="hero-banner">
            <h1>🍊 Citrus Leaf Doctor</h1>
            <p>Upload a citrus leaf photo to compare live predictions from four trained models, side by side.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


# --------------------------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------------------------

def render_sidebar(metrics_df):
    st.sidebar.header("📤 Upload")
    uploaded_file = st.sidebar.file_uploader("Citrus leaf photo", type=["jpg", "jpeg", "png"])
    true_class = st.sidebar.selectbox("I know the true class (optional)", TRUE_CLASS_OPTIONS, index=0)

    st.sidebar.markdown("---")
    st.sidebar.header("🏆 Leaderboard")
    st.sidebar.caption("From this project's held-out test-set evaluation.")

    leaderboard = metrics_df.copy()
    leaderboard["_acc_pct"] = leaderboard["test_accuracy"].apply(to_accuracy_percent)
    leaderboard = leaderboard.sort_values("_acc_pct", ascending=False)

    rank_medals = ["🥇", "🥈", "🥉", "4️⃣"]
    row_chunks = []
    for i, (_, row) in enumerate(leaderboard.iterrows()):
        arch_key = row["architecture"]
        medal = rank_medals[i] if i < len(rank_medals) else f"{i + 1}."
        color = MODEL_COLORS.get(arch_key, APP_MUTED_TEXT)
        # Built on one line with no embedded newlines/indentation. A multi-line indented
        # f-string here produces a whitespace-only line between concatenated rows, which
        # Streamlit's Markdown parser reads as a blank line ending the raw-HTML block; the
        # next row then starts with 4+ leading spaces, which Markdown treats as a code
        # block instead of HTML. That's why only the first row used to render as a styled
        # card and the rest fell back to showing literal escaped tag text.
        row_chunks.append(
            f'<div class="leaderboard-row" style="border-left-color:{color};">'
            f'<span class="lb-rank">{medal}</span>'
            f'<span class="lb-name" style="color:{color} !important;">{ARCH_DISPLAY_NAMES.get(arch_key, arch_key)}</span>'
            f'<span class="lb-acc">{row["_acc_pct"]:.2f}%</span>'
            f'<span class="lb-meta">{format_param_count(row["param_count"])} params · {row["mean_inference_latency_ms"]:.2f} ms</span>'
            f"</div>"
        )
    st.sidebar.markdown(f'<div class="leaderboard-list">{"".join(row_chunks)}</div>', unsafe_allow_html=True)

    return uploaded_file, true_class


def render_session_tally():
    tally = st.session_state.tally
    if sum(v["total"] for v in tally.values()) == 0:
        return
    st.sidebar.markdown("---")
    st.sidebar.header("📊 Session tally")
    row_chunks = []
    for arch_key in ARCH_ORDER:
        t = tally[arch_key]
        color = MODEL_COLORS[arch_key]
        acc_str = f"{100 * t['correct'] / t['total']:.1f}% correct" if t["total"] > 0 else "no labels yet"
        # Same single-line construction as the leaderboard above, and for the same reason.
        row_chunks.append(
            f'<div class="leaderboard-row" style="border-left-color:{color};">'
            f'<span class="lb-name" style="color:{color} !important;">{ARCH_DISPLAY_NAMES[arch_key]}</span>'
            f'<span class="lb-acc">{t["correct"]}/{t["total"]}</span>'
            f'<span class="lb-meta">{acc_str}</span>'
            f"</div>"
        )
    st.sidebar.markdown(f'<div class="leaderboard-list">{"".join(row_chunks)}</div>', unsafe_allow_html=True)


# --------------------------------------------------------------------------------------
# Main-area sections
# --------------------------------------------------------------------------------------

def render_verdict_cards(results, true_class):
    true_internal = TRUE_CLASS_TO_INTERNAL.get(true_class)
    cols = st.columns(4)
    for col, arch_key in zip(cols, ARCH_ORDER):
        r = results[arch_key]
        accent = MODEL_COLORS[arch_key]
        card_class = "verdict-card"
        if true_internal is not None:
            card_class += " correct" if r["pred_label"] == true_internal else " incorrect"
        with col:
            st.markdown(
                f"""
                <div class="{card_class}" style="border-top: 6px solid {accent};">
                    <div class="model-name" style="color:{accent} !important;">{ARCH_DISPLAY_NAMES[arch_key]}</div>
                    <div class="pred-label">{CLASS_ICONS[r['pred_label']]} {r['pred_display']}</div>
                    <div class="meta">Confidence: {r['confidence'] * 100:.1f}%</div>
                    <div class="meta">{r['latency_ms']:.2f} ms</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def update_session_tally(results, true_class, upload_id):
    if true_class == "Unknown":
        return
    # Streamlit reruns this whole script on every widget interaction, so without this
    # guard the same upload would get tallied again each time e.g. a sidebar widget changes.
    tally_key = f"{upload_id}::{true_class}"
    if tally_key in st.session_state.tallied_keys:
        return
    true_internal = TRUE_CLASS_TO_INTERNAL[true_class]
    for arch_key in ARCH_ORDER:
        st.session_state.tally[arch_key]["total"] += 1
        if results[arch_key]["pred_label"] == true_internal:
            st.session_state.tally[arch_key]["correct"] += 1
    st.session_state.tallied_keys.add(tally_key)


def compute_ensemble(results):
    probs_stack = np.stack([results[a]["probs"] for a in ARCH_ORDER], axis=0)
    soft_vote_idx = int(np.argmax(probs_stack.mean(axis=0)))

    votes = [results[a]["pred_idx"] for a in ARCH_ORDER]
    vote_counts = Counter(votes)
    max_count = max(vote_counts.values())
    # Tie-break: if more than one class received the maximum number of votes, there is no
    # single majority winner — report the tie honestly rather than picking one arbitrarily.
    top_classes = [c for c, n in vote_counts.items() if n == max_count]
    if len(top_classes) == 1:
        hard_vote_idx = top_classes[0]
        hard_vote_label = DISPLAY_NAMES[IDX_TO_CLASS[hard_vote_idx]]
        agreement_label = f"{max_count}/4 models agree — {hard_vote_label}"
        majority_idx = hard_vote_idx
    else:
        tied_names = ", ".join(DISPLAY_NAMES[IDX_TO_CLASS[c]] for c in top_classes)
        hard_vote_label = f"Tie ({tied_names})"
        agreement_label = "Split decision — no majority"
        majority_idx = None

    return {
        "soft_vote_idx": soft_vote_idx,
        "soft_vote_label": DISPLAY_NAMES[IDX_TO_CLASS[soft_vote_idx]],
        "hard_vote_label": hard_vote_label,
        "agreement_label": agreement_label,
        "majority_idx": majority_idx,
        "agree_count": max_count,
    }


def render_ensemble_panel(results, ensemble):
    st.subheader("🧭 Ensemble verdict")
    c1, c2 = st.columns(2)
    c1.metric("Soft vote (mean probability)", ensemble["soft_vote_label"])
    c2.metric("Hard vote (majority prediction)", ensemble["hard_vote_label"])

    st.markdown(f"**{ensemble['agreement_label']}**")
    segs = ""
    for arch_key in ARCH_ORDER:
        agrees = ensemble["majority_idx"] is not None and results[arch_key]["pred_idx"] == ensemble["majority_idx"]
        color = "#43A047" if agrees else "#E0E0E0"
        segs += f'<div class="agreement-seg" style="background-color:{color};"></div>'
    st.markdown(f'<div class="agreement-track">{segs}</div>', unsafe_allow_html=True)


def render_radar_chart(results):
    st.subheader("📈 Confidence by class")
    categories = [DISPLAY_NAMES[c] for c in CLASS_ORDER]
    n = len(categories)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(5.5, 5.5), subplot_kw={"projection": "polar"})
    apply_matplotlib_theme(fig, ax)
    ax.grid(color=APP_BORDER, alpha=0.8)

    for arch_key in ARCH_ORDER:
        values = results[arch_key]["probs"].tolist()
        values += values[:1]
        color = MODEL_COLORS[arch_key]
        ax.plot(angles, values, color=color, linewidth=2, label=ARCH_DISPLAY_NAMES[arch_key])
        ax.fill(angles, values, color=color, alpha=0.15)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, color=APP_TEXT, fontsize=10)
    ax.tick_params(axis="y", colors=APP_MUTED_TEXT)
    ax.set_ylim(0, 1)
    ax.set_title("Predicted class probabilities", pad=20, color=APP_GREEN, fontweight="bold")

    legend = ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.15), fontsize=8, frameon=True)
    legend.get_frame().set_facecolor(APP_SURFACE)
    legend.get_frame().set_edgecolor(APP_BORDER)
    for text in legend.get_texts():
        text.set_color(APP_TEXT)

    st.pyplot(fig)
    plt.close(fig)


def render_gradcam_grid(image, results):
    st.subheader("🔥 Grad-CAM: where each model is looking")
    image_np = np.array(image.resize((224, 224)))

    fig, axes = plt.subplots(2, 2, figsize=(9, 9))
    apply_matplotlib_theme(fig, axes)

    for ax, arch_key in zip(axes.flat, ARCH_ORDER):
        r = results[arch_key]
        heatmap_u8 = np.uint8(255 * r["heatmap"])
        heatmap_color = cv2.applyColorMap(heatmap_u8, cv2.COLORMAP_JET)
        heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
        overlay = cv2.addWeighted(image_np, 0.6, heatmap_color, 0.4, 0)
        ax.imshow(overlay)
        ax.set_title(
            f"{ARCH_DISPLAY_NAMES[arch_key]}\n{r['pred_display']} ({r['confidence'] * 100:.1f}%)",
            fontsize=10, color=APP_TEXT, fontweight="bold",
        )
        ax.axis("off")

    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def render_trust_index(results, metrics_df):
    st.subheader("🏆 Model trust index")
    metrics_by_arch = metrics_df.set_index("architecture")
    rows = []
    for arch_key in ARCH_ORDER:
        test_acc_fraction = to_accuracy_percent(metrics_by_arch.loc[arch_key, "test_accuracy"]) / 100
        trust_pct = results[arch_key]["confidence"] * test_acc_fraction * 100
        rows.append((arch_key, trust_pct))
    rows.sort(key=lambda x: x[1], reverse=True)

    max_trust = max(trust for _, trust in rows) or 1.0
    for arch_key, trust_pct in rows:
        color = MODEL_COLORS[arch_key]
        bar_width = max(4.0, (trust_pct / max_trust) * 100)
        st.markdown(
            f"""
            <div class="trust-row">
                <div class="trust-row-header">
                    <span style="color:{color} !important; font-weight:700;">{ARCH_DISPLAY_NAMES[arch_key]}</span>
                    <span style="font-weight:700;">{trust_pct:.1f}%</span>
                </div>
                <div class="trust-bar-track">
                    <div class="trust-bar-fill" style="width:{bar_width:.1f}%; background-color:{color};"></div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    st.caption(
        "Trust score = this image's confidence × that model's own test-set accuracy. "
        "It's a simple, illustrative heuristic for this one prediction — not a rigorous metric."
    )


def render_download_section(image, uploaded_file, results, ensemble, true_class):
    st.subheader("💾 Save this run")
    if st.button("Save this run to disk"):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = RUNS_DIR / timestamp
        run_dir.mkdir(parents=True, exist_ok=True)

        image.save(run_dir / uploaded_file.name)

        run_data = {
            "filename": uploaded_file.name,
            "timestamp": timestamp,
            "true_class": TRUE_CLASS_TO_INTERNAL.get(true_class),
            "ensemble": {
                "soft_vote": ensemble["soft_vote_label"],
                "hard_vote": ensemble["hard_vote_label"],
                "agreement": ensemble["agreement_label"],
            },
            "models": {
                arch_key: {
                    "predicted_class": results[arch_key]["pred_label"],
                    "confidence": results[arch_key]["confidence"],
                    "latency_ms": results[arch_key]["latency_ms"],
                }
                for arch_key in ARCH_ORDER
            },
        }
        with open(run_dir / "run.json", "w") as f:
            json.dump(run_data, f, indent=2)

        st.success(f"Saved to {run_dir}")


def render_footer():
    st.markdown(
        """
        <div class="footer-disclaimer">
        This is a course-project demonstration tool trained on a specific dataset and split —
        not a validated diagnostic system. Treat predictions on photos very different from the
        training distribution (different lighting, camera, background, or a plant part not
        represented in training) with appropriate skepticism.
        </div>
        """,
        unsafe_allow_html=True,
    )


# --------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------

def main():
    setup_page()

    if "tally" not in st.session_state:
        st.session_state.tally = {arch: {"correct": 0, "total": 0} for arch in ARCH_ORDER}
    if "tallied_keys" not in st.session_state:
        st.session_state.tallied_keys = set()

    with st.spinner("Loading models..."):
        models, device = load_all_models()
        metrics_df = load_metrics()
        transforms = build_transforms()

    uploaded_file, true_class = render_sidebar(metrics_df)
    render_session_tally()

    if uploaded_file is None:
        st.info("Upload a citrus leaf photo from the sidebar to run all four models.")
        render_footer()
        return

    image = Image.open(uploaded_file).convert("RGB")

    with st.spinner("Running all four models..."):
        results = {}
        for arch_key in ARCH_ORDER:
            info = models[arch_key]
            results[arch_key] = run_model(
                info["model"], info["target_layer"], image, transforms[arch_key], device
            )

    ensemble = compute_ensemble(results)

    annotated = annotate_image_with_verdict(image, ensemble, results)
    st.image(annotated, width=420)
    st.caption(f"Image dimensions: {image.size[0]} × {image.size[1]} px")

    upload_id = f"{uploaded_file.name}_{uploaded_file.size}"
    update_session_tally(results, true_class, upload_id)

    render_verdict_cards(results, true_class)

    tab_ensemble, tab_gradcam, tab_trust, tab_save = st.tabs(
        ["🧭 Ensemble & Confidence", "🔥 Grad-CAM", "🏆 Trust Index", "💾 Save Run"]
    )
    with tab_ensemble:
        render_ensemble_panel(results, ensemble)
        render_radar_chart(results)
    with tab_gradcam:
        render_gradcam_grid(image, results)
    with tab_trust:
        render_trust_index(results, metrics_df)
    with tab_save:
        render_download_section(image, uploaded_file, results, ensemble, true_class)

    render_footer()


if __name__ == "__main__":
    main()