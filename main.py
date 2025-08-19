import streamlit as st
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F

# ---------------- Model (unchanged) ----------------
def gn(c): return nn.GroupNorm(8 if c >= 8 else 1, c)

class DSConv(nn.Module):
    def __init__(self, in_ch, out_ch, s=1, p=1):
        super().__init__()
        self.dw = nn.Conv2d(in_ch, in_ch, 3, stride=s, padding=p, groups=in_ch, bias=False)
        self.dw_gn = gn(in_ch)
        self.pw = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        self.pw_gn = gn(out_ch)
        self.act = nn.SiLU(inplace=True)
    def forward(self, x):
        x = self.act(self.dw_gn(self.dw(x)))
        x = self.act(self.pw_gn(self.pw(x)))
        return x

class DSBlock(nn.Module):
    def __init__(self, in_ch, out_ch, p_drop=0.0):
        super().__init__()
        self.conv1 = DSConv(in_ch, out_ch)
        self.conv2 = DSConv(out_ch, out_ch)
        self.drop  = nn.Dropout2d(p_drop) if p_drop > 0 else nn.Identity()
        self.skip  = (nn.Identity() if in_ch == out_ch
                      else nn.Sequential(nn.Conv2d(in_ch, out_ch, 1, bias=False), gn(out_ch)))
    def forward(self, x):
        h = self.conv1(x)
        h = self.drop(self.conv2(h))
        return F.silu(h + self.skip(x), inplace=True)

class UpLite(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch, p_drop=0.0):
        super().__init__()
        self.reduce = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        self.bn     = gn(out_ch)
        self.act    = nn.SiLU(inplace=True)
        self.fuse   = DSBlock(out_ch + skip_ch, out_ch, p_drop=p_drop)
    def forward(self, x, skip):
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        x = self.act(self.bn(self.reduce(x)))
        x = torch.cat([x, skip], dim=1)
        return self.fuse(x)

class LiteUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, base=32, p_drop=0.05):
        super().__init__()
        self.enc1 = DSBlock(in_channels, base, p_drop=0.0)
        self.pool = nn.MaxPool2d(2)
        self.enc2 = DSBlock(base,   base*2, p_drop=p_drop)
        self.enc3 = DSBlock(base*2, base*4, p_drop=p_drop)
        self.enc4 = DSBlock(base*4, base*8, p_drop=p_drop)
        self.bot  = DSBlock(base*8, base*16, p_drop=p_drop)
        self.up4 = UpLite(base*16, base*8,  base*8,  p_drop=p_drop)
        self.up3 = UpLite(base*8,  base*4,  base*4,  p_drop=p_drop)
        self.up2 = UpLite(base*4,  base*2,  base*2,  p_drop=0.0)
        self.up1 = UpLite(base*2,  base,    base,    p_drop=0.0)
        self.head = nn.Conv2d(base, out_channels, 1)
    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b  = self.bot(self.pool(e4))
        d4 = self.up4(b, e4)
        d3 = self.up3(d4, e3)
        d2 = self.up2(d3, e2)
        d1 = self.up1(d2, e1)
        return self.head(d1)  # logits

# ---------------- App config ----------------
IMG_SIZE = 512   # <-- set to exactly what you trained with (256/384/512)
MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]
MODEL_PATH = "best_model.pth"
IN_CHANNELS = 3
OUT_CHANNELS = 1
BASE = 32
P_DROP = 0.05

def preprocess_image(image: Image.Image):
    image = image.convert("RGB")  # enforce RGB
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD),
    ])
    x = transform(image).unsqueeze(0)   # [1,3,H,W]
    return x

@st.cache_resource(show_spinner=False)
def load_model(model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LiteUNet(in_channels=IN_CHANNELS, out_channels=OUT_CHANNELS, base=BASE, p_drop=P_DROP)
    # If your checkpoint was saved with model.state_dict(), this is correct:
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.to(device).eval()
    return model, device

def predict_prob(model, x, device):
    with torch.no_grad():
        x = x.to(device)
        logits = model(x)                # [1,1,H,W]
        prob = torch.sigmoid(logits)     # [1,1,H,W] in [0,1]
    return prob.squeeze(0).squeeze(0).cpu().numpy(), logits.detach().float().cpu().numpy()

def main():
    st.title("Retinal Vessel Segmentation (Lite U-Net)")
    st.caption("Tip: if the mask looks blank, lower the threshold to ~0.30–0.45.")

    # Sidebar controls
    thr = st.sidebar.slider("Segmentation threshold", 0.0, 1.0, 0.45, 0.01)
    show_heatmap = st.sidebar.checkbox("Show probability heatmap", value=True)
    st.sidebar.write(f"Model: base={BASE}, img={IMG_SIZE}x{IMG_SIZE}")

    # Load model once
    try:
        model, device = load_model(MODEL_PATH)
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return

    uploaded_file = st.file_uploader("Upload a retinal image...", type=["jpg", "jpeg", "png", "tif", "tiff"])
    if uploaded_file is not None:
        img = Image.open(uploaded_file)
        st.image(img, caption="Original", use_container_width=True)

        x = preprocess_image(img)
        prob, logits = predict_prob(model, x, device)  # prob: [H,W]

        # Diagnostics
        st.write(f"**Prob stats** — min: {prob.min():.4f} | max: {prob.max():.4f} | mean: {prob.mean():.4f}")
        if prob.max() < 0.1:
            st.warning("Very low probabilities. Try lowering the threshold (e.g., 0.30) or check preprocessing/IMG_SIZE matches training.")

        # Thresholded mask
        mask = (prob >= thr).astype(np.float32)  # 0/1 float for display
        col1, col2 = st.columns(2)
        with col1:
            st.image(mask, caption=f"Binary mask @ thr={thr:.2f}", use_container_width=True, clamp=True)
        with col2:
            if show_heatmap:
                st.image(prob, caption="Probability heatmap", use_container_width=True, clamp=True)

        # Optionally overlay
        if st.checkbox("Show overlay"):
            import cv2
            img_resized = img.convert("RGB").resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
            base_im = np.array(img_resized, dtype=np.float32) / 255.0
            overlay = base_im.copy()
            overlay[..., 1] = np.maximum(overlay[..., 1], mask)  # add green where vessels
            overlay[..., 2] = np.maximum(overlay[..., 2], mask*0.4)
            blend = (0.6*base_im + 0.4*overlay).clip(0,1)
            st.image(blend, caption="Overlay", use_container_width=True, clamp=True)

if __name__ == "__main__":
    main()
