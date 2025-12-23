import os
import re
from io import BytesIO

import cv2
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import requests
import torch
from PIL import Image, ImageOps
from transformers import AutoImageProcessor, AutoModel

from logo_extractor import LogoExtractor
# ---------------- CONFIG ----------------
PARQUET_PATH = "logos.snappy.parquet"
OUT_FILE = "logo_dinov2_embeddings.parquet"
LOGO_DIR = "logo_images"

MAX_DOMAINS = 200
OUT_SIZE = 512
LOGODEV_PUBLIC_KEY = "pk_TD0uWQxyTyqHex-UjA4WmQ"

device = "cuda" if torch.cuda.is_available() else "cpu"
 # deterministic

MODEL_NAME = "facebook/dinov2-base"
processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME).to(device)
model.eval()

os.makedirs(LOGO_DIR, exist_ok=True)

# ---------------- UTILS ----------------
def safe(s):
    return re.sub(r"[^a-zA-Z0-9._-]", "_", s)

def download_logo(domain):
    url = f"https://img.logo.dev/{domain}?token={LOGODEV_PUBLIC_KEY}&size=512&format=png"
    r = requests.get(url, timeout=12)
    if r.status_code == 200 and "image" in r.headers.get("Content-Type", ""):
        return r.content
    return None

import numpy as np
import cv2
from PIL import Image, ImageOps

from PIL import ImageOps
import numpy as np

from rembg import remove
from PIL import Image
from io import BytesIO
from logo_extractor import LogoExtractor

def remove_bg_bytes(image_bytes: bytes) -> Image.Image:
    # rembg returns PNG bytes with alpha
    out_png_bytes = remove(image_bytes)
    return Image.open(BytesIO(out_png_bytes)).convert("RGBA")

def rgba_to_white_rgb(img_rgba: Image.Image) -> Image.Image:
    bg = Image.new("RGBA", img_rgba.size, (255,255,255,255))
    return Image.alpha_composite(bg, img_rgba).convert("RGB")

def crop_to_alpha(img_rgba: Image.Image, alpha_thr=10) -> Image.Image:
    a = np.array(img_rgba.split()[-1])  # alpha channel
    ys, xs = np.where(a > alpha_thr)
    if len(xs) == 0:
        return img_rgba
    return img_rgba.crop((xs.min(), ys.min(), xs.max()+1, ys.max()+1))

def pad_square_resize(img_rgb: Image.Image, out_size=256) -> Image.Image:
    w, h = img_rgb.size
    s = max(w, h)
    img_rgb = ImageOps.expand(
        img_rgb,
        ((s-w)//2, (s-h)//2, s-w-(s-w)//2, s-h-(s-h)//2),
        fill=(255,255,255)
    )
    return img_rgb.resize((out_size, out_size), Image.Resampling.LANCZOS)

def normalize_logo(bytes_img):
    img = Image.open(BytesIO(bytes_img)).convert("RGBA")
    bg = Image.new("RGBA", img.size, (255,255,255,255))
    img = Image.alpha_composite(bg, img).convert("RGB")

    g = np.array(img.convert("L"))
    mask = g < 245
    if mask.any():
        ys, xs = np.where(mask)
        img = img.crop((xs.min(), ys.min(), xs.max()+1, ys.max()+1))

    w, h = img.size
    s = max(w, h)
    img = ImageOps.expand(img,
        ((s-w)//2, (s-h)//2, s-w-(s-w)//2, s-h-(s-h)//2),
        fill=(255,255,255)
    )

    return img.resize((256, 256), Image.Resampling.LANCZOS)


def np_to_pil_rgb(arr: np.ndarray) -> Image.Image:
    if arr.dtype != np.uint8:
        arr = arr.astype(np.uint8)
    return Image.fromarray(arr, mode="RGB")


def embed(pil_img):
    inputs = processor(images=pil_img, return_tensors="pt").to(device)
    with torch.no_grad():
        e = model(**inputs).last_hidden_state[:,0]
    e = e / e.norm(dim=-1, keepdim=True)
    return e.squeeze(0).cpu().numpy()

# ---------------- MAIN ----------------
df = pq.read_table(PARQUET_PATH).to_pandas()

records = {}
index = 0
cnt = 0

import io
LogoExtractor = LogoExtractor()

for domain in df["domain"][:MAX_DOMAINS]:
    index += 1
    if (index % 20  == 0):
        print (f"[{index}/{min(len(df), MAX_DOMAINS)}] Processing {domain}...")
    img_bytes = download_logo(domain)
    if img_bytes is None:
        continue
    img = Image.open(BytesIO(img_bytes)).convert("RGBA")

    norm = normalize_logo(img_bytes)
    icon = LogoExtractor.extract_best(img, remove_bg_bytes(img_bytes))


    emb_global = embed(norm)
    emb_icon = embed(icon)
    #save normalized and icon images for aamco for debugging
    
    records[domain] = {
        "emb_global": emb_global.tolist(),
        "emb_icon": emb_icon.tolist(),
        "image_bytes": img_bytes
    }

   

pd.DataFrame.from_dict(records, orient="index").to_parquet(OUT_FILE)
print(f"Saved {len(records)} DINOv2 global+icon embeddings → {OUT_FILE}")
