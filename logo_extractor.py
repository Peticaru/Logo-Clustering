from __future__ import annotations

import numpy as np
import cv2
from PIL import Image, ImageOps
from typing import Optional, Tuple


class LogoExtractor:
    def __init__(self, out_size: int = 256, pad_ratio: float = 0.12):
        self.out_size = out_size
        self.pad_ratio = pad_ratio

    def extract_best(self, original: Image.Image, normalized: Image.Image, debug: bool = False) -> Image.Image:
        result_from_orig = self.extract(original)
        result_from_norm = self.extract(normalized)

        score_orig = self._quality_score(result_from_orig)
        score_norm = self._quality_score(result_from_norm)


        return result_from_orig if score_orig >= score_norm else result_from_norm

    def _quality_score(self, image: Image.Image) -> float:

        gray_img = cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2GRAY)
        _, bin_mask = cv2.threshold(gray_img, 245, 255, cv2.THRESH_BINARY_INV)
        return float((bin_mask > 0).mean())
    
    def to_rgb(self, img: Image.Image) -> np.ndarray:
        arr = np.array(img.convert("RGBA")).astype(np.float32)
        rgb = arr[..., :3]
        a = arr[..., 3:4] / 255.0
        out = rgb * a + 255.0 * (1.0 - a)
        return out.astype(np.uint8)
    
    def extract(self, img: Image.Image) -> Image.Image:
        rgb = self.to_rgb(img)
        h, w = rgb.shape[:2]

        mask = self.build_mask(rgb)

        fg_ratio = (mask > 0).mean()
        if fg_ratio < 0.001:
            return self.resize(rgb)

        bbox = self.largest_component(mask, w, h)
        if bbox is None:
            bbox = self.bbox(mask)
            if bbox is None:
                return self.resize(rgb)

        x0, y0, x1, y1 = bbox
        cropped = rgb[y0:y1 + 1, x0:x1 + 1]
        return self.resize(cropped)



    def build_mask(self, rgb: np.ndarray) -> np.ndarray:
        border_pixels = np.concatenate([rgb[0], rgb[-1], rgb[:, 0], rgb[:, -1]], axis=0)
        estimated_bg = np.median(border_pixels, axis=0).astype(np.int16)

        diff_map = np.abs(rgb.astype(np.int16) - estimated_bg[None, None, :]).sum(axis=2)

        border_diff_vals = np.abs(border_pixels.astype(np.int16) - estimated_bg[None, :]).sum(axis=1)
        auto_thr = int(max(30, np.median(border_diff_vals) * 2))

        mask = (diff_map > auto_thr).astype(np.uint8) * 255

        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
        return mask

    def largest_component(self, mask: np.ndarray, w: int, h: int) -> Optional[Tuple[int, int, int, int]]:
        count, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)

        if count <= 1:
            return None  

        largest_idx = 1 + np.argmax(stats[1:, 4])
        x, y, box_w, box_h, _ = stats[largest_idx]

        pad = int(self.pad_ratio * max(box_w, box_h))
        x0 = max(0, x - pad)
        y0 = max(0, y - pad)
        x1 = min(w - 1, x + box_w + pad)
        y1 = min(h - 1, y + box_h + pad)

        return (x0, y0, x1, y1)

    def bbox(self, mask: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        ys, xs = np.where(mask > 0)
        if not len(xs):
            return None
        return (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))

    def resize(self, rgb: np.ndarray) -> Image.Image:
        pil = Image.fromarray(rgb)
        w, h = pil.size
        side = max(w, h)

        pad_left = (side - w) // 2
        pad_top = (side - h) // 2
        pad_right = side - w - pad_left
        pad_bottom = side - h - pad_top

        padded = ImageOps.expand(pil, (pad_left, pad_top, pad_right, pad_bottom), fill=(255, 255, 255))
        return padded.resize((self.out_size, self.out_size), Image.Resampling.LANCZOS)
