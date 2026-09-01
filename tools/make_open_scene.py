"""Synthesize an empty, obstacle-free indoor scene in the style of the demo
frames (carpet floor, plain walls, ceiling lights, fisheye vignette).
'cur' simulates one step forward via a slight center zoom of 'past'."""
import os

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

W, H = 552, 444
HORIZON = int(H * 0.42)
rng = np.random.default_rng(0)

img = np.zeros((H, W, 3), dtype=np.float32)

# Ceiling: light gray gradient getting brighter toward horizon
for y in range(HORIZON):
    img[y, :] = 190 + 25 * (y / HORIZON)

# Floor: gray-brown carpet with speckle noise, darker near the horizon
for y in range(HORIZON, H):
    depth = (y - HORIZON) / (H - HORIZON)   # 0 far, 1 near
    base = 95 + 35 * depth
    img[y, :] = base
carpet_noise = rng.normal(0, 9, (H - HORIZON, W, 1)).repeat(3, axis=2)
img[HORIZON:] += carpet_noise
img[HORIZON:, :, 2] *= 0.94  # slight warm tint like the demo carpet

pil = Image.fromarray(np.clip(img, 0, 255).astype(np.uint8))
draw = ImageDraw.Draw(pil)

# Distant back wall: a light band just above the horizon
draw.rectangle([0, HORIZON - 26, W, HORIZON], fill=(215, 213, 208))

# Ceiling light strips
for cx in (W * 0.3, W * 0.7):
    draw.ellipse([cx - 40, 20, cx + 40, 42], fill=(245, 243, 235))

pil = pil.filter(ImageFilter.GaussianBlur(1.2))

# Fisheye-style dark circular vignette like the real frames
arr = np.asarray(pil).astype(np.float32)
yy, xx = np.mgrid[0:H, 0:W]
r = np.sqrt(((xx - W / 2) / (W / 2)) ** 2 + ((yy - H / 2) / (H / 2)) ** 2)
mask = np.clip((1.25 - r) / 0.18, 0, 1)[..., None]
arr *= mask
past = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
past.save(os.path.expanduser("~/edgevla/open_past.png"))

# 'cur' = one step forward: zoom into the center by ~8%
z = 0.92
box = (int(W * (1 - z) / 2), int(H * (1 - z) / 2),
       int(W * (1 + z) / 2), int(H * (1 + z) / 2))
cur = past.crop(box).resize((W, H), Image.BILINEAR)
cur.save(os.path.expanduser("~/edgevla/open_cur.png"))
print("OPEN_SCENE_OK")
