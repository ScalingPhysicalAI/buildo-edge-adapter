"""Create horizontally mirrored copies of the office test scene.

Mirroring flips the furniture layout left<->right, so an edge model that
truly reads the current image must produce a mirrored trajectory for the
same (mirrored) goal. Used by the mirror_left demo case in the README.
"""
import os

from PIL import Image

HOME = os.path.expanduser("~/edgevla")
for n in ("past", "cur"):
    img = Image.open(f"{HOME}/AsyncVLA/inference/{n}.png")
    img.transpose(Image.FLIP_LEFT_RIGHT).save(f"{HOME}/{n}_mirror.png")
print("MIRRORED")
