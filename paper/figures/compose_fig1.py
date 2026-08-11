#!/usr/bin/env python
"""Crop and compose the two ChimeraX panels into Figure 1.

ChimeraX frames to the window, so both panels arrive small and off-centre in a
lot of white. Cropping here rather than re-rendering keeps the camera identical
between panels, which is the property the figure depends on: both are cropped to
one shared box, so any difference the reader sees is the protein and not the
framing.

  python paper/figures/compose_fig1.py
"""
from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
PANELS = [("panel_a.png", "a", "apo (2PKF)"),
          ("panel_b.png", "b", "holo (2PKK), 2-fluoroadenosine bound")]
OUT = HERE / "fig1_adenosine_kinase.png"

#: White space kept around the union of both silhouettes.
PAD = 40
#: Gap between panels.
GUTTER = 60


def _bbox(im: Image.Image) -> tuple[int, int, int, int]:
    """Bounding box of everything that is not background.

    Alpha when the PNG has it, since the panels are saved with a transparent
    background; luminance otherwise. Thresholding on brightness alone would clip
    the holo panel, whose surface is rendered translucent and so sits much closer
    to white than the apo panel's.
    """
    if im.mode == "RGBA":
        alpha = im.getchannel("A")
        box = alpha.getbbox()
        if box:
            return box
    grey = im.convert("L")
    mask = grey.point(lambda v: 255 if v < 250 else 0)
    return mask.getbbox() or (0, 0, im.width, im.height)


def _font(size: int):
    for name in ("arialbd.ttf", "arial.ttf", "DejaVuSans-Bold.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def main() -> None:
    images = []
    for fname, _, _ in PANELS:
        p = HERE / fname
        if not p.exists():
            raise SystemExit(f"missing {p}; run the ChimeraX script first")
        images.append(Image.open(p).convert("RGBA"))

    # One shared crop box, so the panels stay directly comparable.
    boxes = [_bbox(im) for im in images]
    left = max(min(b[0] for b in boxes) - PAD, 0)
    top = max(min(b[1] for b in boxes) - PAD, 0)
    right = min(max(b[2] for b in boxes) + PAD, min(im.width for im in images))
    bottom = min(max(b[3] for b in boxes) + PAD, min(im.height for im in images))
    cropped = [im.crop((left, top, right, bottom)) for im in images]

    w, h = cropped[0].size
    label_h = int(h * 0.09)
    canvas = Image.new("RGBA", (w * 2 + GUTTER, h + label_h), (255, 255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    tag_font = _font(int(label_h * 0.62))
    cap_font = _font(int(label_h * 0.36))

    for i, (im, (_, tag, caption)) in enumerate(zip(cropped, PANELS)):
        x = i * (w + GUTTER)
        canvas.alpha_composite(im, (x, label_h))
        draw.text((x + 8, 2), tag, font=tag_font, fill=(20, 20, 20, 255))
        draw.text((x + 8 + int(label_h * 0.75), int(label_h * 0.30)),
                  caption, font=cap_font, fill=(70, 70, 70, 255))

    canvas.convert("RGB").save(OUT, dpi=(300, 300))
    print(f"wrote {OUT}  ({canvas.width}x{canvas.height}, "
          f"cropped from {images[0].width}x{images[0].height})")


if __name__ == "__main__":
    main()
