"""Printable generic labels (no brands, no real medicine) + the gold visible text.

Writes eval/labels/labels.pdf (US Letter, 300 dpi, actual size: print at 100 %)
and eval/labels/labels.json (label id -> every word printed on it), which the
eval uses as ground truth for "only visible text".
"""

from __future__ import annotations

import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

DPI = 300
CM = DPI / 2.54
PAGE = (int(8.5 * DPI), int(11 * DPI))
MARGIN = int(0.3 * DPI)
GAP = int(0.4 * CM)
FONTS = "/System/Library/Fonts/Supplemental/"
OUT = Path(__file__).resolve().parents[1] / "eval" / "labels"

# id, put it on, width cm, height cm, colour, title, big line, small lines
LABELS = [
    ("soup", "на круглую банку овсянки", 9.3, 12, "#c0392b", "TOMATO SOUP", "400 g",
     ["Heat 5 minutes", "Stir before serving", "Best before 12/2027"]),
    ("pasta", "на коробку пасты, спереди", 9.3, 12, "#1f4e9c", "ROTINI PASTA", "340 g",
     ["Boil 8 minutes", "6 servings", "Best before 03/2028"]),
    ("bread_mix", "на вторую коробку пасты", 9.3, 13, "#7a4b1e", "BANANA BREAD MIX", "Bake at 350°F",
     ["1. Heat oven to 350°F.", "2. Mix with 2 eggs, 1/3 cup oil", "   and 2/3 cup water.", "3. Pour into a greased pan.",
      "4. Bake 45 minutes.", "5. Cool 10 minutes before slicing.", "Makes 10 slices.", "Best before 01/2028"]),
    ("rice", "на пакет риса", 9.3, 12, "#2e7d5b", "WHITE RICE", "907 g",
     ["1 cup rice + 2 cups water", "Simmer 18 minutes", "Rest 5 minutes", "Best before 06/2028"]),
    ("soup_seasoning", "на пакетик приправы", 8, 10, "#e67e22", "SOUP SEASONING", "25 g",
     ["1 sachet for 2.5 L", "Best before 09/2027"]),
    ("sugar", "на баночку специй", 5, 5.5, "#555555", "SUGAR", "", []),
    ("salt", "на баночку специй", 5, 5.5, "#555555", "SALT", "", []),
    ("paprika", "на баночку специй", 5, 5.5, "#b03a2e", "PAPRIKA", "40 g", []),
    ("vitamin_d", "на флакон или коробку лекарства", 6, 7, "#6c3483", "VITAMIN D3", "1000 IU",
     ["60 softgels"]),
    ("sleep_aid", "на флакон или коробку лекарства", 6, 7, "#34495e", "SLEEP AID", "",
     ["Take 1 tablet", "at bedtime"]),
    ("pain_relief", "на коробку лекарства", 8, 7.5, "#1a5276", "PAIN RELIEF", "",
     ["Take 1 tablet every 6 hours.", "Do not take more than", "4 tablets in 24 hours.", "Keep away from children."]),
]


def font(size_px: int, bold: bool = True) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(FONTS + ("Arial Bold.ttf" if bold else "Arial.ttf"), size_px)


def fit(draw: ImageDraw.ImageDraw, text: str, max_w: int, start: int, bold: bool = True) -> ImageFont.FreeTypeFont:
    size = start
    while size > 10 and draw.textlength(text, font=font(size, bold)) > max_w:
        size -= 2
    return font(size, bold)


def render(label) -> Image.Image:
    _, _, w_cm, h_cm, colour, title, big, small = label
    w, h = int(w_cm * CM), int(h_cm * CM)
    img = Image.new("RGB", (w, h), "white")
    d = ImageDraw.Draw(img)
    pad = int(0.35 * CM)
    band = int(h * (0.30 if small else 0.45))
    d.rectangle((0, 0, w, band), fill=colour)
    f = fit(d, title, w - 2 * pad, int(band * 0.42))
    d.text((w / 2, band / 2), title, font=f, fill="white", anchor="mm")
    y = band + pad
    if big:
        f = fit(d, big, w - 2 * pad, int(h * 0.12))
        d.text((w / 2, y), big, font=f, fill="black", anchor="ma")
        y += f.size + pad
    if small:
        step = min(int(h * 0.075), (h - y - pad) // max(1, len(small)))
        f = font(max(14, int(step * 0.72)), bold=False)
        for line in small:
            d.text((pad, y), line, font=f, fill="black")
            y += step
    # dashed cut line
    for x in range(0, w, 24):
        d.line((x, 0, x + 12, 0), fill="#999", width=2)
        d.line((x, h - 2, x + 12, h - 2), fill="#999", width=2)
    for yy in range(0, h, 24):
        d.line((0, yy, 0, yy + 12), fill="#999", width=2)
        d.line((w - 2, yy, w - 2, yy + 12), fill="#999", width=2)
    return img


def pack(images: list[tuple[str, Image.Image]]) -> list[Image.Image]:
    pages, page = [], None
    x = y = row_h = 0
    for name, im in images:
        if page is None:
            page, x, y, row_h = Image.new("RGB", PAGE, "white"), MARGIN, MARGIN, 0
        if x + im.width > PAGE[0] - MARGIN:
            x, y, row_h = MARGIN, y + row_h + GAP, 0
        if y + im.height > PAGE[1] - MARGIN:
            pages.append(page)
            page, x, y, row_h = Image.new("RGB", PAGE, "white"), MARGIN, MARGIN, 0
        page.paste(im, (x, y))
        ImageDraw.Draw(page).text((x, y + im.height + 4), name, font=font(28, False), fill="#888")
        x += im.width + GAP
        row_h = max(row_h, im.height + 40)
    if page is not None:
        pages.append(page)
    return pages


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rendered = [(lab[1], render(lab)) for lab in LABELS]
    pages = pack(rendered)
    pages[0].save(OUT / "labels.pdf", save_all=True, append_images=pages[1:], resolution=DPI)
    gold = {lab[0]: {"put_on": lab[1], "text": " ".join([lab[5], lab[6], *lab[7]]).strip()} for lab in LABELS}
    (OUT / "labels.json").write_text(json.dumps(gold, indent=2, ensure_ascii=False))
    for i, p in enumerate(pages, 1):
        p.resize((p.width // 3, p.height // 3)).save(OUT / f"preview-{i}.png")
    print(len(LABELS), "labels on", len(pages), "pages ->", OUT)


if __name__ == "__main__":
    main()
