"""Printable labels laid out like real US packs (no brands) + the gold visible text.

Each sticker is a front panel and a side panel with a fold line between them, the way a
can, box or bottle shows its name on the front and its facts on the side:
- front: product name, descriptor, net quantity in the bottom 30 % at the FDA minimum
  height for the panel area (FDA Dietary Supplement Labeling Guide ch. III table);
- side: Nutrition / Supplement / Drug Facts, ingredients, directions, and a BEST BY or
  EXP date printed like an ink stamp;
- medicine: real generic ingredients (no brands), Drug Facts headings in the order of
  21 CFR 201.66(c), body text >= 6 pt. Directions follow the common OTC wording.
Nadya, 2026-10-21: make it as on real packs, dates / weights / use.

Writes eval/labels/labels.pdf (US Letter, 300 dpi, actual size: print at 100 %)
and eval/labels/labels.json (label id -> every word printed on it, front and side),
which the eval uses as ground truth for "only visible text".
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
SQ_IN = 2.54 * 2.54


def pt(points: float) -> int:
    return round(points / 72 * DPI)


def net_pt(w_cm: float, h_cm: float) -> float:
    """Net quantity size for the front panel area: 1/16 in up to 5 sq in, else 1/8 in (up to 25)."""
    letter_in = 1 / 16 if w_cm * h_cm / SQ_IN <= 5 else 1 / 8
    return letter_in * 72 / 0.72  # cap height ~0.72 of the font size


# Side-panel blocks: ("facts", title, [(left, right, bold)]) · ("head", text) · ("text", text)
# · ("stamp", text). Front: (name, descriptor lines, net quantity lines).
LABELS = [
    dict(id="soup", put_on="на круглую банку овсянки (обернуть)", front=(9.3, 12), side=6.5, colour="#c0392b",
         name="TOMATO SOUP", desc=["Condensed", "Ready in 5 minutes"], net=["NET WT 14.1 OZ (400 g)"],
         blocks=[("stamp", "BEST BY DEC 12 2027  L7231"),
                 ("facts", "Nutrition Facts", [("About 3 servings per container", "", False),
                  ("Serving size 1/2 cup (120 mL)", "", True), ("Calories", "90", True), ("Total Fat 0g", "0%", False),
                  ("Sodium 480mg", "21%", False), ("Total Carb. 20g", "7%", False), ("Total Sugars 12g", "", False),
                  ("Protein 2g", "", False)]),
                 ("head", "INGREDIENTS"),
                 ("text", "TOMATO PUREE (WATER, TOMATO PASTE), SUGAR, WHEAT FLOUR, SALT, CITRIC ACID, SPICE. "
                          "CONTAINS: WHEAT."),
                 ("head", "DIRECTIONS"),
                 ("text", "STOVE: Stir 1 can soup and 1 can water. Heat 5 minutes, stirring often."),
                 ("text", "MICROWAVE: Mix in a covered bowl. Heat 2 1/2 to 3 minutes. Stir before serving.")]),
    dict(id="pasta", put_on="на коробку пасты: перед + бок (согнуть)", front=(9.3, 12), side=6.5, colour="#1f4e9c",
         name="ROTINI", desc=["Enriched Macaroni Product"], net=["NET WT 12 OZ (340 g)"],
         blocks=[("stamp", "BEST BY MAR 05 2028"),
                 ("facts", "Nutrition Facts", [("About 6 servings per container", "", False),
                  ("Serving size 2 oz (56g) dry", "", True), ("Calories", "200", True), ("Total Fat 1g", "1%", False),
                  ("Sodium 0mg", "0%", False), ("Total Carb. 42g", "15%", False), ("Dietary Fiber 3g", "11%", False),
                  ("Protein 7g", "", False)]),
                 ("head", "INGREDIENTS"),
                 ("text", "SEMOLINA (WHEAT), DURUM WHEAT FLOUR, NIACIN, IRON (FERROUS SULFATE), THIAMINE "
                          "MONONITRATE, RIBOFLAVIN, FOLIC ACID. CONTAINS: WHEAT."),
                 ("head", "COOKING DIRECTIONS"),
                 ("text", "1. Bring 4 to 6 quarts of water to a boil. 2. Add pasta and stir. "
                          "3. Boil uncovered 8 minutes, stirring now and then. 4. Drain and serve.")]),
    dict(id="bread_mix", put_on="на вторую коробку пасты: перед + бок", front=(9.3, 13), side=6.5, colour="#7a4b1e",
         name="BANANA BREAD MIX", desc=["Makes 1 loaf · 10 slices"], net=["NET WT 14 OZ (397 g)"],
         blocks=[("stamp", "BEST BY JAN 18 2028"),
                 ("head", "YOU WILL NEED"),
                 ("text", "2 eggs · 1/3 cup oil · 2/3 cup water"),
                 ("head", "DIRECTIONS"),
                 ("text", "1. Heat oven to 350°F. Grease the bottom of an 8 x 4 inch loaf pan."),
                 ("text", "2. Stir mix, water, oil and eggs until blended."),
                 ("text", "3. Pour into the pan."),
                 ("text", "4. Bake 45 to 50 minutes, until a toothpick comes out clean."),
                 ("text", "5. Cool 10 minutes before slicing."),
                 ("facts", "Nutrition Facts", [("10 servings per container", "", False),
                  ("Serving size 1/10 package (40g)", "", True), ("Calories", "150", True),
                  ("Total Fat 1g", "1%", False), ("Sodium 220mg", "10%", False), ("Total Carb. 33g", "12%", False)]),
                 ("head", "INGREDIENTS"),
                 ("text", "ENRICHED FLOUR, SUGAR, DRIED BANANA, LEAVENING (BAKING SODA), SALT, NATURAL FLAVOR. "
                          "CONTAINS: WHEAT.")]),
    dict(id="rice", put_on="на пакет риса (перед + бок/зад)", front=(9.3, 12), side=6.5, colour="#2e7d5b",
         name="WHITE RICE", desc=["Extra Long Grain"], net=["NET WT 2 LB (907 g)"],
         blocks=[("stamp", "BEST BY JUN 2028  2154R"),
                 ("facts", "Nutrition Facts", [("About 20 servings per container", "", False),
                  ("Serving size 1/4 cup (45g) dry", "", True), ("Calories", "160", True),
                  ("Total Fat 0g", "0%", False), ("Sodium 0mg", "0%", False), ("Total Carb. 36g", "13%", False),
                  ("Protein 3g", "", False)]),
                 ("head", "INGREDIENTS"),
                 ("text", "LONG GRAIN WHITE RICE."),
                 ("head", "STOVETOP"),
                 ("text", "1 cup rice + 2 cups water. Bring to a boil, cover and simmer 18 minutes. "
                          "Remove from heat and rest 5 minutes. Fluff with a fork.")]),
    dict(id="soup_seasoning", put_on="на пакетик приправы: перед + зад", front=(8, 10), side=5.5, colour="#e67e22",
         name="SOUP SEASONING", desc=["with vegetables & herbs"], net=["NET WT 0.88 OZ (25 g)"],
         blocks=[("stamp", "BEST BY 09/2027"),
                 ("head", "INGREDIENTS"),
                 ("text", "SALT, DRIED VEGETABLES (CARROT, ONION, PARSNIP), MALTODEXTRIN, SUGAR, SPICES, TURMERIC."),
                 ("head", "DIRECTIONS"),
                 ("text", "Add 1 sachet to 2.5 L (10 cups) of boiling soup. Simmer 5 minutes."),
                 ("text", "Store in a dry place.")]),
    dict(id="sugar", put_on="на баночку специй (обернуть)", front=(5, 5.5), side=4, colour="#555555",
         name="SUGAR", desc=["Granulated"], net=["NET WT 4.5 OZ (128 g)"],
         blocks=[("stamp", "BEST BY 11/2028"), ("head", "INGREDIENTS"), ("text", "CANE SUGAR.")]),
    dict(id="salt", put_on="на баночку специй (обернуть)", front=(5, 5.5), side=4, colour="#555555",
         name="SALT", desc=["Iodized"], net=["NET WT 6 OZ (170 g)"],
         blocks=[("stamp", "BEST BY 04/2029"),
                 ("facts", "Nutrition Facts", [("Serving size 1/4 tsp (1.5g)", "", True),
                  ("Sodium 590mg", "26%", False)]),
                 ("head", "INGREDIENTS"),
                 ("text", "SALT, CALCIUM SILICATE, POTASSIUM IODIDE.")]),
    dict(id="paprika", put_on="на баночку специй (обернуть)", front=(5, 5.5), side=4, colour="#b03a2e",
         name="PAPRIKA", desc=["Ground"], net=["NET WT 1.41 OZ (40 g)"],
         blocks=[("stamp", "BEST BY 02/2028"), ("head", "INGREDIENTS"), ("text", "PAPRIKA.")]),
    dict(id="vitamin_d", put_on="на флакон или коробку лекарства (обернуть)", front=(6, 8), side=7, colour="#6c3483",
         name="VITAMIN D3", desc=["25 mcg (1,000 IU)", "Supports bone health*", "DIETARY SUPPLEMENT"],
         net=["60 SOFTGELS"],
         blocks=[("stamp", "EXP 08/2027  LOT 24D115"),
                 ("facts", "Supplement Facts", [("Serving Size 1 Softgel", "", True),
                  ("Vitamin D (as cholecalciferol) 25 mcg (1,000 IU)", "125%", False)]),
                 ("head", "SUGGESTED USE"),
                 ("text", "Adults: take 1 softgel daily with a meal."),
                 ("head", "OTHER INGREDIENTS"),
                 ("text", "Soybean oil, gelatin, glycerin, water."),
                 ("text", "Keep out of reach of children. Store in a cool, dry place."),
                 ("text", "*This statement has not been evaluated by the Food and Drug Administration. This product "
                          "is not intended to diagnose, treat, cure, or prevent any disease.")]),
    dict(id="sleep_aid", put_on="на флакон или коробку лекарства (обернуть)", front=(6, 8), side=7.5, colour="#34495e",
         name="SLEEP AID", desc=["Diphenhydramine HCl 25 mg", "Nighttime Sleep-Aid"], net=["24 CAPLETS"],
         blocks=[("stamp", "EXP 05/2027  LOT 3K0921"),
                 ("drug", "Drug Facts"),
                 ("facts", "", [("Active ingredient (in each caplet)", "Purpose", True),
                  ("Diphenhydramine HCl 25 mg", "Nighttime sleep-aid", False)]),
                 ("head", "Use"), ("text", "for relief of occasional sleeplessness"),
                 ("head", "Warnings"),
                 ("text", "Do not use for children under 12 years of age; with any other product containing "
                          "diphenhydramine. When using this product avoid alcoholic drinks. Stop use and ask a "
                          "doctor if sleeplessness lasts more than 2 weeks. Keep out of reach of children."),
                 ("head", "Directions"),
                 ("text", "adults and children 12 years and over: take 2 caplets at bedtime if needed, or as "
                          "directed by a doctor. children under 12 years: do not use."),
                 ("head", "Other information"), ("text", "store at 20-25°C (68-77°F)")]),
    dict(id="pain_relief", put_on="на коробку лекарства: перед + бок", front=(8, 7.5), side=8, colour="#1a5276",
         name="PAIN RELIEF", desc=["Extra Strength", "Acetaminophen 500 mg", "Pain Reliever / Fever Reducer"],
         net=["100 CAPLETS"],
         blocks=[("stamp", "EXP 10/2027  LOT 5P2207"),
                 ("drug", "Drug Facts"),
                 ("facts", "", [("Active ingredient (in each caplet)", "Purpose", True),
                  ("Acetaminophen 500 mg", "Pain reliever/fever reducer", False)]),
                 ("head", "Uses"), ("text", "temporarily relieves minor aches and pains; temporarily reduces fever"),
                 ("head", "Warnings"),
                 ("text", "Liver warning: severe liver damage may occur if you take more than 4,000 mg in 24 hours, "
                          "with other drugs containing acetaminophen, or 3 or more alcoholic drinks every day. "
                          "Keep out of reach of children."),
                 ("head", "Directions"),
                 ("text", "adults and children 12 years and over: take 2 caplets every 6 hours while symptoms last. "
                          "Do not take more than 6 caplets in 24 hours. children under 12 years: ask a doctor.")]),
]


def font(size_px: int, bold: bool = False, mono: bool = False) -> ImageFont.FreeTypeFont:
    name = ("Courier New Bold.ttf" if mono else "Arial Bold.ttf" if bold else "Arial.ttf")
    return ImageFont.truetype(FONTS + name, size_px)


def fit(draw: ImageDraw.ImageDraw, text: str, max_w: int, start: int) -> ImageFont.FreeTypeFont:
    size = start
    while size > 10 and draw.textlength(text, font=font(size, True)) > max_w:
        size -= 2
    return font(size, True)


def wrap(draw: ImageDraw.ImageDraw, text: str, f: ImageFont.FreeTypeFont, max_w: int) -> list[str]:
    lines, cur = [], ""
    for word in text.split():
        nxt = f"{cur} {word}".strip()
        if cur and draw.textlength(nxt, font=f) > max_w:
            lines.append(cur)
            cur = word
        else:
            cur = nxt
    return lines + [cur] if cur else lines


def front_panel(lab: dict) -> Image.Image:
    w_cm, h_cm = lab["front"]
    w, h = int(w_cm * CM), int(h_cm * CM)
    img = Image.new("RGB", (w, h), "white")
    d = ImageDraw.Draw(img)
    pad = int(0.3 * CM)
    band = int(h * 0.32)
    d.rectangle((0, 0, w, band), fill=lab["colour"])
    f = fit(d, lab["name"], w - 2 * pad, int(band * 0.42))
    d.text((w / 2, band / 2), lab["name"], font=f, fill="white", anchor="mm")
    y = band + pad
    for i, line in enumerate(lab["desc"]):
        f = font(pt(11 if i == 0 and w_cm > 5.5 else 8), bold=i == 0)
        d.text((w / 2, y), line, font=f, fill="black", anchor="ma")
        y += int(f.size * 1.3)
    f = font(round(pt(net_pt(w_cm, h_cm))), bold=True)
    y = h - pad - len(lab["net"]) * int(f.size * 1.2)  # bottom 30 % of the panel
    for line in lab["net"]:
        d.text((w / 2, y), line, font=f, fill="black", anchor="ma")
        y += int(f.size * 1.2)
    return img


def side_panel(lab: dict) -> Image.Image:
    w, h = int(lab["side"] * CM), int(lab["front"][1] * CM)
    img = Image.new("RGB", (w, h), "white")
    d = ImageDraw.Draw(img)
    pad = int(0.25 * CM)
    body, head, inner = font(pt(6.5)), font(pt(7.5), bold=True), w - 2 * pad
    y = pad
    for kind, *rest in lab["blocks"]:
        if kind == "stamp":
            f = font(pt(8), mono=True)
            d.text((w - pad, y), rest[0], font=f, fill="#3a3a3a", anchor="ra")
            y += int(f.size * 1.6)
        elif kind == "drug":
            f = font(pt(11), bold=True)
            d.text((pad, y), rest[0], font=f, fill="black")
            y += int(f.size * 1.15)
            d.line((pad, y, w - pad, y), fill="black", width=6)
            y += 8
        elif kind == "facts":
            title, rows = rest
            box_top = y
            if title:
                f = font(pt(10), bold=True)
                d.text((pad + 8, y + 4), title, font=f, fill="black")
                y += int(f.size * 1.25)
                d.line((pad + 8, y, w - pad - 8, y), fill="black", width=5)
                y += 6
            for left, right, bold in rows:
                f = font(pt(6.5), bold=bold)
                for line in wrap(d, left, f, inner - 16 - (int(d.textlength(right, font=f)) + 12 if right else 0)):
                    d.text((pad + 8, y), line, font=f, fill="black")
                    y += int(f.size * 1.2)
                if right:
                    d.text((w - pad - 8, y - int(f.size * 1.2)), right, font=f, fill="black", anchor="ra")
                d.line((pad + 8, y + 1, w - pad - 8, y + 1), fill="black", width=1)
                y += 4
            if title:
                d.rectangle((pad, box_top, w - pad, y + 4), outline="black", width=3)
            y += int(0.2 * CM)
        elif kind == "head":
            d.text((pad, y), rest[0], font=head, fill="black")
            y += int(head.size * 1.2)
        else:
            for line in wrap(d, rest[0], body, inner):
                d.text((pad, y), line, font=body, fill="black")
                y += int(body.size * 1.2)
            y += int(body.size * 0.4)
    if y > h:
        raise ValueError(f"{lab['id']}: side panel overflows by {(y - h) / CM:.1f} cm")
    return img


def render(lab: dict) -> Image.Image:
    front, side = front_panel(lab), side_panel(lab)
    img = Image.new("RGB", (front.width + side.width, front.height), "white")
    img.paste(front, (0, 0))
    img.paste(side, (front.width, 0))
    d = ImageDraw.Draw(img)
    for yy in range(0, img.height, 30):  # fold line between front and side
        d.line((front.width, yy, front.width, yy + 10), fill="#bbbbbb", width=2)
    for x in range(0, img.width, 24):  # dashed cut line
        d.line((x, 0, x + 12, 0), fill="#999", width=2)
        d.line((x, img.height - 2, x + 12, img.height - 2), fill="#999", width=2)
    for yy in range(0, img.height, 24):
        d.line((0, yy, 0, yy + 12), fill="#999", width=2)
        d.line((img.width - 2, yy, img.width - 2, yy + 12), fill="#999", width=2)
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
        ImageDraw.Draw(page).text((x, y + im.height + 4), name, font=font(28), fill="#888")
        x += im.width + GAP
        row_h = max(row_h, im.height + 40)
    if page is not None:
        pages.append(page)
    return pages


def words(lab: dict) -> tuple[str, str]:
    front = " ".join([lab["name"], *lab["desc"], *lab["net"]])
    side = []
    for kind, *rest in lab["blocks"]:
        if kind == "facts":
            side += [rest[0]] + [f"{a} {b}".strip() for a, b, _ in rest[1]]
        else:
            side.append(rest[0])
    return front, " ".join(s for s in side if s)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rendered = [(f"{lab['put_on']}  · сгиб по серой линии", render(lab)) for lab in LABELS]
    pages = pack(rendered)
    pages[0].save(OUT / "labels.pdf", save_all=True, append_images=pages[1:], resolution=DPI)
    gold = {}
    for lab in LABELS:
        front, side = words(lab)
        gold[lab["id"]] = {"put_on": lab["put_on"], "front": front, "side": side, "text": f"{front} {side}"}
    (OUT / "labels.json").write_text(json.dumps(gold, indent=2, ensure_ascii=False))
    for old in OUT.glob("preview-*.png"):
        old.unlink()
    for i, p in enumerate(pages, 1):
        p.resize((p.width // 3, p.height // 3)).save(OUT / f"preview-{i}.png")
    print(len(LABELS), "labels on", len(pages), "pages ->", OUT)


if __name__ == "__main__":
    main()
