import os
import re
import json
import time
import random
import fitz  # PyMuPDF
from openai import OpenAI
from openai import APIError, RateLimitError, APITimeoutError, BadRequestError
import math

# =========================
# CONFIG
# =========================
INPUT_DIR = "input"
OUTPUT_DIR = "output"
PDF_IN  = os.path.join(INPUT_DIR, "wrong_original.pdf")
PDF_OUT = os.path.join(OUTPUT_DIR, "wrong_output.pdf")

# PDF_IN  = os.path.join(INPUT_DIR, "right_original.pdf")
# PDF_OUT = os.path.join(OUTPUT_DIR, "right_output.pdf")

FONTFILE = os.path.join("fonts", "PingFangSC.ttc")   # or leave as-is if you already use auto-detect
FONTNAME = "CN"

MODEL = "gpt-4.1-mini"
BATCH_SIZE = 50
CACHE_PATH = os.path.join("cache", ".cache_zh.json")

# Chinese sizing inside same bbox
ZH_SIZE_SCALE = 0.5
ZH_MIN_FONTSIZE = 2.5

# Draw red boxes around original English bboxes (debug)
DRAW_REDBOX = True
REDBOX_WIDTH = 0.8

CHECKPOINT_EVERY = 5

GLOSSARY = {
    "GENERAL NOTES": "一般说明",
    "TYP.": "典型",
    "VERIFY IN FIELD": "现场核实",
}

SKIP_REGEX = [
    r"^\s*$",
    r"^\s*[A-Z]{1,3}\d+(\.\d+)?\s*$",               # A0.1, S101
    r"^\s*\d+\s*/\s*[A-Z]\d+(\.\d+)?\s*$",          # 3/A401
    r"^[A-Z]{1,5}[-_][A-Z0-9-]{3,}$",               # TK-PS03-01
    r"^[0-9\s,.'\"-]+$",                            # mostly numbers/symbols
    r"^\(?[EN]\)?$",                                # (E) or (N) alone
]
SKIP_RE = [re.compile(p) for p in SKIP_REGEX]


# =========================
# HELPERS
# =========================
max_arrows_per_page = 10000   # to avoid clutter
scale = 40                 # arrow length in points
width = 1.5                # stroke width

def draw_arrow(page, x0, y0, x1, y1, color=(1, 0, 0), w=1.5):
    page.draw_line((x0, y0), (x1, y1), color=color, width=w, overlay=True)
    ang = math.atan2(y1 - y0, x1 - x0)
    head_len = 10
    head_ang = math.radians(25)
    hx1 = x1 - head_len * math.cos(ang - head_ang)
    hy1 = y1 - head_len * math.sin(ang - head_ang)
    hx2 = x1 - head_len * math.cos(ang + head_ang)
    hy2 = y1 - head_len * math.sin(ang + head_ang)
    page.draw_polyline([(hx1, hy1), (x1, y1), (hx2, hy2)], color=color, width=w, overlay=True)

def rotate_rect_cw_90n(derot, rect: fitz.Rect, page_w: float, page_h: float, deg_cw: int) -> fitz.Rect:
    """
    Rotate an axis-aligned rect by deg_cw in {0,90,180,270} clockwise around the page.
    Coordinates assume PyMuPDF convention: origin top-left, x right, y down.

    Returns a new axis-aligned rect in the rotated coordinate system.
    """
    deg_cw %= 360
    if deg_cw not in (0, 90, 180, 270):
        raise ValueError("deg_cw must be one of 0, 90, 180, 270")

    x0, y0, x1, y1 = rect.x0, rect.y0, rect.x1, rect.y1

    if deg_cw == 0:
        return fitz.Rect(x0, y0, x1, y1)

    if deg_cw == 90:
        # (x, y) -> (page_h - y, x)
        # Apply to all corners, then bbox
        pts = [
            (page_h - y0, x0),
            (page_h - y0, x1),
            (page_h - y1, x1),
            (page_h - y1, x0),
        ]
    elif deg_cw == 180:
        # (x, y) -> (page_w - x, page_h - y)
        pts = [
            (page_w - x0, page_h - y0),
            (page_w - x1, page_h - y0),
            (page_w - x1, page_h - y1),
            (page_w - x0, page_h - y1),
        ]
    else:  # deg_cw == 270
        # print("rotated by 270 degrees")
        # (x, y) -> (y, page_w - x)
        # pts = [
        #     (y0, page_w - x0),
        #     (y0, page_w - x1),
        #     (y1, page_w - x1),
        #     (y1, page_w - x0),
        # ]
        h = min(rect.height, rect.width)
        # print("x0: {}, y0: {}, x1: {}, y1: {}".format(x0, y0, x1, y1))
        return fitz.Rect(x0 * derot, y0 * derot, x1 * derot, y1 * derot)
       
        pts = [
            (y0, page_h - x0),
            (y0, page_h - x1),
            (y1, page_h - x1),
            (y1, page_h - x0),
        ]

    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    return fitz.Rect(min(xs), min(ys), max(xs), max(ys))

def norm(s: str) -> str:
    return " ".join((s or "").replace("\u00a0", " ").split()).strip()

def glossary_override(src: str) -> str | None:
    s = norm(src)
    if s in GLOSSARY:
        return GLOSSARY[s]
    if s.endswith(".") and s[:-1] in GLOSSARY:
        return GLOSSARY[s[:-1]]
    return None

def should_translate(t: str) -> bool:
    t = norm(t)
    if len(t) < 2 or len(t) > 400:
        return False
    for rx in SKIP_RE:
        if rx.match(t):
            return False
    if len(t) <= 6 and re.fullmatch(r"[A-Z0-9]+", t):
        return False
    return True

def load_cache(path: str) -> dict:
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def save_cache(path: str, cache: dict):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)

def pick_fontfile(cfg: str) -> str:
    if cfg and os.path.exists(cfg):
        return cfg

    candidates = [
        "/System/Library/Fonts/Supplemental/PingFang.ttc",
        "/System/Library/Fonts/PingFang.ttc",
        "/System/Library/Fonts/Hiragino Sans GB.ttc",
        "/System/Library/Fonts/STHeiti Medium.ttc",
        "/System/Library/Fonts/STHeiti Light.ttc",
    ]
    for p in candidates:
        if os.path.exists(p):
            return p

    local_candidates = [
        os.path.join("fonts", "PingFangSC.ttc"),
        "PingFangSC.ttc",
    ]
    for local in local_candidates:
        if os.path.exists(local):
            return local

    raise FileNotFoundError(
        "No Chinese font file found. Set FONTFILE to an absolute path like "
        "/System/Library/Fonts/Supplemental/PingFang.ttc"
    )

def call_with_retry(fn, max_tries=6):
    for attempt in range(1, max_tries + 1):
        try:
            return fn()
        except (RateLimitError, APITimeoutError, APIError) as e:
            wait = min(60, (2 ** (attempt - 1)) + random.random())
            print(f"[WARN] API transient error {attempt}/{max_tries}: {e}. Sleep {wait:.1f}s")
            time.sleep(wait)
        except BadRequestError:
            raise
    raise RuntimeError("API failed after retries.")

def translate_batch(client: OpenAI, texts: list[str]) -> dict[str, str]:
    schema = {
        "type": "object",
        "properties": {
            "items": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {"src": {"type": "string"}, "zh": {"type": "string"}},
                    "required": ["src", "zh"],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["items"],
        "additionalProperties": False,
    }

    gloss_lines = "\n".join([f'- "{k}" -> "{v}"' for k, v in GLOSSARY.items()])

    prompt = f"""Translate architectural drawing text from English to Simplified Chinese.

Rules:
- Keep numbers, units, and codes unchanged (e.g., 3/A401, TK-PS03-01, 12'-6", 6").
- Keep sheet/detail references unchanged.
- Use formal construction/engineering Chinese.
- MUST follow these fixed translations when they appear as standalone terms:
{gloss_lines}

Return JSON only: {{"items":[{{"src":"...","zh":"..."}}]}}
Translate these items:
{json.dumps(texts, ensure_ascii=False)}
"""

    def _do():
        return client.responses.create(
            model=MODEL,
            input=prompt,
            temperature=0.2,
            text={
                "format": {
                    "type": "json_schema",
                    "name": "translation_items",
                    "schema": schema,
                    "strict": True,
                }
            },
        )

    resp = call_with_retry(_do)
    content = (resp.output_text or "").strip()
    if not content:
        raise RuntimeError("Empty model output. Check quota/billing/model.")

    data = json.loads(content)

    out = {}
    for it in data["items"]:
        src = norm(it["src"])
        zh = norm(it["zh"])
        forced = glossary_override(src)
        out[src] = forced if forced else zh
    return out


ROTATE_DEGREE = 270
GAP_PT = 0

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    if not os.path.exists(PDF_IN):
        raise FileNotFoundError(f"PDF not found: {PDF_IN}")

    fontfile = pick_fontfile(FONTFILE)
    print(f"[INFO] Using fontfile: {fontfile}")

    client = OpenAI()
    cache = load_cache(CACHE_PATH)

    doc = fitz.open(PDF_IN)
    print(f"[INFO] Pages: {doc.page_count}")

    # Embed font on each page
    for i, page in enumerate(doc, start=1):
        page.insert_font(fontname=FONTNAME, fontfile=fontfile)
        if i % 5 == 0 or i == doc.page_count:
            print(f"[INFO] Font embedded on page {i}/{doc.page_count}")

    # 1) collect spans + todo strings
    spans_per_page = []
    todo = set()
    arrows = 0

    for p_idx, page in enumerate(doc, start=1):
        # print(f"page {i}: rotation={page.rotation}")
        # page.set_rotation(ROTATE_DEGREE)
        cx = page.rect.x0 + page.rect.width / 2
        cy = page.rect.y0 + page.rect.height / 2

        length = min(page.rect.width, page.rect.height) * 0.18
        draw_arrow(page, cx, cy, cx + length, cy, color=(0, 1, 0), w=4)

        label_rect = fitz.Rect(cx + 10, cy - 30, cx + 220, cy - 5)
        page.insert_textbox(label_rect, "+X", fontsize=18, color=(0, 1, 0), overlay=True)

        d = page.get_text("dict")
        spans = []
        for block in d.get("blocks", []):
            if block.get("type", 0) != 0:
                continue
            for line in block.get("lines", []):
                # if arrows >= max_arrows_per_page:
                #     break

                # dirv = line.get("dir")
                # bbox = line.get("bbox")
                # if not dirv or not bbox:
                #     continue

                # dx, dy = float(dirv[0]), float(dirv[1])
                # L = math.hypot(dx, dy) or 1.0
                # dx, dy = dx / L, dy / L

                # x0, y0, x1, y1 = bbox
                # cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
                # ex, ey = cx + dx * scale, cy + dy * scale

                # draw_arrow(page, cx, cy, ex, ey)
                # arrows += 1

                if "dir" in line:
                    print("LINE dir:", line["dir"])
                for sp in line.get("spans", []):
                    t = norm(sp.get("text") or "")
                    if not t:
                        continue
                    bbox = sp.get("bbox")
                    if not bbox:
                        continue
                    fs = float(sp.get("size", 8.0))
                    # if "dir" in sp:
                    #     print("SPAN dir:", sp["dir"], "text:", (sp.get("text","") or "")[:30])
                    # else:
                    #     print("no direction")
                    dr = line["dir"]
                    spans.append((t, bbox, fs, dr))
                    if should_translate(t) and glossary_override(t) is None and t not in cache:
                        todo.add(t)

        spans_per_page.append(spans)

        if p_idx % 5 == 0 or p_idx == doc.page_count:
            print(f"[INFO] Scanned page {p_idx}/{doc.page_count} | spans: {sum(len(x) for x in spans_per_page)} | todo: {len(todo)}")

    # 2) translate batches
    todo = sorted(todo)
    print(f"[INFO] Unique strings to translate: {len(todo)}")
    for i in range(0, len(todo), BATCH_SIZE):
        chunk = todo[i:i+BATCH_SIZE]
        mapping = translate_batch(client, chunk)
        cache.update(mapping)
        save_cache(CACHE_PATH, cache)
        print(f"[INFO] Translated {min(i+BATCH_SIZE, len(todo))}/{len(todo)}")

    # 3) insert Chinese IN THE SAME BOX + checkpoint every 5 pages
    inserted = 0
    base, ext = os.path.splitext(PDF_OUT)
    if not base:
        base = "output_bilingual"

    for page_idx, (page, spans) in enumerate(zip(doc, spans_per_page), start=1):
        # page.set_rotation(ROTATE_DEGREE)
        # ox, oy = page.rect.x0, page.rect.y0  # usually (0,0)

        # # Arrow tail point (offset into the page)
        # tx, ty = ox + 120, oy + 120

        # # Draw arrow shaft
        # page.draw_line((tx, ty), (ox, oy), color=(1, 0, 0), width=3, overlay=True)

        # # Draw arrowhead
        # angle = math.atan2(oy - ty, ox - tx)
        # head_len = 18
        # head_ang = math.radians(28)

        # x1 = ox + head_len * math.cos(angle + head_ang)
        # y1 = oy + head_len * math.sin(angle + head_ang)
        # x2 = ox + head_len * math.cos(angle - head_ang)
        # y2 = oy + head_len * math.sin(angle - head_ang)

        # page.draw_polyline([(x1, y1), (ox, oy), (x2, y2)], color=(1, 0, 0), width=3, overlay=True)

        # Label near the origin
        # label_rect = fitz.Rect(ox + 10, oy + 5, ox + 260, oy + 50)
        # page.insert_textbox(
        #     label_rect,
        #     "Origin (0,0)",
        #     fontsize=16,
        #     color=(1, 0, 0),
        #     overlay=True,
        #     align=fitz.TEXT_ALIGN_LEFT,
        # )

        for src, (x0, y0, x1, y1), fs, dr in spans:
            # dx, dy = sp.get("dir", (1.0, 0.0))   # baseline direction
            dx, dy = dr[0], dr[1]
            angle = 360 - (math.degrees(math.atan2(dy, dx)) + 360) % 360
            nx, ny = dy, -dx
            L = math.hypot(nx, ny) or 1.0
            nx, ny = nx/L, ny/L
            anchor_x = x1 + nx * GAP_PT
            anchor_y = y0 + ny * GAP_PT

            if not should_translate(src):
                continue

            # Debug: draw red bbox of the original English
            if DRAW_REDBOX:
                page.draw_rect(fitz.Rect(x0, y0, x1, y1), color=(1, 0, 0), width=REDBOX_WIDTH, overlay=True)

            zh = glossary_override(src) or cache.get(src, "")
            zh = norm(zh)
            if not zh:
                continue

            # Chinese font sizing inside the same box
            zh_fs = max(ZH_MIN_FONTSIZE, fs * ZH_SIZE_SCALE)

            # Use EXACT same bbox as English
            rect = fitz.Rect(x0, y0, x1, y1)
            W = page.rect.width
            H = page.rect.height

            rot = 0  # 0/90/180/270
            # If you want to "undo" the page rotation, rotate clockwise by rot:
            # rect = rotate_rect_cw_90n(page.derotation_matrix,rect, W, H,  deg_cw=rot)

            # page.insert_textbox(
            #     rect,
            #     zh,
            #     fontname=FONTNAME,
            #     fontsize=zh_fs,
            #     color=(1, 0, 0),  # red
            #     overlay=True,
            #     align=fitz.TEXT_ALIGN_LEFT,
            #     # rotate=page.rotation

            # )
            # page.insert_text(
            #     fitz.Point(anchor_x, anchor_y),
            #     zh,
            #     fontname=FONTNAME,
            #     fontsize=zh_fs,
            #     color=(1, 0, 0),
            #     rotate=angle,
            #     overlay=True,
            # )
            # page.insert_text(
            #     fitz.Point(anchor_x, anchor_y),
            #     zh,
            #     fontname=FONTNAME,
            #     fontsize=zh_fs,
            #     color=(1, 0, 0),
            #     rotate=angle,
            #     overlay=True,
            #     # rotate=page.rotation
            # )
            page.insert_text(
                fitz.Point(x0, y0),
                zh,
                fontname=FONTNAME,
                fontsize=zh_fs,
                color=(1, 0, 0),
                rotate=angle,
                overlay=True,
                # rotate=page.rotation
            )

            inserted += 1

        if page_idx % CHECKPOINT_EVERY == 0 or page_idx == doc.page_count:
            chk_path = f"{base}_p{page_idx:03d}{ext}"
            if os.path.exists(chk_path):
                os.remove(chk_path)
            doc.save(chk_path, garbage=4, deflate=True)
            print(f"[CHECKPOINT] Saved {chk_path} | page {page_idx}/{doc.page_count} | inserted: {inserted}")

    # final save
    if os.path.exists(PDF_OUT):
        os.remove(PDF_OUT)
    doc.save(PDF_OUT, garbage=4, deflate=True)
    doc.close()
    print(f"[DONE] Saved: {PDF_OUT} | inserted zh items: {inserted}")


if __name__ == "__main__":
    main()
