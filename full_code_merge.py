import os
import re
import json
import time
import random
import math
import fitz  # PyMuPDF
from openai import OpenAI
from openai import APIError, RateLimitError, APITimeoutError, BadRequestError

# =========================
# CONFIG
# =========================
INPUT_DIR = "input"
OUTPUT_DIR = "output"

PDF_IN  = os.path.join(INPUT_DIR, "archi.pdf")
PDF_OUT = os.path.join(OUTPUT_DIR, "archi_output_merge.pdf")

# PDF_IN  = os.path.join(INPUT_DIR, "right_original.pdf")
# PDF_OUT = os.path.join(OUTPUT_DIR, "right_original_output.pdf")

# PDF_IN  = os.path.join(INPUT_DIR, "wrong_original.pdf")
# PDF_OUT = os.path.join(OUTPUT_DIR, "wrong_original_output.pdf")

FONTFILE = os.path.join("fonts", "PingFangSC.ttc")  # or auto-detect via pick_fontfile
FONTNAME = "CN"

MODEL = "gpt-4.1-mini"
BATCH_SIZE = 30
CACHE_PATH = os.path.join("cache", ".cache_zh.json")

# placement
GAP_PT = 6
RIGHT_MARGIN_PT = 18
MIN_RIGHT_WIDTH_PT = 140       # avoids skinny boxes that wrap 1-char-per-line
HEIGHT_MULT = 3.0              # allow multi-line Chinese
FONTSIZE_SCALE = 0.55          # Chinese font relative to source mean font size
MIN_FONTSIZE = 6.0

# paragraph merge tuning
MERGE_GAP_PT = 3.0
MERGE_INDENT_PT = 10.0
MERGE_FONT_TOL = 0.6

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
# BASIC HELPERS
# =========================
def snap_dir_to_90(dx: float, dy: float) -> tuple[tuple[float, float], int]:
    """Snap a direction vector to the nearest 0/90/180/270."""
    ang = (math.degrees(math.atan2(dy, dx)) + 360) % 360
    ang90 = (int(round(ang / 90.0)) * 90) % 360
    # Convert snapped angle back to unit direction
    if ang90 == 0:
        return (1.0, 0.0), 0
    if ang90 == 90:
        return (0.0, 1.0), 90
    if ang90 == 180:
        return (-1.0, 0.0), 180
    return (0.0, -1.0), 270

def get_block_direction(block: dict) -> tuple[tuple[float, float], int]:
    """
    Compute dominant direction for a text block using its lines' 'dir'.
    Returns (dir_vector, angle_deg) snapped to 0/90/180/270.
    """
    lines = block.get("lines", [])
    if not lines:
        return (1.0, 0.0), 0

    # vote by line "length" (bbox width/height) to avoid tiny lines dominating
    votes = {0: 0.0, 90: 0.0, 180: 0.0, 270: 0.0}

    for ln in lines:
        d = ln.get("dir", (1.0, 0.0))
        dx, dy = float(d[0]), float(d[1])

        # weight: use line bbox perimeter/area-ish
        bb = ln.get("bbox")
        w = h = 1.0
        if bb:
            w = max(1.0, float(bb[2] - bb[0]))
            h = max(1.0, float(bb[3] - bb[1]))
        weight = w + h  # simple, stable

        _, ang90 = snap_dir_to_90(dx, dy)
        votes[ang90] += weight

    best_ang = max(votes, key=votes.get)
    # Return unit vector for best angle
    if best_ang == 0:
        return (1.0, 0.0), 0
    if best_ang == 90:
        return (0.0, 1.0), 360-90
    if best_ang == 180:
        return (-1.0, 0.0), 180
    return (0.0, -1.0), 360-270

def norm(s: str) -> str:
    return " ".join((s or "").replace("\u00a0", " ").split()).strip()

def apply_glossary_exact(t: str) -> str | None:
    t = norm(t)
    if t in GLOSSARY:
        return GLOSSARY[t]
    if t.endswith(".") and t[:-1] in GLOSSARY:
        return GLOSSARY[t[:-1]]
    return None

def should_translate_text(t: str) -> bool:
    t = norm(t)
    if len(t) < 2 or len(t) > 1200:  # groups can be longer than single spans
        return False
    for rx in SKIP_RE:
        if rx.match(t):
            return False
    if len(t) <= 6 and re.fullmatch(r"[A-Z0-9]+", t):
        return False
    return True

def load_cache(path: str) -> dict:
    if not path or not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_cache(path: str, cache: dict) -> None:
    if not path:
        return
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)

def pick_fontfile(cfg_fontfile: str) -> str:
    if cfg_fontfile and os.path.exists(cfg_fontfile):
        return cfg_fontfile

    mac_candidates = [
        "/System/Library/Fonts/Supplemental/PingFang.ttc",
        "/System/Library/Fonts/PingFang.ttc",
        "/System/Library/Fonts/Hiragino Sans GB.ttc",
        "/System/Library/Fonts/STHeiti Medium.ttc",
        "/System/Library/Fonts/STHeiti Light.ttc",
    ]
    for p in mac_candidates:
        if os.path.exists(p):
            return p

    local = os.path.join("fonts", "PingFangSC.ttc")
    if os.path.exists(local):
        return local

    raise FileNotFoundError("No Chinese font file found (PingFang/Hiragino/STHeiti).")


# =========================
# PARAGRAPH / SENTENCE GROUPING
# =========================
_RX_BULLET = re.compile(r"^\s*[\-\u2013\u2014\u2022\u25CF\u25AA\u25E6]\s+")
_RX_NUM    = re.compile(r"^\s*\(?\d+[\)\.]\s+")
_RX_ALPHA  = re.compile(r"^\s*[A-Z][\)\.]\s+")
_RX_NOTE   = re.compile(r"^\s*(NOTE|NOTES|CAUTION|WARNING)\s*[:\-]\s*", re.IGNORECASE)
_RX_DETAIL = re.compile(r"^\s*\d+\s*/\s*[A-Z]\d+(\.\d+)?\s*$")  # 3/A401
_RX_SHEET  = re.compile(r"^\s*[A-Z]{1,3}\d+(\.\d+)?\s*$")      # A0.1
_RX_ALLCAPS_HEADING = re.compile(r"^[A-Z0-9 \-/]{5,}$")

def is_new_item_start(text: str) -> bool:
    t = norm(text)
    if not t:
        return False
    if _RX_BULLET.match(t) or _RX_NUM.match(t) or _RX_ALPHA.match(t):
        return True
    if _RX_NOTE.match(t):
        return True
    if len(t) <= 40 and _RX_ALLCAPS_HEADING.match(t) and not _RX_DETAIL.match(t) and not _RX_SHEET.match(t):
        return True
    return False

def looks_like_sentence_end(text: str) -> bool:
    t = norm(text)
    return bool(t) and t[-1] in ".;:?!。；：？！"

def _mean_font_size(line: dict) -> float:
    ss = [float(sp.get("size", 0.0)) for sp in line.get("spans", []) if sp.get("size") is not None]
    return (sum(ss) / len(ss)) if ss else 0.0

def _line_text(line: dict) -> str:
    return norm(" ".join((sp.get("text") or "").strip() for sp in line.get("spans", []) if (sp.get("text") or "").strip()))

def should_merge_lines(prev_line: dict, next_line: dict,
                       gap_pt: float = MERGE_GAP_PT,
                       indent_pt: float = MERGE_INDENT_PT,
                       font_tol: float = MERGE_FONT_TOL) -> bool:
    if not prev_line.get("spans") or not next_line.get("spans"):
        return False
    if "bbox" not in prev_line or "bbox" not in next_line:
        return False

    # direction match (baseline)
    d1 = prev_line.get("dir", (1.0, 0.0))
    d2 = next_line.get("dir", (1.0, 0.0))
    if (abs(d1[0] - d2[0]) + abs(d1[1] - d2[1])) > 0.2:
        return False

    fs1 = _mean_font_size(prev_line)
    fs2 = _mean_font_size(next_line)
    if fs1 and fs2 and abs(fs1 - fs2) > font_tol:
        return False

    x0a, y0a, x1a, y1a = prev_line["bbox"]
    x0b, y0b, x1b, y1b = next_line["bbox"]

    gap = y0b - y1a
    if gap < -1.0:
        return False
    if gap > gap_pt:
        return False

    indent = abs(x0b - x0a)
    if indent > indent_pt:
        return False

    prev_text = _line_text(prev_line)
    next_text = _line_text(next_line)
    if not prev_text or not next_text:
        return False

    if is_new_item_start(next_text):
        return False

    if looks_like_sentence_end(prev_text):
        if re.match(r"^[A-Z]", next_text) and indent > 2.0:
            return False

    return True


# =========================
# API + RETRY
# =========================
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
- This is construction / architectural content.
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
    for it in data.get("items", []):
        src = norm(it.get("src", ""))
        zh = norm(it.get("zh", ""))
        if not src:
            continue
        forced = apply_glossary_exact(src)
        out[src] = forced if forced else zh
    return out


# =========================
# MAIN (MERGE LINES INTO GROUPS)
# =========================
def main():
    os.makedirs(INPUT_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(CACHE_PATH) or ".", exist_ok=True)

    cache = load_cache(CACHE_PATH)
    fontfile = pick_fontfile(FONTFILE)
    print(f"[INFO] Using fontfile: {fontfile}")

    doc = fitz.open(PDF_IN)
    print(f"[INFO] Pages: {doc.page_count}")

    # Embed font on each page
    for i, page in enumerate(doc, start=1):
        page.insert_font(fontname=FONTNAME, fontfile=fontfile)
        if i % 5 == 0 or i == doc.page_count:
            print(f"[INFO] Font embedded on page {i}/{doc.page_count}")

    # -------- Pass 1: build grouped paragraphs per page + todo --------
    groups_per_page = []   # list[page] -> list[dict{text,bbox,fs,dir}]
    todo = set()

    for p_idx, page in enumerate(doc, start=1):
        d = page.get_text("dict")
        page_groups = []

        for block in d.get("blocks", []):
            if block.get("type", 0) != 0:
                continue

            bdir, bang = get_block_direction(block)
            # print("BLOCK dir:", bdir, "angle:", bang)

            lines = block.get("lines", [])
            i = 0
            while i < len(lines):
                # start a group
                cur = lines[i]
                group_lines = [cur]
                j = i + 1
                while j < len(lines) and should_merge_lines(lines[j-1], lines[j]):
                    group_lines.append(lines[j])
                    j += 1

                # build group text
                texts = [_line_text(ln) for ln in group_lines]
                group_text = norm(" ".join([t for t in texts if t]))

                # union bbox across group lines
                bxs = [ln.get("bbox") for ln in group_lines if ln.get("bbox")]
                if not bxs or not group_text:
                    i = j
                    continue
                x0 = min(bb[0] for bb in bxs)
                y0 = min(bb[1] for bb in bxs)
                x1 = max(bb[2] for bb in bxs)
                y1 = max(bb[3] for bb in bxs)

                # representative font size (mean across lines)
                sizes = [_mean_font_size(ln) for ln in group_lines if _mean_font_size(ln) > 0]
                fs = sum(sizes) / len(sizes) if sizes else 8.0

                # direction: take first line dir
                dr = group_lines[0].get("dir", (1.0, 0.0))

                # store group
                page_groups.append({
                    "src": group_text,
                    "bbox": (x0, y0, x1, y1),
                    "fs": fs,
                    "dir": dr,
                    "ang": bang
                })

                # add to todo if needs translation
                if should_translate_text(group_text) and apply_glossary_exact(group_text) is None and group_text not in cache:
                    todo.add(group_text)

                i = j

        groups_per_page.append(page_groups)

        if p_idx % 5 == 0 or p_idx == doc.page_count:
            total_groups = sum(len(x) for x in groups_per_page)
            print(f"[INFO] Scanned page {p_idx}/{doc.page_count} | groups so far: {total_groups} | unique todo: {len(todo)}")

    # -------- Pass 2: translate groups --------
    todo = sorted(todo)
    print(f"[INFO] Unique groups to translate: {len(todo)}")

    if todo:
        client = OpenAI()
        for i in range(0, len(todo), BATCH_SIZE):
            chunk = todo[i:i + BATCH_SIZE]
            mapping = translate_batch(client, chunk)
            cache.update(mapping)
            save_cache(CACHE_PATH, cache)
            print(f"[INFO] Translated {min(i + BATCH_SIZE, len(todo))}/{len(todo)} groups")
    else:
        save_cache(CACHE_PATH, cache)

    # -------- Pass 3: insert Chinese once per group --------
    inserted = 0
    for page_idx, (page, page_groups) in enumerate(zip(doc, groups_per_page), start=1):
        page_w = page.rect.width

        for g in page_groups:
            ang = g["ang"]
            src = g["src"]
            if not should_translate_text(src):
                continue

            zh = apply_glossary_exact(src) or cache.get(src, "")
            zh = norm(zh)
            if not zh:
                continue

            x0, y0, x1, y1 = g["bbox"]
            fs = float(g["fs"])

            zh_fs = max(MIN_FONTSIZE, fs * FONTSIZE_SCALE)

            # place to the right of the GROUP bbox
            zh_x0 = x1 + GAP_PT
            max_w = page_w - zh_x0 - RIGHT_MARGIN_PT
            h = max(zh_fs * 1.2, (y1 - y0))

            if max_w < MIN_RIGHT_WIDTH_PT:
                # fallback below (prevents skinny wrapping)
                rect = fitz.Rect(
                    x0,
                    y1 + 2,
                    page_w - RIGHT_MARGIN_PT,
                    y1 + 2 + h * HEIGHT_MULT
                )
            else:
                rect = fitz.Rect(
                    zh_x0,
                    y0,
                    zh_x0 + max_w,
                    y0 + h * HEIGHT_MULT
                )
            if page.rotation == 0:
                page.insert_textbox(
                    rect,
                    zh,
                    fontname=FONTNAME,
                    fontsize=zh_fs,
                    color=(1, 0, 0),
                    overlay=True,
                    align=fitz.TEXT_ALIGN_LEFT,
                )
            else:
                page.insert_text(
                    fitz.Point(x0, y0),
                    zh,
                    fontname=FONTNAME,
                    fontsize=zh_fs,
                    color=(1, 0, 0),
                    rotate=ang,
                    overlay=True,
                )
            inserted += 1

        print(f"[INFO] Done inserting page {page_idx}/{doc.page_count} | inserted so far: {inserted}")

    # save output
    os.makedirs(os.path.dirname(PDF_OUT) or ".", exist_ok=True)
    if os.path.exists(PDF_OUT):
        os.remove(PDF_OUT)
    doc.save(PDF_OUT, garbage=4, deflate=True)
    doc.close()
    print(f"[DONE] Saved: {PDF_OUT} | inserted zh groups: {inserted}")


if __name__ == "__main__":
    main()