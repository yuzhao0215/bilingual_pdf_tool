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
# CONFIG (same as yours)
# =========================
INPUT_DIR = "input"
OUTPUT_DIR = "output"
PDF_IN  = os.path.join(INPUT_DIR, "archi.pdf")
OUT_DIR = OUTPUT_DIR

FONTFILE = os.path.join("fonts", "PingFangSC.ttc")
FONTNAME = "CN"

MODEL = "gpt-4.1-mini"
BATCH_SIZE = 30
CACHE_PATH = os.path.join("cache", ".cache_zh.json")

GAP_PT = 6
RIGHT_MARGIN_PT = 18
FONTSIZE_SCALE = 0.50

GLOSSARY = {
    "GENERAL NOTES": "一般说明",
    "TYP.": "典型",
    "VERIFY IN FIELD": "现场核实",
}

SKIP_REGEX = [
    r"^\s*$",
    r"^\s*[A-Z]{1,3}\d+(\.\d+)?\s*$",
    r"^\s*\d+\s*/\s*[A-Z]\d+(\.\d+)?\s*$",
    r"^[A-Z]{1,5}[-_][A-Z0-9-]{3,}$",
    r"^[0-9\s,.'\"-]+$",
    r"^\(?[EN]\)?$",
]
SKIP_RE = [re.compile(p) for p in SKIP_REGEX]

# =========================
# HELPERS (same as yours)
# =========================
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
    raise FileNotFoundError("No Chinese font file found.")

def call_with_retry(fn, max_tries=6):
    for attempt in range(1, max_tries + 1):
        try:
            return fn()
        except (RateLimitError, APITimeoutError, APIError) as e:
            wait = min(60, (2 ** (attempt - 1)) + random.random())
            print(f"[WARN] API transient error on attempt {attempt}/{max_tries}: {e}. Sleeping {wait:.1f}s")
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
        raise RuntimeError("Empty model output.")
    data = json.loads(content)

    out = {}
    for it in data["items"]:
        src = norm(it["src"])
        zh = norm(it["zh"])
        forced = glossary_override(src)
        out[src] = forced if forced else zh
    return out

# =========================
# MAIN (single-page output)
# =========================
def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    if not os.path.exists(PDF_IN):
        raise FileNotFoundError(f"PDF not found: {PDF_IN}")

    fontfile = pick_fontfile(FONTFILE)
    print(f"[INFO] Using fontfile: {fontfile}")

    client = OpenAI()
    cache = load_cache(CACHE_PATH)

    # Open input once; we will extract page-by-page but WRITE output page-by-page
    src_doc = fitz.open(PDF_IN)
    print(f"[INFO] Pages: {src_doc.page_count}")

    # -------- PASS A: build todo across whole doc (keeps API behavior same as yours) --------
    todo = set()
    spans_per_page = []  # store only metadata, not output doc

    for p_idx in range(src_doc.page_count):
        page = src_doc[p_idx]
        d = page.get_text("dict")

        spans = []
        for block in d.get("blocks", []):
            if block.get("type", 0) != 0:
                continue
            for line in block.get("lines", []):
                dr = line.get("dir", (1.0, 0.0))
                for sp in line.get("spans", []):
                    t = norm(sp.get("text") or "")
                    if not t:
                        continue
                    bbox = sp.get("bbox")
                    if not bbox:
                        continue
                    fs = float(sp.get("size", 8.0))
                    spans.append((t, bbox, fs, dr))

                    if should_translate(t) and glossary_override(t) is None and t not in cache:
                        todo.add(t)

        spans_per_page.append(spans)

        if (p_idx + 1) % 5 == 0 or (p_idx + 1) == src_doc.page_count:
            print(f"[INFO] Scanned page {p_idx+1}/{src_doc.page_count} | spans so far: {sum(len(x) for x in spans_per_page)} | unique todo: {len(todo)}")

    # -------- PASS B: translate missing strings --------
    todo = sorted(todo)
    print(f"[INFO] Unique strings to translate: {len(todo)}")
    for i in range(0, len(todo), BATCH_SIZE):
        chunk = todo[i:i + BATCH_SIZE]
        mapping = translate_batch(client, chunk)
        cache.update(mapping)
        save_cache(CACHE_PATH, cache)
        print(f"[INFO] Translated {min(i + BATCH_SIZE, len(todo))}/{len(todo)}")

    # -------- PASS C: write ONE output file per page --------
    total_inserted = 0

    for page_idx in range(src_doc.page_count):
        # Create a new 1-page PDF by copying the page
        out_doc = fitz.open()
        out_doc.insert_pdf(src_doc, from_page=page_idx, to_page=page_idx)
        out_page = out_doc[0]

        # Embed font on this one page
        out_page.insert_font(fontname=FONTNAME, fontfile=fontfile)

        page_w = out_page.rect.width
        inserted = 0

        spans = spans_per_page[page_idx]
        for src, (x0, y0, x1, y1), fs, dr in spans:
            if not should_translate(src):
                continue

            zh = glossary_override(src) or cache.get(src, "")
            zh = norm(zh)
            if not zh:
                continue

            # rotation (snap to 0/90/180/270)
            dx, dy = dr[0], dr[1]
            raw = (math.degrees(math.atan2(dy, dx)) + 360) % 360
            angle = (int(round(raw / 90.0)) * 90) % 360
            angle = (360 - angle) % 360  # clockwise

            zh_fs = max(6.0, fs * FONTSIZE_SCALE)

            # place to the right (same as your logic)
            zh_x0 = x1 + GAP_PT
            max_w = page_w - zh_x0 - RIGHT_MARGIN_PT
            if max_w < 20:
                # fallback below
                rect = fitz.Rect(x0, y1 + 2, page_w - RIGHT_MARGIN_PT, y1 + (y1 - y0) + 6)
            else:
                rect = fitz.Rect(zh_x0, y0, zh_x0 + max_w, y1 + 6)

            out_page.insert_text(
                fitz.Point(x0, y0),
                zh,
                fontname=FONTNAME,
                fontsize=zh_fs,
                color=(1, 0, 0),
                rotate=angle,
                overlay=True,
            )

            inserted += 1
            total_inserted += 1

        out_path = os.path.join(OUT_DIR, f"archi_output_p{page_idx+1:03d}.pdf")
        if os.path.exists(out_path):
            os.remove(out_path)
        out_doc.save(out_path, garbage=4, deflate=True)
        out_doc.close()

        print(f"[INFO] Saved {out_path} | page {page_idx+1}/{src_doc.page_count} | inserted this page: {inserted} | inserted total: {total_inserted}")

    src_doc.close()
    print("[DONE] All pages exported to:", OUT_DIR)

if __name__ == "__main__":
    main()
