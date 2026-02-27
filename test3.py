import os
import re
import json
import fitz  # PyMuPDF
from openai import OpenAI

# =========================
# CONFIG
# =========================
PDF_IN  = "sample2.pdf"
PDF_OUT = "sample_bilingual_zh_red_only.pdf"

FONTFILE = "PingFangSC.ttc"
# FONTFILE = "/System/Library/Fonts/Hiragino Sans GB.ttc"
# FONTFILE = "/System/Library/Fonts/STHeiti Medium.ttc"
FONTNAME = "CN"

MODEL = "gpt-4.1-mini"
BATCH_SIZE = 50
CACHE_PATH = ".cache_zh.json"

GAP_PT = 6
RIGHT_MARGIN_PT = 18
FONTSIZE_SCALE = 0.90

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
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)

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

    resp = client.responses.create(
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


def main():
    if not os.path.exists(FONTFILE):
        raise FileNotFoundError(f"Font file not found: {FONTFILE}")

    client = OpenAI()
    cache = load_cache(CACHE_PATH)

    doc = fitz.open(PDF_IN)

    # Embed font on each page (so Chinese doesn't become ?????)
    for page in doc:
        page.insert_font(fontname=FONTNAME, fontfile=FONTFILE)

    # 1) collect spans + todo strings
    spans_per_page = []
    todo = set()

    for page in doc:
        d = page.get_text("dict")
        spans = []
        for block in d.get("blocks", []):
            if block.get("type", 0) != 0:
                continue
            for line in block.get("lines", []):
                for sp in line.get("spans", []):
                    t = norm(sp.get("text") or "")
                    if not t:
                        continue
                    bbox = sp.get("bbox")
                    if not bbox:
                        continue
                    fs = float(sp.get("size", 8.0))
                    spans.append((t, bbox, fs))
                    if should_translate(t) and glossary_override(t) is None and t not in cache:
                        todo.add(t)
        spans_per_page.append(spans)

    # 2) translate batches
    todo = sorted(todo)
    for i in range(0, len(todo), BATCH_SIZE):
        chunk = todo[i:i+BATCH_SIZE]
        mapping = translate_batch(client, chunk)
        cache.update(mapping)
        save_cache(CACHE_PATH, cache)

    # 3) insert Chinese in RED next to each bbox
    inserted = 0
    for page, spans in zip(doc, spans_per_page):
        page_w = page.rect.width

        for src, (x0, y0, x1, y1), fs in spans:
            if not should_translate(src):
                continue

            zh = glossary_override(src) or cache.get(src, "")
            zh = norm(zh)
            if not zh:
                continue

            zh_fs = max(6.0, fs * FONTSIZE_SCALE)

            # to the right (fallback below)
            zh_x0 = x1 + GAP_PT
            max_w = page_w - zh_x0 - RIGHT_MARGIN_PT

            if max_w < 20:
                zh_x0 = x0
                zh_y0 = y1 + 2
                rect = fitz.Rect(zh_x0, zh_y0, page_w - RIGHT_MARGIN_PT, zh_y0 + (y1 - y0) + 6)
            else:
                rect = fitz.Rect(zh_x0, y0, zh_x0 + max_w, y1 + 6)

            page.insert_textbox(
                rect,
                zh,
                fontname=FONTNAME,
                fontsize=zh_fs,
                color=(1, 0, 0),  # RED
                overlay=True,
                align=fitz.TEXT_ALIGN_LEFT,
            )
            inserted += 1

    doc.save(PDF_OUT)
    doc.close()
    print(f"Saved: {PDF_OUT} | inserted zh items: {inserted}")


if __name__ == "__main__":
    main()