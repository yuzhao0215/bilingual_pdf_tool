#!/usr/bin/env python3
import argparse
import csv
import json
import os
import re
from dataclasses import dataclass
from typing import Dict, List, Tuple

import fitz  # PyMuPDF
import yaml
from openai import OpenAI


def norm(s: str) -> str:
    # Normalize whitespace and keep original punctuation/case
    return " ".join((s or "").replace("\u00a0", " ").split()).strip()


def load_glossary(path: str) -> Dict[str, str]:
    if not path or not os.path.exists(path):
        return {}
    out = {}
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row or len(row) < 2:
                continue
            en = norm(row[0])
            zh = norm(row[1])
            if en:
                out[en] = zh
    return out


def load_cache(path: str) -> Dict[str, str]:
    if not path or not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_cache(path: str, cache: Dict[str, str]) -> None:
    if not path:
        return
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)


def pick_fontfile(cfg_fontfile: str) -> str:
    # If user sets it explicitly, trust it.
    if cfg_fontfile and os.path.exists(cfg_fontfile):
        return cfg_fontfile

    # Try common macOS PingFang locations
    mac_candidates = [
        "/System/Library/Fonts/PingFang.ttc",
        "/System/Library/Fonts/Supplemental/PingFang.ttc",
    ]
    for p in mac_candidates:
        if os.path.exists(p):
            return p

    # Fallback (Linux containers often have Noto CJK)
    linux_candidates = [
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc",
    ]
    for p in linux_candidates:
        if os.path.exists(p):
            return p

    local_candidates = [
        os.path.join("fonts", "PingFangSC.ttc"),
        "PingFangSC.ttc",
    ]
    for p in local_candidates:
        if os.path.exists(p):
            return p

    raise FileNotFoundError(
        "No CJK font file found. Set settings.yaml font.fontfile to your PingFang.ttc path."
    )


@dataclass
class Span:
    text: str
    bbox: Tuple[float, float, float, float]  # (x0, y0, x1, y1)
    fontsize: float


def extract_spans(page: fitz.Page) -> List[Span]:
    """
    Extract per-span text with bbox and fontsize.
    This is more precise than blocks for right-side placement.
    """
    d = page.get_text("dict")
    spans: List[Span] = []
    for block in d.get("blocks", []):
        for line in block.get("lines", []):
            for sp in line.get("spans", []):
                t = norm(sp.get("text", ""))
                if not t:
                    continue
                bbox = tuple(sp.get("bbox", (0, 0, 0, 0)))
                fs = float(sp.get("size", 10.0))
                spans.append(Span(text=t, bbox=bbox, fontsize=fs))
    return spans


def compile_filters(cfg: dict):
    filters = cfg["filters"]
    if "skip_regex_file" in filters:
        path = filters["skip_regex_file"]
        with open(path, "r", encoding="utf-8") as f:
            pats = [line.strip() for line in f if line.strip() and not line.strip().startswith("#")]
        skip = [re.compile(p) for p in pats]
    else:
        skip = [re.compile(p) for p in filters["skip_regex"]]

    min_chars = int(filters.get("min_chars", 2))
    max_chars = int(filters.get("max_chars", 999999))
    return skip, min_chars, max_chars


def should_translate(t: str, skip_res, min_chars: int, max_chars: int) -> bool:
    if len(t) < min_chars or len(t) > max_chars:
        return False
    for rx in skip_res:
        if rx.match(t):
            return False
    # Extra: skip "ID-like" short tokens
    if len(t) <= 6 and re.fullmatch(r"[A-Z0-9]+", t):
        return False
    return True


def apply_glossary_exact(t: str, glossary: Dict[str, str]) -> str | None:
    # exact match on normalized string
    if t in glossary:
        return glossary[t]
    # common variant: drop trailing periods
    if t.endswith(".") and t[:-1] in glossary:
        return glossary[t[:-1]]
    return None


def translate_batch(client: OpenAI, texts: List[str], glossary: Dict[str, str], model: str) -> Dict[str, str]:
    """
    Translate a list of English strings to Simplified Chinese.
    Returns mapping src->zh.
    Enforces glossary via explicit instructions + post-override on exact terms.
    """
    # Build glossary hints (short to keep prompt small)
    gloss_lines = [f'- "{k}" -> "{v}"' for k, v in list(glossary.items())[:400]]
    gloss_hint = "\n".join(gloss_lines)

    # Ask for strict JSON
    prompt = f"""You are translating architectural drawing text from English to Simplified Chinese.
Requirements:
- Keep numbers, units, and codes unchanged (e.g., 3/A401, TK-PS03-01, 12'-6", 6").
- Keep sheet/detail references unchanged.
- Use formal construction/engineering Chinese.
- MUST follow these fixed translations when they appear as standalone terms:
{gloss_hint}

Return ONLY valid JSON with this schema:
{{
  "items": [{{"src": "...", "zh": "..."}}]
}}

Translate these items:
{json.dumps(texts, ensure_ascii=False)}
"""

    resp = client.responses.create(
        model=model,
        input=prompt,
        temperature=0.2,
    )
    # The SDK returns text; parse JSON
    content = resp.output_text
    data = json.loads(content)
    out = {}
    for it in data.get("items", []):
        src = norm(it.get("src", ""))
        zh = norm(it.get("zh", ""))
        if not src:
            continue
        # enforce exact glossary override (strong)
        forced = apply_glossary_exact(src, glossary)
        out[src] = forced if forced else zh
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Input PDF (bare filename resolves under input/)")
    ap.add_argument("--output", required=True, help="Output PDF (bare filename resolves under output/)")
    ap.add_argument("--settings", default="settings.yaml", help="YAML settings file")
    ap.add_argument("--glossary", default=os.path.join("glossary", "glossary.csv"), help="Glossary CSV")
    args = ap.parse_args()
    input_path = args.input if os.path.isabs(args.input) or os.path.dirname(args.input) else os.path.join("input", args.input)
    output_path = args.output if os.path.isabs(args.output) or os.path.dirname(args.output) else os.path.join("output", args.output)
    glossary_path = args.glossary if os.path.isabs(args.glossary) or os.path.dirname(args.glossary) else os.path.join("glossary", args.glossary)

    cfg = yaml.safe_load(open(args.settings, "r", encoding="utf-8"))
    glossary = load_glossary(glossary_path)

    fontfile = pick_fontfile(cfg["font"].get("fontfile", ""))
    gap_pt = float(cfg["layout"]["gap_pt"])
    right_margin = float(cfg["layout"]["right_margin_pt"])
    below_gap = float(cfg["layout"]["fallback_below_gap_pt"])
    fontsize_scale = float(cfg["font"]["fontsize_scale"])
    cache_path = cfg["translation"]["cache_path"]
    model = cfg["translation"]["model"]
    batch_size = int(cfg["translation"]["batch_size"])

    skip_res, min_chars, max_chars = compile_filters(cfg)
    cache = load_cache(cache_path)

    # Collect all translatable unique strings
    doc = fitz.open(input_path)
    all_spans_per_page: List[List[Span]] = []
    todo_set = set()

    for page in doc:
        spans = extract_spans(page)
        all_spans_per_page.append(spans)
        for sp in spans:
            t = sp.text
            if not should_translate(t, skip_res, min_chars, max_chars):
                continue
            if apply_glossary_exact(t, glossary) is not None:
                continue  # already covered
            if t not in cache:
                todo_set.add(t)

    todo = sorted(todo_set)
    if todo:
        client = OpenAI()
        for i in range(0, len(todo), batch_size):
            chunk = todo[i:i+batch_size]
            mapping = translate_batch(client, chunk, glossary, model=model)
            cache.update(mapping)
            save_cache(cache_path, cache)
    else:
        # still ensure cache saved path exists for later
        save_cache(cache_path, cache)

    # Apply overlay text directly into the same PDF, then save a copy
    for page_index, page in enumerate(doc):
        page_w = page.rect.width
        spans = all_spans_per_page[page_index]
        for sp in spans:
            src = sp.text
            if not should_translate(src, skip_res, min_chars, max_chars):
                continue

            zh = apply_glossary_exact(src, glossary)
            if zh is None:
                zh = cache.get(src, "")
            zh = norm(zh)
            if not zh:
                continue

            x0, y0, x1, y1 = sp.bbox
            zh_fs = max(6.0, sp.fontsize * fontsize_scale)

            # Primary placement: to the right
            zh_x0 = x1 + gap_pt
            max_w = page_w - zh_x0 - right_margin
            # If no room, fallback below
            if max_w < 30:
                zh_x0 = x0
                y_shift = (y1 - y0) + below_gap
                zh_y0 = y0 + y_shift
                zh_rect = fitz.Rect(zh_x0, zh_y0, page_w - right_margin, zh_y0 + 3*(y1-y0))
            else:
                zh_rect = fitz.Rect(zh_x0, y0, zh_x0 + max_w, y1 + 2*(y1-y0))

            # Insert Chinese text
            page.insert_textbox(
                zh_rect,
                zh,
                fontfile=fontfile,
                fontsize=zh_fs,
                color=tuple(cfg["font"]["color_rgb"]),
                overlay=True,
                align=fitz.TEXT_ALIGN_LEFT,
            )

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    doc.save(output_path)
    doc.close()
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
