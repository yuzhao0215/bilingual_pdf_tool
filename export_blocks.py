import os
import csv
import fitz  # PyMuPDF

file_name = "archi_Extracted_p28.pdf"
PDF_IN = "./input/" + file_name
CSV_OUT = "./output/" + file_name.split('.')[0] + "_blocks.csv"

# optional: ignore very small blocks
MIN_CHARS = 5

def block_text(block: dict) -> str:
    parts = []
    for line in block.get("lines", []):
        for sp in line.get("spans", []):
            t = (sp.get("text") or "").strip()
            if t:
                parts.append(t)
    # Keep spaces; you can also join with "\n" if you prefer
    return " ".join(parts)

doc = fitz.open(PDF_IN)

rows = []
block_id = 0
for p_idx in range(doc.page_count):
    page = doc[p_idx]
    d = page.get_text("dict")

    for b in d.get("blocks", []):
        if b.get("type", 0) != 0:
            continue
        bbox = b.get("bbox")
        if not bbox:
            continue
        text = block_text(b).strip()
        if len(text) < MIN_CHARS:
            continue

        block_id += 1
        rows.append({
            "id": f"b{block_id:06d}",
            "page": p_idx + 1,   # 1-based
            "x0": bbox[0],
            "y0": bbox[1],
            "x1": bbox[2],
            "y1": bbox[3],
            "text": text,
            "label_interest": "",   # blank for labeling
        })

doc.close()

with open(CSV_OUT, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else
                       ["id","page","x0","y0","x1","y1","text","label_interest"])
    w.writeheader()
    w.writerows(rows)

print("Saved:", CSV_OUT, "rows:", len(rows))