# Bilingual PDF Generator (English + Simplified Chinese)
This tool takes a **vector (selectable text) PDF**, extracts English text with coordinates, translates it with the OpenAI API
(with glossary enforcement), then writes **Chinese to the right** of the original text and saves a single merged bilingual PDF.

## Install
```bash
python -m venv .venv
source .venv/bin/activate
pip install -U openai pymupdf pyyaml
```

## Set API key
```bash
export OPENAI_API_KEY="YOUR_KEY"
```

## Prepare glossary (optional but recommended)
Edit `glossary/glossary.csv`:
- GENERAL NOTES,一般说明
- TYP.,典型
- VERIFY IN FIELD,现场核实

## Configure font + placement
Edit `settings.yaml`.

### PingFang SC font path on macOS
Common paths (pick the one that exists on your Mac):
- /System/Library/Fonts/PingFang.ttc
- /System/Library/Fonts/Supplemental/PingFang.ttc

Quick check:
```bash
ls /System/Library/Fonts | grep -i pingfang
ls /System/Library/Fonts/Supplemental | grep -i pingfang
```

## Run
```bash
python bilingualize.py \
  --input "input.pdf" \
  --output "output_bilingual.pdf" \
  --settings settings.yaml \
  --glossary glossary/glossary.csv
```

## Notes / limitations
- This is optimized for *vector PDFs*. If a sheet is scanned, extractable text may be empty.
- For tight areas near the right margin, the tool falls back to placing Chinese **below** the English.
- Drawings contain many IDs/dimensions that should not be translated; see `settings.yaml` skip patterns.
