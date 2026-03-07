import io
import fitz  # PyMuPDF
import pandas as pd
import streamlit as st
from PIL import Image

st.set_page_config(page_title="PDF Block Labeler", layout="wide")
st.title("PDF Block Labeler (Interested vs Not)")

uploaded_pdf = st.file_uploader("Upload PDF", type=["pdf"])
uploaded_csv = st.file_uploader("Upload blocks CSV", type=["csv"])

zoom = st.slider("Render zoom", 1.0, 4.0, 2.0, 0.25)
margin_pts = st.slider("Crop margin (pt)", 0.0, 60.0, 15.0, 1.0)
only_unlabeled = st.checkbox("Navigate only unlabeled", value=True)

REQUIRED = ["page","x0","y0","x1","y1","text"]

def ensure_cols(df: pd.DataFrame) -> pd.DataFrame:
    miss = [c for c in REQUIRED if c not in df.columns]
    if miss:
        raise ValueError(f"Missing columns: {miss}")
    if "id" not in df.columns:
        df = df.copy()
        df["id"] = [f"row_{i}" for i in range(len(df))]
    if "label_interest" not in df.columns:
        df = df.copy()
        df["label_interest"] = pd.NA
    return df

@st.cache_data(show_spinner=False)
def load_pdf_bytes(up) -> bytes:
    return up.read()

@st.cache_data(show_spinner=False)
def load_df(up) -> pd.DataFrame:
    return ensure_cols(pd.read_csv(up))

def render_page_pix(pdf_bytes: bytes, page_index: int, zoom: float):
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    page = doc[page_index]
    pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
    doc.close()
    return pix

def crop_bbox_image(pix, bbox_pts, zoom: float, margin_pts: float):
    x0, y0, x1, y1 = bbox_pts
    m = margin_pts
    x0m, y0m, x1m, y1m = x0 - m, y0 - m, x1 + m, y1 + m

    px0 = int(max(0, x0m * zoom))
    py0 = int(max(0, y0m * zoom))
    px1 = int(min(pix.width,  x1m * zoom))
    py1 = int(min(pix.height, y1m * zoom))

    if px1 <= px0 or py1 <= py0:
        return None

    img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
    return img.crop((px0, py0, px1, py1))

def goto_next(df, idx, only_unlabeled):
    j = idx + 1
    while j < len(df):
        if (not only_unlabeled) or pd.isna(df.at[j, "label_interest"]):
            return j
        j += 1
    return idx

def goto_prev(df, idx, only_unlabeled):
    j = idx - 1
    while j >= 0:
        if (not only_unlabeled) or pd.isna(df.at[j, "label_interest"]):
            return j
        j -= 1
    return idx

if not uploaded_pdf or not uploaded_csv:
    st.stop()

pdf_bytes = load_pdf_bytes(uploaded_pdf)
df = load_df(uploaded_csv)

if "idx" not in st.session_state:
    st.session_state.idx = 0
    if only_unlabeled:
        for i in range(len(df)):
            if pd.isna(df.at[i, "label_interest"]):
                st.session_state.idx = i
                break

if "df" not in st.session_state:
    st.session_state.df = df
else:
    if len(st.session_state.df) != len(df):
        st.session_state.df = df
        st.session_state.idx = 0

df = st.session_state.df
idx = st.session_state.idx
row = df.iloc[idx]

page_num = int(row["page"])
page_index = page_num - 1
bbox = (float(row["x0"]), float(row["y0"]), float(row["x1"]), float(row["y1"]))
text = str(row["text"])

total = len(df)
labeled = df["label_interest"].notna().sum()
st.caption(f"Blocks: {total} | Labeled: {labeled} | Remaining: {total - labeled}")

left, right = st.columns([1, 1], gap="large")

with left:
    st.subheader("Block crop")
    pix = render_page_pix(pdf_bytes, page_index, zoom)
    crop = crop_bbox_image(pix, bbox, zoom, margin_pts)
    if crop is None:
        st.warning("Invalid bbox crop.")
    else:
        st.image(crop, use_container_width=True)

with right:
    st.subheader("Block text")
    st.write(f"**Row:** {idx+1}/{total} | **ID:** {row['id']} | **Page:** {page_num}")
    st.text_area("Extracted block text", value=text, height=220)
    st.write("**Current label:**", row.get("label_interest", pd.NA))

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        if st.button("✅ Interested (1)", use_container_width=True):
            df.at[idx, "label_interest"] = 1
            st.session_state.idx = goto_next(df, idx, only_unlabeled)
    with c2:
        if st.button("❌ Not interested (0)", use_container_width=True):
            df.at[idx, "label_interest"] = 0
            st.session_state.idx = goto_next(df, idx, only_unlabeled)
    with c3:
        if st.button("⏭ Skip", use_container_width=True):
            st.session_state.idx = goto_next(df, idx, only_unlabeled)
    with c4:
        if st.button("⬅ Back", use_container_width=True):
            st.session_state.idx = goto_prev(df, idx, only_unlabeled)

st.divider()
out_csv = io.StringIO()
df.to_csv(out_csv, index=False)
st.download_button(
    "⬇️ Download labeled blocks CSV",
    data=out_csv.getvalue().encode("utf-8"),
    file_name="blocks_labeled.csv",
    mime="text/csv",
)