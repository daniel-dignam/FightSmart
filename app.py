"""FightSmart — Streamlit MVP (Phase 8).

Run from the project root with:
    streamlit run app.py

Upload a fight video; the app detects high-confidence highlight windows
using the motion-based rule detector and lets you preview a clip of any
window (clips include a 1.5 s buffer so they show the full sequence).
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st

# Make the project's src/ package importable regardless of how Streamlit
# sets the working directory.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.pipeline import detect_highlights, extract_clip  # noqa: E402

# Reject unreasonably large uploads so the app can't fill the disk.
MAX_UPLOAD_BYTES = 2 * 1024 * 1024 * 1024  # 2 GB


st.set_page_config(page_title="FightSmart", layout="wide")
st.title("\U0001F94B FightSmart")
st.caption(
    "Rule-based MMA highlight detection: motion boost vs a rolling baseline. "
    "High-confidence windows only, with a 1.5 s clip buffer."
)

uploaded = st.sidebar.file_uploader("Upload a fight video (.mp4)", type=["mp4"])
local_path = st.sidebar.text_input("...or use a local file path", "")
high_conf = st.sidebar.slider("High-confidence peak threshold", 1.2, 3.0, 1.8, 0.1)
pad_s = st.sidebar.slider("Clip buffer (s)", 0.0, 3.0, 1.5, 0.5)

# Resolve the video source: a local path (streamed, no upload) or an upload.
video_path: Path | None = None
local_path = (local_path or "").strip()
if local_path:
    video_path = Path(local_path)
    if not video_path.exists():
        st.error(f"Local file not found: {video_path}")
        st.stop()
elif uploaded is not None:
    if getattr(uploaded, "size", 0) > MAX_UPLOAD_BYTES:
        st.error(
            f"File too large ({uploaded.size / 1e9:.1f} GB). "
            f"Please upload under {MAX_UPLOAD_BYTES / 1e9:.0f} GB."
        )
        st.stop()
    tmp = Path(tempfile.mkdtemp(prefix="fightsmart_app_"))
    # Sanitise the filename so a path in the upload name can't escape tmp.
    safe_name = Path(uploaded.name or "fight.mp4").name
    video_path = tmp / safe_name
    video_path.write_bytes(uploaded.getvalue())

if video_path is None:
    st.info("Upload a fight video or enter a local file path to detect highlights.")
    st.stop()

scratch = Path(tempfile.mkdtemp(prefix="fightsmart_clips_"))
with st.spinner(
    "Analysing video. Motion extraction on a long video takes a few minutes..."
):
    result = detect_highlights(video_path, high_conf_peak=high_conf, pad_s=pad_s)

st.success(
    f"Done — {result['n_windows']} high-confidence highlights "
    f"in a {result['duration_s']:.0f}s video."
)

st.subheader("Highlight windows")
df = pd.DataFrame(result["windows"])
st.dataframe(df[["start_mmss", "end_mmss", "duration_s", "peak_score"]])

if result["windows"]:
    options = [
        f"{w['start_mmss']} - {w['end_mmss']}  (peak {w['peak_score']:.2f})"
        for w in result["windows"]
    ]
    choice = st.selectbox("Preview a clip", options)
    idx = options.index(choice)
    w = result["windows"][idx]
    clip = scratch / f"clip_{idx}.mp4"
    with st.spinner("Cutting clip..."):
        extract_clip(video_path, w["start_s"], w["end_s"], clip)
    st.video(str(clip))
