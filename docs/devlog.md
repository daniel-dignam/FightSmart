# FightSmart — Development Log

Personal project: AI-assisted MMA highlight detection. This log records the
decisions, findings and validation so the work can be revised and defended.

## Goal

Take a full-length MMA fight video and produce a short list of **timestamped,
high-confidence highlight moments** that a coach or fan can review, ranked by
an excitement score.

## Stack & setup

- Python 3.12 (venv in `.venv`)
- OpenCV (frame reading + motion), scipy (audio), scikit-learn (ML)
- pandas / numpy / matplotlib (analysis), Streamlit (app layer)
- `pip install -r requirements.txt`

Run the test suite:
```
.\.venv\Scripts\activate
pytest            # 40 tests
```

## Repository map

```
src/
  video/loader.py      video metadata + frame access
  video/motion.py      motion feature extraction (mean-abs frame diff)
  audio/features.py    WAV loading + windowed RMS energy + dB
  detection/combine.py signal binning, boost, window finding, evaluation
  features/dataset.py  build a labelled feature matrix from signals
  models/baseline.py   Random Forest baseline (Phase 6)
  pipeline.py          end-to-end detect_highlights() used by the app
app.py                 Streamlit MVP (Phase 8)
notebooks/             one notebook per phase (02-05)
data/labels/           ground-truth + manual validation results
docs/roadmap.md        original phase plan
```

## Phase summaries

### Phase 1 — Environment
Python 3.12 venv + reproducible requirements + pytest baseline.

### Phase 2 — Video exploration
- Reference fight: Islam Makhachev vs Alexander Volkanovski 1 (UFC).
- Video: 1920×1080, 30 fps, 65,162 frames, ~36 min (2.16 GB, H.264).
- **Key decision:** the fight's highlights last >1s, so sample frames at a
  low rate (5 fps) instead of analysing all 65k frames — a ~6x compute saving
  with negligible detection loss.

### Phase 3 — Audio analysis
- Extracted audio to mono WAV (22.05 kHz), computed windowed RMS energy and dB.
- **Finding:** absolute loudness is a *weak* highlight signal for MMA —
  only ~55% of labelled events sit above the median energy (~chance). Cause:
  broadcast audio is compressed/limited and crowd noise is constant.
- A relative **energy boost** (energy vs a rolling baseline) is a better proxy.

### Phase 4 — Rule-based detection
- Built motion feature (mean-abs frame difference on downscaled grayscale).
- **Finding:** motion separates highlights better than audio (68% vs 55% of
  events above the median).
- A naive global threshold is a weak *absolute* detector (low precision) —
  the honest baseline. It motivated the ML phases.
- **Finding:** adding audio to the blend *reduced* F1 (its noise outweighed
  its signal) — audio is deliberately excluded from the app pipeline.

### Phase 5 — Dataset
- Cleaned the 60 manual labels (parsed times, fixed typos, separated
  structural boundary events from highlights).
- Built a per-0.5s-bin feature matrix (motion, motion boost, audio, audio
  boost) labelled positive within ±3s of a highlight.

### Phase 6 — ML baseline
- Random Forest (balanced class weights), reproducible stratified split.
- On the held-out test set, ML beat the rule-based threshold:
  rule-based F1 0.08 vs Random Forest F1 0.29.
- Honest caveats: single-fight proof of concept, per-bin evaluation, and the
  stratified split can leak adjacent bins of the same event (documented; the
  real fix is more fights).

### Phase 7 — Hardening (independent code review)
An independent agent reviewed the code (no shared context). Verdict: **pass**,
no security or logic errors. Applied its worthwhile suggestions:
- `relative_boost` now excludes the current bin from its own baseline (sharper
  spikes)
- `evaluate_windows` uses one-to-one greedy matching (no recall inflation)
- guards for empty signal arrays and non-positive window/hop
- edge-case tests added

### Phase 8 — Application layer
- `src/pipeline.detect_highlights()`: end-to-end video → highlight windows.
- `app.py`: Streamlit MVP — upload a video, detect highlights, preview clips.
- Uses the **generalizable rule-based motion detector** (the RF was a
  single-fight experiment) and **excludes audio** (Phase 3/4 finding).
- Clips are padded by a 1.5 s buffer so they show the full action sequence.

## Manual validation (the key result)

All 32 high-confidence predicted windows were checked against the raw footage:

| Round | Hits | Miss | Partial |
|-------|------|------|---------|
| 1     | 8    | 0    | 0       |
| 2     | 10   | 0    | 0       |
| 3     | 11   | 2    | 1       |
| **Total** | **29** | **2** | **1** |

**Precision on the high-confidence set: 29/32 = 90.6%** (30/32 = 93.75% if the
partial counts). Full results: `data/labels/validation_results.csv`.

What this means: when only high-confidence windows (peak score high, duration
≥ 1s) are surfaced, ~9 in 10 are genuine highlights — the right behaviour for
a product that shows a user a shortlist. The overall per-bin F1 (~0.29) is
lower because the whole timeline is busy; precision on what a user *sees* is
what matters.

The 1 partial was a window ending just before the action — fixed by adding a
clip buffer. The 2 misses (round 3, clocks 0:58 & 0:54) were high-confidence
false positives (likely camera cuts / high-motion non-highlights).

## Known limitations & next steps

- **Single-fight data.** The ML experiment used one fight; the app's
  rule-based path is generalisable, but a trained model needs more labelled
  fights to be trustworthy.
- **Region-blind motion.** Whole-frame motion can't tell a cage cut from a
  takedown. A spatial grid (e.g. a 5x5 region model) would cut false positives
  — a natural extension.
- **Round clock** for the app requires round-boundary times; on an arbitrary
  upload only absolute mm:ss is shown.
- **Performance:** motion extraction on a long video takes a few minutes.
  Future work: cache per-video features, or stream frames.

## How to run the app

```
streamlit run app.py
```
Upload a fight video in the sidebar; the app lists highlight windows and lets
you preview a clip of any window.
