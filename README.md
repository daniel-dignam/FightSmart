# FightSmart

FightSmart is a personal AI-assisted MMA highlight detection project. The goal is to learn the full software lifecycle while building a working MVP that can identify interesting moments in full-length fights using a combination of video, audio, and rule-based analysis.

## Project Vision

- Input: full MMA fight video
- Output: timestamped highlight moments with confidence scores
- Core ideas:
  - video analysis
  - audio excitement detection
  - event heuristics
  - structured labeling
  - machine learning classifier
  - future multimodal AI reasoning

## Current Development Philosophy

I am deliberately building iteratively:

1. Working script
2. Working pipeline
3. Rule-based solution
4. Traditional ML
5. Deep learning
6. AI-assisted reasoning

This means I am not jumping directly to deep learning. I am learning by building a solid baseline first.

## Repository Structure

```text
FightSmart/
├── .gitignore
├── .vscode/
│   └── settings.json
├── data/
│   ├── raw/
│   ├── processed/
│   └── labels/
├── docs/
│   └── roadmap.md
├── notebooks/
├── src/
│   ├── __init__.py
│   ├── audio/
│   │   └── __init__.py
│   ├── video/
│   │   └── __init__.py
│   ├── features/
│   │   └── __init__.py
│   ├── models/
│   │   └── __init__.py
│   └── utils/
│       └── __init__.py
├── tests/
│   ├── __init__.py
│   └── test_smoke.py
├── requirements.txt
├── README.md
└── .venv/
```

## Phase Roadmap

### Phase 1: Environment Setup

- Set up VS Code
- Install Python
- Create a virtual environment
- Initialize repository structure
- Create reproducible requirements

### Phase 2: Video Exploration

- Load MMA videos
- Inspect metadata
- Extract sample frames
- Learn the video processing pipeline

### Phase 3: Audio Analysis

- Extract audio streams
- Measure energy and volume spikes
- Detect crowd excitement

### Phase 4: Rule-Based Highlight Detection

- Combine visual and audio heuristics
- Generate candidate highlight timestamps

### Phase 5: Dataset Creation

- Manually label highlight moments
- Turn event notes into a trainable dataset

### Phase 6: Machine Learning

- Engineer features
- Train classifiers
- Evaluate with metrics

### Phase 7: Deep Learning

- Explore video/audio embeddings
- Build multimodal models

### Phase 8: Application Layer

- Streamlit MVP
- Upload video
- Generate highlights

## Engineering Standards

I will practice good software engineering habits:

- keep code simple and readable
- use small, testable functions
- document assumptions clearly
- validate experiments with reproducible scripts
- prefer practical tools over hype-driven tooling
- keep notebooks for exploration, keep reusable logic in source files

## Local Setup

### 1. Create a virtual environment

```powershell
python -m venv .venv
```

### 2. Activate it

```powershell
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Run tests

```powershell
pytest
```

## Recommended Workflow

- use small feature branches in Git
- commit often with meaningful messages
- keep notebooks focused on exploration
- move reusable logic into the src package
- write tests for behavior that matters
- document major decisions in docs/

## Mentorship Notes

This project is not just about building the app. It is also about rebuilding strong engineering fundamentals.

I will focus on:

- Python habits and clarity
- project structure and packaging
- debugging and experimentation
- data workflows and reproducibility
- practical ML without overengineering

## Next Step

Phases 1-8 are implemented: signal extraction, an honest rule-based baseline,
a supervised ML comparison, hardening after an independent code review, and a
Streamlit MVP (`app.py`). See `docs/devlog.md` for the full record, including
the manual validation result (29/32 = 90.6% precision on the high-confidence
set). Natural next steps are more labelled fights and a spatial/regional
motion model.
