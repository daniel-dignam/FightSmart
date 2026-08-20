# FightSmart Roadmap

## Phase 1: Environment Setup

Goals:

- create a clean Python project structure
- configure VS Code for Python development
- set up Git and a virtual environment
- keep dependencies reproducible

Deliverable:

- working local project foundation

## Phase 2: Video Exploration

Questions to answer:

- what metadata is available from each video?
- how long are fights?
- what does a frame look like in practice?
- how should I sample frames and extract features?

Outputs:

- notebook with exploratory video inspection
- utility scripts for metadata loading

## Phase 3: Audio Analysis

Questions to answer:

- where are the loud moments in the fight?
- how do crowd sounds differ from fight sounds?
- what signal can I use as an excitement proxy?

Outputs:

- audio energy summaries
- simple excitement scoring notebook

## Phase 4: Rule-Based Detection

Questions to answer:

- what combinations of frame changes and audio spikes are useful?
- how do I turn raw signals into candidate highlight windows?

Outputs:

- candidate highlight timestamps
- first baseline system

## Phase 5: Dataset Creation

Questions to answer:

- what counts as a highlight?
- how do I label events consistently?
- what metadata should each label include?

Outputs:

- annotated dataset for model training

## Phase 6: ML

Goals:

- engineer features from audio and video signals
- train a baseline classifier
- validate on labeled examples

## Phase 7: Deep Learning

Goals:

- evaluate more advanced multimodal representations
- compare with classical approaches
- decide whether deeper models are justified

## Phase 8: Application Layer

Goals:

- create a minimal user interface
- let a user upload a fight video
- generate highlight outputs in a simple workflow
