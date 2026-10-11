# Badminton Analysis — Current Documentation

This directory documents only functionality that has already been implemented and verified in the project. It intentionally does not describe planned business/API layers, shuttle tracking, pose/action recognition, team assignment, player naming, scoring, or commentary.

## Current implemented pipeline

```text
Recorded badminton video
  ↓
Shot boundaries / sampled shot frames
  ↓
Court keypoint detection + homography
  ↓
Player segmentation
  ↓
Court-based player filtering
  ↓
ByteTrack local tracking
  ↓
Track appearance collection
  ↓
Within-shot track association diagnostics
  ↓
Experimental cross-shot P1–P4 gallery matching
```

The current system is still in an experimentation/prototyping phase. Reusable model adapters and feature-processing code live under `src/badminton_analysis/`, while end-to-end checks are exercised from `scratchpad/`.

## Documents

- `architecture/current-pipeline.md` — implemented data flow and component boundaries.
- `model-catalog.md` — models and algorithms currently implemented and why they are used.
- `modules/` — module-by-module behavior.
- `modules/cross-shot-reid.md` — experimental persistent anonymous-player
  matching, helper functions, and scratchpad commands.
- `experiments/` — verified experiments and observed results.
- `audits/` — dated output-quality audits with measured issues and limitations.
- `audits/match-reid-output-2026-09-13.md` — audit of the current full-match
  identity MP4 and its per-frame JSON results.
- `audits/match-reid-rerun-2026-09-19.md` — measured diagnosis of unresolved
  identities after the stricter court-view rerun.
- `adr/` — architectural decisions already made.
- `glossary.md` — project terms used in code and documentation.
