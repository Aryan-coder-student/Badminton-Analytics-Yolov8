# Current Implemented Pipeline

## Scope

This document describes what currently exists in code and scratchpad experiments. It does not describe future business/API orchestration.

## Data flow

```text
Recorded video
  ↓
Shot metadata / frame extraction
  ↓
Court keypoints
  ↓
Homography / court geometry
  ↓
YOLO segmentation
  ↓
Person filtering by court geometry
  ↓
ByteTrack
  ↓
TrackedSegmentation
  ↓
TrackAppearanceCollector
  ↓
DINOv2 + HSV track descriptors
  ↓
AssociationCandidate scoring
  ↓
Experimental cross-shot gallery matching
  ↓
P1/P2/P3/P4 or unresolved
  ↓
Render court points + player masks + boxes + identity labels
```

## Important identity boundary

The project distinguishes between a **local tracker ID** and an experimental
persistent player identity.

A ByteTrack `track_id` means only: “these detections are believed to be the same moving object across nearby consecutive frames.”

The association layer does not rename or merge tracker IDs. For example, Track 2 and Track 5 may be reported as a likely same-player predecessor/successor pair while remaining distinct local IDs in the raw tracking data.

## Experimental match-level identity

The cross-shot experiment adds an identity layer *above* ByteTrack. It builds a
gallery for each anonymous player (`P1`–`P4`) from accepted earlier tracks and
compares a new shot's local tracks with those galleries. A new track is only
labelled when it has both a sufficiently strong appearance score and a clear
margin over its second-best player. Otherwise it is explicitly `unresolved`.

The experiment resets ByteTrack at every shot boundary but retains the player
galleries. It is reusable experimental logic, not production/API code.

The rendering pass reads the saved per-frame identity JSON and replays only the
verified MAIN_COURT shots to recover masks and court geometry. This avoids
placing full-resolution masks in JSON and avoids rerunning DINOv2 matching.

## Why masks are preserved

The segmentation mask stays paired with the exact detection and track. This avoids losing the correspondence between:

- bounding box
- player pixels
- local track ID
- source frame

That pairing is important because appearance features must be extracted from the same player instance that the tracker identified.

## Current separation of responsibilities

### `model/*`
Owns interchangeable model adapters and their interfaces, such as embeddings, clustering, detection, segmentation, and tracking.

### `appearance/*`
Owns domain-specific appearance preparation and aggregation rather than a model adapter. It builds masked crops, HSV features, appearance observations, track summaries, and association evidence.

### `scratchpad/*`
Owns experiments, video replay, visualization, and smoke tests. It is not yet the production application layer.
