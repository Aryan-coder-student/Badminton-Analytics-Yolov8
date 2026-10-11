# Experiment — Offline Track Appearance Descriptors

## Purpose

Build a visual representation for every ByteTrack local track using multiple observations rather than a single frame.

## Recorded result

With sampling interval five:

- Track 1: 30 observations
- Track 2: 3 observations
- Track 3: 32 observations
- Track 4: 32 observations
- Track 5: 22 observations

Every final DINOv2 descriptor has 768 dimensions and every HSV descriptor has 96 dimensions. Aggregate norms were verified to be 1.0 within float32 precision.

Track 2 and Track 5 intentionally remained separate at this stage.
