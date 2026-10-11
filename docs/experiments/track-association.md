# Experiment — Local Track Association

## Purpose

Test whether a later local track is likely to be the same physical player as an earlier lost track.

## Current policy used for diagnostics

The initial conservative experiment uses explicit thresholds for DINOv2 similarity, HSV similarity, frame gap, and court movement.

## Verified result

`Track 2 -> Track 5` was the only time-valid ordered pair in the sample shot:

- DINOv2: `0.8888`
- HSV: `0.9907`
- elapsed frames: `25`
- court distance: `0.26 m`
- result: likely association

No function rewrites or merges the tracker IDs.
