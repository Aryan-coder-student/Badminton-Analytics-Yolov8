# Module — Track Appearance

## What it does

For sampled observations of each local track, the project creates:

- the exact tracked segmentation
- source frame number
- masked RGB crop
- normalized DINOv2 embedding
- mask-aware HSV descriptor

`TrackAppearanceCollector` groups those observations by local `track_id` and aggregates them.

## Sampling

The default experiment samples every five frames per track.

## Current descriptor sizes

- DINOv2: 768 dimensions
- HSV: 96 dimensions

Both final aggregate descriptors are normalized.

## Structural validation

Appearance extraction currently rejects only structurally invalid inputs, including invalid/empty boxes, mismatched mask dimensions, and empty masks.

It does not currently apply subjective crop-quality thresholds.
