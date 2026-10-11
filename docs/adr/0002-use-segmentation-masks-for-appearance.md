# ADR 0002 — Use Segmentation Masks for Player Appearance Features

## Status

Accepted in the current appearance pipeline.

## Context

Bounding-box crops include court/background pixels. Player appearance comparison should emphasize the person rather than the surroundings.

## Decision

Preserve the YOLO segmentation mask and use masked player pixels for appearance feature preparation.

## Consequences

- DINOv2 receives a cleaner player-focused crop
- HSV color descriptors ignore background pixels
- the exact segmentation must remain paired with the corresponding tracked detection
