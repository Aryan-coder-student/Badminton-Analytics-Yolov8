# Module — Court Reconstruction

## What it does

A YOLO pose checkpoint predicts 30 badminton-court keypoints. The detected image points are mapped to a regulation court through homography.

## Why

Court geometry provides a stable reference independent of image perspective. It is used for:

- identifying the playable region
- projecting player foot positions to court coordinates
- later association diagnostics based on physical distance

## Current visualization behavior

Court keypoints are preserved in debug output. Experiments intentionally avoid relying on the model's default item box, and the court reconstruction prototype can render custom geometry from the homography.
