# Module — Player Segmentation and Court Filtering

## What it does

`YOLOSegmenter` uses `yolo11n-seg.pt` to return a `Segmentation` containing:

- existing `Detection` metadata
- a full-resolution boolean mask

Court filtering then uses each person's bottom-centre foot point and the detected court geometry to reject officials/spectators outside the playing area.

## Verified result

The sample frame produces four accepted court players.

## Why segmentation instead of only a box

The mask is reused later for appearance extraction so the descriptor focuses on the player rather than background court pixels.
