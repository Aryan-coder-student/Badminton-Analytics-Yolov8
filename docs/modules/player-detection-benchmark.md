# Milestone 1 — Player detection and segmentation benchmark

## Goal

Before tuning YOLO, court filtering, or ReID, measure whether the existing
segmentation pipeline actually detects the on-court players and preserves their
pixel masks. The September full-match ReID audit measured **identity coverage**,
not ground-truth detector accuracy.

This benchmark deliberately uses absolute video frames and no player IDs.
Tracking identity correctness belongs to Milestone 2.

## Build the hand-labelled evaluation set

Start with roughly 100–200 **representative frames** from the same match video,
spread across the match rather than just consecutive frames. Include:

- clear full-court shots with all four players
- fast movement, jumps, occlusions, and overlapping players
- near- and far-court players, even when partially hidden
- camera transitions, close-ups, spectators, and officials
- late footage after Shot 313 where existing court gating stops

Keep a held-out portion from different shots or time ranges to verify that tuning
does not simply overfit the observed failures. Label the **visible on-court
players** on each frame. Negative/non-court frames can have zero labelled
players. Record the camera view for reliable category-level comparisons.

For each player label, supply a tight XYXY bounding box in **original video pixel
coordinates** and optionally a binary full-frame PNG segmentation mask.
Ground-truth masks must be pixel-aligned with predictions; the evaluator refuses
shape mismatches rather than resizing.

Keep the dataset outside Git, e.g. `data/benchmark/annotations.json`.

## Example annotation manifest

```json
{
  "video": "data/sample2.mp4",
  "frames": [
    {
      "frame_index": 5400,
      "camera_view": "wide",
      "players": [
        {
          "bbox": [420, 170, 480, 390],
          "mask_path": "ground_truth_masks/frame_5400_player_1.png"
        }
      ]
    },
    {
      "frame_index": 5410,
      "camera_view": "transition",
      "players": []
    }
  ]
}
```

Coordinates above are illustrative, **not real annotations**. Use zero-based
absolute video frame indices. The `mask_path` is relative to the JSON directory;
mask foreground pixels must be nonzero and backgrounds zero. Omit `mask_path`
until masks have been manually labelled.

## Step 1: Predict with existing YOLO segmentation

```bash
uv sync
uv run python -m badminton_analysis.evaluation.cli predict \
  --video data/sample2.mp4 \
  --annotations data/benchmark/annotations.json \
  --output data/benchmark/yolo_predictions.json \
  --confidence 0.1 \
  --device cpu
```

The predictor reuses `Segmenter("yolo")`, filters to the COCO person class
before YOLO mask upscaling, and processes one selected frame at a time to reduce
peak VRAM. It writes predicted full-frame mask PNGs plus bounding boxes and
confidence in the predictions JSON. Use `--device cuda:0` when supported and
`--model <checkpoint>` to evaluate alternative YOLO segmentation checkpoints.

**Limitation:** This measures *raw YOLO person segmentation*, not the full
court-filtered experimental path, because court filtering currently lives in
Git-ignored `scratchpad/` scripts. Off-court officials and spectators may count
as false positives. Evaluate the same frames after extracting the proven court
filter into version-controlled code in the follow-up PR; do not present raw YOLO
scores as final court-player detection accuracy.

## Step 2: Evaluate boxes and masks

```bash
uv run python -m badminton_analysis.evaluation.cli evaluate \
  --annotations data/benchmark/annotations.json \
  --predictions data/benchmark/yolo_predictions.json \
  --iou-threshold 0.5 \
  --output data/benchmark/yolo_metrics.json
```

Overall and per-`camera_view` outputs include:

- TP, FP, FN, precision, recall, F1 from one-to-one box IoU matching
- mean IoU of accepted box matches
- mean mask IoU for matched players with **both** masks present
- counts of reference masks, evaluated pairs, and missing prediction masks
- proportion of non-empty frames where *all* labelled players are matched
- `four_player_coverage` on frames with exactly four labelled players

The matcher maximizes eligible matches before optimizing their combined IoU,
so duplicate detections count as false positives. Missing prediction frames
count as zero detections. Predictions on unlabelled frames raise an error.
Undefined metrics (e.g. no labelled four-player frames or no masks) return
`null`, not a made-up success score.

The evaluation IoU threshold is not the model's confidence threshold.

## Exit criteria and next PR

1. Create/review the labelled evaluation set, keeping a held-out set.
2. Run raw segmentation and identify the largest false-negative/false-positive
   cases and low-mask-quality examples.
3. Extract the existing court filter into the tracked package and benchmark
   raw vs filtered outputs on exactly the same frames.
4. Compare confidence, model checkpoint and input resolution only after the
   baseline, including inference speed and memory use.
5. Set numeric acceptance gates after inspecting labelled results. High
   instance-level recall alone is insufficient if many full-four-player frames
   still miss someone.

**Current status:** This PR adds the reproducible evaluation infrastructure,
not verified detector improvements. No match video, masks, or ground-truth
annotations were available here to run actual model accuracy measurements.
