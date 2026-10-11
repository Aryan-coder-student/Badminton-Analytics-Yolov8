# Badminton Court Visualization

## Overview

The implementation in `scratchpad/test_court.py` detects 30 badminton-court
keypoints and creates a clean court overlay. It does not use Ultralytics'
default annotated image, so the `item` label and blue bounding box are not
included.

The processing flow is:

```text
Source frame
    -> YOLO pose inference
    -> 30 detected keypoints
    -> projective court-grid correction
    -> custom OpenCV line rendering
    -> output image
```

## Model Loading and Inference

The court checkpoint is downloaded through `hf_hub_download()` and loaded with
Ultralytics' standard `YOLO` class using `task="pose"`.

```python
checkpoint = hf_hub_download(MODEL_REPOSITORY, MODEL_FILENAME)
model = YOLO(checkpoint, task="pose")
result = model.predict(SOURCE_IMAGE, verbose=False)[0]
```

The bounding box is still useful internally for selecting the highest-confidence
court when multiple detections exist. It is never drawn. Rendering starts from
`result.orig_img`, which is the clean source image.

## Reconstructing the Court

The model returns 30 points representing six cross-court rows with five
intersections per row. The canonical grid uses regulation court measurements:

- Width positions: `0.00, 0.46, 3.05, 5.64, 6.10` metres.
- Length positions: `0.00, 0.76, 4.72, 8.68, 12.64, 13.40` metres.

Some raw predictions can be duplicated or misplaced. Connecting them directly
produces diagonal artifacts. To prevent this, the implementation sorts two
reliable detected rows from left to right and fits a homography with
`cv2.findHomography()`. `cv2.perspectiveTransform()` then projects the complete
canonical grid into the camera view. This provides straight, consistent court
geometry even when individual keypoints are noisy.

## Drawing the Lines

The custom renderer draws:

- All six horizontal court lines.
- Left and right doubles sidelines.
- Left and right singles sidelines.
- Separate centre-service lines on each side of the net.
- A small marker at every projected intersection.

Lines and points are rendered with anti-aliasing using `cv2.line()` and
`cv2.circle()`. The completed image is written directly with `cv2.imwrite()`.

## Running the Script

From the repository root, run:

```bash
uv run python -m scratchpad.test_court
```

Current input:

```text
data/shot_frames/shot_0018_75.jpg
```

Generated output:

```text
data/court_predictions/test_court/shot_0018_75.jpg
```

The source frame is never modified. Change `SOURCE_IMAGE` and `OUTPUT_IMAGE` in
`scratchpad/test_court.py` to process a different frame.
