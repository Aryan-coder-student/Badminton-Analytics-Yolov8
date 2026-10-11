# Experiment — ByteTrack on Shot 18

## Input

Shot 18, approximately `170.5–175.733` seconds.

## Process

- segment people at low confidence for tracking recall
- filter detections to an expanded court playing area
- retain the four strongest player masks
- update ByteTrack frame by frame
- render local tracker IDs using stable colors

## Output

A 1280x720, 30 FPS, 158-frame annotated video was generated.

## Result

Observed local IDs: `[1, 2, 3, 4, 5]`.

The near-left player fragmented from Track 2 to Track 5 during fast movement.
