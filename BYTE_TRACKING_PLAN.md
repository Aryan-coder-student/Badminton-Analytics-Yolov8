# ByteTrack Court-Player Tracking Plan

## Goal

Process a continuous main-court window from `data/sample2.mp4`, segment people,
keep the four players inside the court ROI, assign persistent ByteTrack IDs, and
write an annotated video with `track_id` above each player mask.

Re-identification, appearance clustering, segmentation-model changes, and
cross-shot identity recovery are outside this step.

## Proposed Flow

```text
Sequential video frame
    ├── first frame -> court pose model -> court polygon (reused)
    └── YOLOSegmenter -> person boxes + masks
                         -> inside-court ROI filter
                         -> four player detections
                         -> ByteTrack.update(...)
                         -> TrackedDetection
                         -> draw track_id on matching mask
                         -> output video
```

Court filtering happens before ByteTrack so officials, spectators, and background
staff never enter its state. Run segmentation at confidence `0.1` so ByteTrack can
still use its low-confidence association stage. The fixed court polygon should be
stable across this main-camera sequence; camera cuts are outside this prototype.

## Data and Model Boundaries

1. Add `src/badminton_analysis/model/tracking/interface.py`:
   - `TrackedDetection` containing `detection: Detection` and `track_id: int`.
   - `TrackingModel.update(detections, frame_shape)` for one ordered frame at a
     time.
   - Do not add `track_batch`; temporal order and persistent state are required.
2. Add `tracking/inference/bytetrack.py`:
   - Own one stateful Ultralytics `BYTETracker` instance.
   - Convert filtered `Detection` objects to the tracker input shape.
   - Use ByteTrack's returned source index to pair each `track_id` with the exact
     input `Detection`; keep Ultralytics-specific array shapes inside this adapter.
3. Add `model/trackers.py` with a minimal `Tracker("bytetrack")` registry and
   delegation façade, consistent with `Detector` and `Segmenter`.
4. Reuse `Detection`, `get_foot_point()`, `create_court_polygon()`, and
   `draw_court_keypoints()`; do not duplicate their geometry logic.

`track_id` must remain distinct from YOLO's `class_id`: the former identifies an
object over time, while the latter identifies the object category.

## Implementation Tasks

1. Add the missing optional tracking dependency with `uv add "lap>=0.5.12"` and
   commit both `pyproject.toml` and `uv.lock`. The installed Ultralytics tracker
   currently cannot import because `lap` is absent.
2. Add only the tracker configuration name to settings. Load ByteTrack's bundled
   YAML instead of copying every threshold into application settings.
3. Create `scratchpad/test_player_tracking.py`:
   - Read sequential frames with `cv2.VideoCapture`.
   - Detect the court once on the first frame because the sample uses a fixed
     broadcast camera.
   - Run `Segmenter("yolo", options={"confidence_threshold": 0.1})` per frame.
   - Filter `person` segmentations with the existing foot-point court test before
     feeding their detections to the same tracker instance.
   - Pair each tracked detection back to its segmentation mask.
   - Draw `Player {track_id}` above the mask and choose a deterministic color from
     `track_id`.
   - Preserve input resolution and FPS with `cv2.VideoWriter`.
4. The default demo processes shot 18 (`170.5`–`175.733` seconds) and writes
   `data/player_predictions/bytetrack/sample2_shot_0018_tracked.mp4`. Command-line
   input, output, start, and end arguments support other continuous windows.
5. Append the completed result, command, validation, and known ID switches to
   `WORK_LOG.md`.

## Tests

- Unit-test conversion of filtered detections into ByteTrack input and
  `TrackedDetection` output.
- Assert the same fake ByteTrack ID survives across consecutive calls.
- Assert court filtering excludes outside people before `TrackingModel.update()`.
- Assert the returned detection pairs with its original segmentation mask.
- Assert rendering uses `track_id`, never list position or `class_id`.
- Run the existing suite with `uv run pytest`.
- Run a short real-video smoke test before processing the full match.

## Acceptance Criteria

- The output video opens and matches the source FPS and resolution, with one
  output frame per frame in the requested window.
- Exactly the four in-court player masks are sent to ByteTrack and highlighted for
  the doubles sample.
- Labels display actual ByteTrack IDs and remain stable while a player stays
  continuously visible.
- Court keypoints remain visible; generated court lines remain hidden.
- Similar players keep the same segmentation colors only when their track IDs
  match; DINOv2, KMeans, and re-identification remain outside this phase.
- Segmentation remains in its existing model layer; the tracker consumes only
  filtered `Detection` values.
- All automated tests pass and the dated work log is updated.
