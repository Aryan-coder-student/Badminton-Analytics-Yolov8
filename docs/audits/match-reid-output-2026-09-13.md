# Match ReID Output Audit — 2026-09-13

## Audited artifacts

- Video: `data/player_predictions/match_reid/sample2_match_persistent_identities.mp4`
- Identity data: `data/player_predictions/match_reid/frame_identities.json`
- Source: `data/sample2.mp4`

Audited snapshot SHA-256:

- MP4: `c18c0278cb50b9e86c9c5f9be3b52e13d77762c52b76166d5f1759e92e7ae9df`
- JSON: `6423a086f42855c005b4a38a0f1cdf85cfa9ca90cad231587765f750e6083b4b`

The MP4 is structurally complete and decodes without errors, but it does **not**
yet meet the goal of showing reliable, stable `P1/P2/P3/P4` identities throughout
the match. This audit records the current output before any tuning or fixes.

## Summary metrics

| Metric | Result |
| --- | ---: |
| Source/output duration | 3,250.03 seconds |
| Source/output video frames | 97,501 |
| Frames containing tracking records | 43,737 (44.86%) |
| Saved player observations | 118,647 |
| Assigned `P1`–`P4` observations | 45,680 (38.50%) |
| Unresolved observations | 72,967 (61.50%) |
| Annotated frames with an unresolved player | 39,886 (91.20%) |
| Frames with four tracked detections | 15,454 (35.33%) |
| Frames with exactly `P1/P2/P3/P4` | 105 (0.24%) |
| Accepted court-shot candidates | 226 |
| Accepted shots containing unresolved tracks | 213 (94.25%) |

Assigned observations are uneven: `P1=16,901`, `P2=17,450`, `P3=6,751`, and
`P4=4,578`. These counts measure label coverage, not identity correctness.

## Findings

### Blocker — Persistent identity coverage is too low

Most detections display `unresolved`, especially later in the match. Of the
15,454 frames where four players are tracked, only 105 frames (0.68%) contain
the complete four-identity set. The output therefore does not maintain visible
`P1/P2/P3/P4` identities across the recorded match.

Examples:

- Shot 18, 00:02:53.13: all four IDs are visible; this is the good baseline.
- Shot 75, 00:10:52.27: two of four tracks are unresolved.
- Shot 120, 00:16:55.53: two of four tracks are unresolved.
- Shot 270, 00:36:29.20: all three visible tracks are unresolved.

### High — Non-standard camera views are treated as `MAIN_COURT`

The processed set contains views that are unsuitable for four-player court ReID:

- Shot 373, 00:51:41.63: tight single-player view; only `P1` is rendered.
- Shot 472, 00:53:24.13: low side-angle view; only `P1` is rendered.

The current verifier checks a court detection on one midpoint frame and only
tests a near/far width ratio. It does not require a complete, stable wide-court
view. These false-positive shots feed unsuitable views into tracking and accepted
appearance galleries.

### High — Court points are invalid on zoomed and side-angle shots

Court points align reasonably on the wide Shot 18 view. In Shots 216, 270, 373,
and 472, points are scattered, clipped, or do not match visible court
intersections. The tracker calculates one court grid from the first detectable
frame and reuses it for the whole shot, so camera movement can also make the grid
stale. Court filtering and foot-point geometry based on these points cannot be
trusted on those shots.

### High — The annotated video has no audio

The source contains H.264 video and AAC audio. The output contains only an MPEG-4
video stream. Rendering through OpenCV preserved duration, frame count,
resolution, and frame rate, but discarded the match audio.

### Medium — Player detection is incomplete

Only 35.33% of annotated frames contain four tracked detections. The remaining
frames contain one, two, or three detections, so identity matching cannot recover
all four players even with perfect appearance descriptors. Invalid camera-view
selection contributes to this number.

### Medium — Labels overlap and hide one another

Labels are placed directly above each bounding box without collision handling.
They overlap heavily when far-court players stand close together, notably in
Shots 75, 120, and 168. Some text becomes partially unreadable.

### Medium — Output encoding is unnecessarily large

The source is 331 MiB; the annotated output is 2.2 GiB (about 6.7 times larger).
The renderer changes H.264 to OpenCV's `mp4v` MPEG-4 codec, producing a much less
space-efficient deliverable.

## Checks that passed

- Full-video FFmpeg decode completed with no errors.
- Output dimensions, rate, duration, and frame count match the source:
  1280×720, 30 FPS, 3,250.03 seconds, 97,501 frames.
- Segmentation masks and contours are visible and generally follow the player
  silhouettes in the sampled frames.
- Bounding boxes in JSON are finite, valid, and inside the video dimensions.
- There are no duplicate frame/track records and no local track changes its
  saved logical identity within a shot.
- No frame assigns the same resolved logical ID to two tracks; raw ByteTrack IDs
  remain visible for debugging.

## Audit limitation

There is no manually labelled ground-truth identity timeline, so this audit
cannot calculate identity accuracy or prove that every displayed `P1`–`P4` is
the correct physical player. It can establish coverage, consistency, media
integrity, and visible overlay defects. A small manually labelled validation set
is required before accepting cross-shot identity correctness.

## Recommended correction order

1. Reject non-wide and unstable court views before tracking or gallery updates.
2. Validate/recalculate court geometry instead of reusing an invalid static grid.
3. Improve four-player detection and segmentation coverage.
4. Measure ReID against manually labelled shots, then tune or replace the
   appearance matcher.
5. Add label collision handling.
6. Mux source audio into an H.264 annotated output and control bitrate/size.

## Improvement status after this snapshot

The audited MP4 itself remains unchanged. The next-run pipeline now:

- verifies three frames per candidate and rejects incomplete, zoomed, side-angle,
  or unstable court views before tracking and gallery updates;
- places labels in non-overlapping slots;
- encodes the final video as H.264 at a controlled bitrate and copies source
  audio with FFmpeg.

A preflight check kept 70 of 245 coarse candidates. It retained representative
wide views including Shots 18, 23, 75, 120, and 168, while rejecting the audited
bad views 216, 270, 373, and 472. ReID coverage must be measured again after a
complete rerun; no identity-accuracy claim is made from this filtering change.
