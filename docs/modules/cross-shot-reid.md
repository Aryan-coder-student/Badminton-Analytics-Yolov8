# Module — Experimental Cross-Shot Player ReID

## What problem it solves

ByteTrack works only inside one camera shot. After a cut, it starts again at
Track 1, Track 2, and so on. This module adds a separate experimental layer
that can label those new local tracks as the same anonymous match player:
`P1`, `P2`, `P3`, or `P4`.

It never changes a raw ByteTrack ID. A frame can therefore show `P2 (track 4)`.

## Main data objects

- `TrackAppearanceGallery`: all sampled DINOv2 and HSV features for one local
  track in one shot, for example `(shot 23, track 4)`.
- `PersistentPlayerGallery`: the accepted appearance samples accumulated for
  one logical player, for example `P2`, across earlier shots.
- `CrossShotSimilarity`: the complete score table: every new track compared
  with every existing player gallery.
- `CrossShotAssignment`: one proposed `track -> player` result, including its
  DINOv2 score, HSV score, combined score, second-best player, and margin.

## Reusable helper functions

`gallery.py` owns the matching maths:

- `pairwise_cosine_similarity()` compares every sample from two galleries.
- `top_k_mean()` keeps the strongest few comparisons, so one bad/blurred frame
  does not dominate the result.
- `score_gallery_pair()` combines DINOv2 and HSV scores. The default weights
  are 80% DINOv2 and 20% HSV.
- `build_cross_shot_similarity()` creates the full new-track × player matrix.
- `assign_cross_shot_players()` uses Hungarian assignment. This means one new
  track cannot be assigned to the same player as another new track in a shot.
- `update_persistent_galleries()` adds samples only from confident proposals.
  It requires both a minimum total score and a margin above the next-best
  player. A rejected track stays `unresolved` unless a later within-shot
  propagation step links it to one accepted track.
- `propagate_local_player_ids()` carries an accepted P-ID to a linked local
  ByteTrack fragment after cross-shot matching. It uses the existing
  within-shot DINOv2/HSV, frame-gap, and court-position policy. It never
  changes an accepted ID and refuses to propagate across a group containing
  conflicting accepted IDs. Propagated fragments do not update the persistent
  appearance gallery.

For the full-match script, the current default minimum combined score is `0.70`
and the required margin is greater than `0.05`. Both are command-line options,
not permanent claims about the correct ReID threshold.

## Scratchpad tools

### `test_track_appearance.py`

Runs segmentation, court filtering, ByteTrack, masked DINOv2, and HSV for one
shot. It saves a compact `.npz` gallery archive under
`data/player_predictions/track_appearance/`.

### `test_cross_shot_reid.py`

Compares two saved shot archives. It prints DINOv2, HSV, and combined score
tables plus proposed matches. It is diagnostic only: it does not update a
player gallery or write a video.

### `test_match_reid.py`

Processes candidate MAIN_COURT shots in time order. Before tracking, it checks
the 25%, 50%, and 75% frames of every candidate. All three must contain a
complete wide-court grid, remain inside the image, have normal broadcast-view
scale and perspective, and remain stable within 5% of the frame diagonal. This
prevents close-ups, side angles, and camera motion from updating player
galleries. Shot tracking also waits for a complete grid instead of accepting
the first partial court detection. It then shares the court and segmentation
models for speed, creates a fresh ByteTrack instance for every shot, keeps
P1–P4 galleries across cuts, and saves:

- `data/player_predictions/match_reid/frame_identities.json`
- `data/player_predictions/match_reid/sample2_match_persistent_identities.mp4`

Each shot now records court-plane track endpoints during the same tracking
pass used for appearance extraction; it does not replay that shot to discover
local fragments. The current saved JSON and MP4 predate local-fragment
propagation. Run the full command again to apply that improvement to output.

### `audit_reid_grouping.py`

Replays the saved 70-shot descriptors and identity records without replacing
either output file. It reports how many unresolved observations the conservative
local-fragment rule could recover and checks whether prior accepted labels
would be changed or lost. The latest replay predicts +1,447 labelled
observations, but does not measure identity correctness.

### `render_match_reid.py`

Regenerates only the final visualization from an existing
`frame_identities.json`. It replays court detection, player segmentation, and
ByteTrack for the saved MAIN_COURT shots, then draws:

- court keypoints without connecting court lines
- a colored segmentation mask and outline on each player
- the player bounding box
- `P# (track #)` or `unresolved (track #)`

It does not rerun DINOv2 or cross-shot identity matching. Labels are assigned
non-overlapping vertical slots when players stand close together. The renderer
first creates the annotated frames, then uses FFmpeg to encode H.264 at a
controlled bitrate and copy the source audio. The old MP4 is kept until the
replacement render completes successfully.

Final MP4 encoding requires an `ffmpeg` executable with the `libopenh264`
encoder available on `PATH`.

## Commands

Compare the already saved sample Shot 18 and Shot 23 galleries:

```bash
uv run python -m scratchpad.test_cross_shot_reid
```

Measure local-fragment recovery from the existing saved match without
changing its JSON or MP4:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
UV_CACHE_DIR=/tmp/badminton-uv-cache \
uv run python -m scratchpad.audit_reid_grouping
```

Run the complete experimental match process and render the final video:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
UV_CACHE_DIR=/tmp/badminton-uv-cache \
uv run python -m scratchpad.test_match_reid
```

On a roughly 4 GiB GPU, YOLO's full-resolution mask upscaling may run out of
VRAM while DINOv2 is also loaded. The shot pipeline now segments one frame at
a time and filters to the COCO person class before mask construction. If VRAM
is still insufficient, run segmentation on CPU while leaving DINOv2's device
selection unchanged:

```bash
BADMINTON_SEGMENTATION_DEVICE=cpu HF_HUB_OFFLINE=1 \
TRANSFORMERS_OFFLINE=1 UV_CACHE_DIR=/tmp/badminton-uv-cache \
uv run python -m scratchpad.test_match_reid \
  --output-dir data/player_predictions/match_reid_improved
```

CPU segmentation is slower. The previous `frame_identities.json` and MP4 in
`match_reid/` remain untouched by this separate output directory. An interrupted
run is not resumable; rerun the full command to generate a complete JSON/MP4.

If `frame_identities.json` already exists, regenerate only the corrected video:

```bash
uv run python -m scratchpad.render_match_reid
```

The render-only command improves labels, encoding, and audio, but it keeps the
shot selection and identities already stored in the JSON. Run the complete
match command to apply the stricter court-shot gate and regenerate identities.

## Verified two-shot result

For Shot 18 → Shot 23, the experiment accepted Track 1 → P1 and Track 4 → P2.
The other two tracks were left unresolved because their evidence was too close
to another possible player. This is deliberate: guessing would contaminate
later player galleries.

## Limits

This is not player recognition. It does not know real names, teams, A1/A2/B1/B2,
or identities across a different camera setup. A dedicated ReID model is the
next stage for improving unresolved players.
