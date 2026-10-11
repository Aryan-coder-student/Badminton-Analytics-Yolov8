# Track Association Experiment Plan

## Goal

Report whether a later local ByteTrack ID is a plausible continuation of an
earlier ID inside one continuous `MAIN_COURT` shot. This is a diagnostic
experiment only: it must never merge, rename, or otherwise mutate ByteTrack
IDs. Team labels, persistent identities, camera-cut re-identification, and
automatic recovery remain out of scope.

## Inputs and boundaries

1. Regenerate `shot_0018_descriptors.npz` with
   `scratchpad.test_track_appearance` before the experiment. This ensures its
   HSV vectors use the current full-mask feature extraction.
2. Replay `track_main_court_shot` once to collect each raw track's first and
   last frame and foot-point. This reuses the existing court filter, exact
   `TrackedSegmentation` pairing, and `Tracker("bytetrack")`; it does not run
   DINOv2 again.
3. Load the final DINOv2 and HSV vectors from the descriptor archive. A clear
   error will be raised if archive IDs or descriptor rows do not match the
   replayed local tracks. Archive first/last frames are sampled-observation
   metadata, so raw replayed endpoints remain the source of temporal checks.

## Proposed reusable API

Add `src/badminton_analysis/appearance/association.py` with small immutable
data objects:

- `TrackAssociationProfile`: one local `track_id`, its first/last frame,
  first/last projected court positions, and final DINOv2 and HSV descriptors.
- `AssociationCandidate`: predecessor and successor IDs, both similarities,
  elapsed frame gap, court distance in metres, temporal validity, and the
  diagnostic `likely_association` result.
- `AssociationPolicy`: explicit initial limits—DINOv2 >= 0.85, HSV >= 0.90,
  elapsed gap <= 60 frames, and court movement <= 4.0 m.

Pure functions will calculate cosine similarity, elapsed gap, court distance,
and candidates. `elapsed_gap` is defined as `successor.first_frame -
predecessor.last_frame`; consecutive frames therefore have a gap of one.
Candidate ranking will prefer likely candidates, then higher mean visual
similarity, shorter court distance, and shorter gap. It is a ranking signal,
not an identity decision.

## Court positions

The scratchpad layer will convert the existing bottom-centre foot point from
image pixels to court metres with an inverse homography from the detected
`court_grid` to `create_canonical_grid()`. The reusable association module
accepts only numeric court positions, so it stays independent of OpenCV,
YOLO, video files, and scratchpad code.

## Script and output

Add `scratchpad/test_track_association.py`. It will build profiles, evaluate
every ordered pair, print overlap/reverse-time pairs as rejected, and print
metrics for every temporally valid predecessor-successor pair. It will also
print the best valid predecessor for each later track. Raw tracks 2 and 5
remain separate throughout; the script only reports the candidate `2 -> 5`.

The current archive provides a useful expected smoke signal: `2 -> 5` has
DINOv2 similarity `0.8888` and HSV similarity `0.9907` before adding temporal
and court-continuity checks.

## Tests and validation

Add deterministic tests in `tests/test_track_association.py` for overlap
rejection, gap definition, cosine similarity, Euclidean court distance, and
evaluation of tracks 2 and 5 without any merge. Tests will use fixed NumPy
vectors and positions—no model, video, or downloaded assets. Validation after
implementation will run:

```bash
uv run pytest -q
uv run python -m scratchpad.test_track_appearance
uv run python -m scratchpad.test_track_association
```

The final report will include changed files, test results, every candidate
score, the best predecessor for Track 5, descriptor dimensions, and any
sample rejected by the explicit policy.
