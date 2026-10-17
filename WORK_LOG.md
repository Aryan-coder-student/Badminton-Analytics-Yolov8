# Project Work Log

## 2026-10-17 — Repository hygiene and dependency lock sync

- Excluded the local BFMD dataset, model directory, and macOS metadata from Git.
- Synchronized `uv.lock` with the existing `pycocotools` project dependency.
- Kept datasets and model checkpoints local while preparing the branch for GitHub.

## 2026-10-11 — BFMD evaluation CI and CVAT player-mask prelabels

- Added GitHub Actions CI for the model-free frame evaluator, shot clustering,
  BFMD integration and camera-view metadata, with light dependencies and
  Python compilation checks. Initial workflow executions passed.
- Added `evaluation/cvat.py` and `evaluation.cli export-cvat` to bundle
  selected BFMD PNG frames, YOLO person boxes and mask-derived COCO polygons
  into a CVAT-ready ZIP. All outputs are marked **unverified prelabels**.
- Added deterministic mask polygon/ZIP export tests, integrated them into
  the CI workflow, and documented the two commands following BFMD frame
  selection. Manual correction is still required before detection benchmarking;
  outer contours can omit interior mask holes, and raw YOLO includes spectators.

## 2026-10-11 — BFMD full-match adapter for Milestone 1

- Added `scripts/prepare_bfmd.py` and `evaluation/bfmd.py` to use official
  BFMD single/doubles match paths while keeping copyrighted BWF TV videos
  and downloaded dataset annotations outside Git.
- The runner can detect scene cuts through `transnetv2-pytorch==1.0.5`
  using provided pretrained weights, or reuse the existing TransNetV2
  shot-range CSV and cached DINOv2 embeddings. It runs the already-existing
  shot-level KMeans workflow and exports grouped contact sheets.
- Added candidate selection distributed across clusters and the match,
  prioritizing distinct camera shots; a shot-disjoint holdout split prevents
  adjacent frames from the same shot leaking into both splits. Select
  reviewed camera views with optional repeated `--camera-view` flags.
- Frames are exported losslessly as PNG. BFMD court-per-frame hints create
  optional **ROI previews** only, never cropped input to view embeddings.
  Optional BFMD singles pose boxes are copied only as unverified prelabels;
  doubles frames have no assumed player ground truth. No mask accuracy
  or actual camera-view accuracy has been claimed.
- Added synthetic-input and dataset-helper test cases and a BFMD runbook.
  Real BFMD video/weights, actual full-match inference and the combined
  test suite remain unverified in this environment.

## 2026-10-11 — Reuse shot-cut extraction and visual clustering for camera-view prelabels

- Confirmed the existing camera-shot pipeline: TransNetV2 boundaries, 25/50/75%
  frame samples, one DINOv2 embedding per frame, per-shot averaging, and KMeans.
  The recorded 1,431-frame experiment found six unsupervised visual clusters
  (silhouette 0.207529); these are **not** verified semantic camera labels.
- Added a tracked, reusable shot-range -> three-percentile sampling -> DINOv2
  embedding cache -> existing Cluster("kmeans") -> shot-cluster CSV path.
  The pipeline accepts exported shot boundaries rather than duplicating the
  existing TransNetV2 boundary detector in Git-ignored scratchpad code.
- Integrated human-reviewed cluster-to-view mapping with optional per-shot
  overrides; all player boxes/masks and their frame indices remain unchanged.
- Added focused tests for sampling, cache round trips, boundary validation,
  cluster metadata, and shot overrides. A synthetic video smoke test passed
  locally for the shot-embedding/cache component. End-to-end model metrics
  remain unmeasured until source video and labels are available.

## 2026-10-11 — Milestone 1 player detection/segmentation benchmark groundwork

- Added a model-independent benchmark with one-to-one IoU matching and
  precision/recall/F1, all-player frame coverage, four-player coverage, and
  mask IoU. No identity or track association changes were made.
- Added a CLI for raw YOLO person segmentation on selected labelled video
  frames and full-size mask/JSON export. Court filtering remains in ignored
  scratchpad code; raw YOLO scores are not equivalent to court-filtered scores.
- Documented manual annotation selection, output format, and evaluation
  commands. No fake ground-truth accuracy measurements were produced.
- Validation: nine new isolated unit tests passed in a local test environment.
  No full-match model inference or full repository test suite was run here;
  match video and model checkpoints were unavailable.

This file is the dated record of material project progress. Add new entries directly
below this note in reverse chronological order. Record outcomes, verification,
important decisions, and unfinished work rather than individual shell commands.

## 2026-09-19 — Low-VRAM segmentation path

- Diagnosed a CUDA OOM inside Ultralytics native mask upscaling during the
  match experiment. The court pipeline batched 16 high-resolution frames while
  DINOv2 was also loaded on a 3.68 GiB GPU.
- Removed the 16-frame batch, segmenting one frame at a time. The player
  segmenter now requests only class 0 (`person`) before YOLO builds masks,
  while the generic YOLO adapter remains configurable for other classes.
- Added `BADMINTON_SEGMENTATION_DEVICE=cpu` as a settings-backed fallback.
  It keeps high-resolution player masks and can be used for both match
  inference and rendering, at the cost of slower segmentation.
- CPU-smoked one real Shot 18 frame: four tracked players with full 720×1280
  masks. A one-shot end-to-end run with CPU segmentation completed Shot 18,
  saved 579 per-frame records to `/tmp/badminton-reid-cpu-smoke`, and mapped
  local tracks 2 and 5 to P2. CUDA could not be tested here because the NVIDIA
  driver is unavailable in this environment. All 75 tests and Ruff checks
  pass. The previous full-match MP4 and JSON were not regenerated.

## 2026-09-19 — Conservative recovery of fragmented local tracks

- Added one-pass first/last court-position collection to the experimental match
  pipeline. The existing within-shot association policy now identifies likely
  non-overlapping fragments; an accepted cross-shot P-ID propagates to the
  fragment only if the local group has no conflicting accepted P-ID. Raw
  ByteTrack IDs, existing cross-shot gates, and persistent gallery updates are
  unchanged.
- Added read-only replay `scratchpad/audit_reid_grouping.py`. On the saved
  70-shot run it predicts 1,447 additional resolved observations: 26,111 ->
  27,558 of 71,134, reducing unresolved from 63.29% to 61.26%. No previously
  accepted labels changed or became unresolved. This is a coverage estimate,
  not ground-truth identity accuracy; saved JSON/MP4 were not regenerated.
- Rejected pre-assignment gallery grouping after replay showed that it would
  lose 6,044 previously accepted observations. Updated the audit and module
  documentation with remaining causes and research directions.
- Added propagation and endpoint tests; the full test suite passes (73 tests).

## 2026-09-19 — Clean match ReID rerun diagnosed

- Confirmed the clean rerun completed on 2026-09-13: 70 court shots, 71,134
  player observations, a new 801 MB H.264/AAC MP4, per-frame JSON, and 70
  descriptor archives. Full MP4 decode passed.
- Confirmed the user's visible issue: 45,023 observations (63.29%) are
  unresolved; only 105 of 13,507 four-detection frames have all four IDs.
- Replayed saved galleries and reproduced every JSON assignment. The main
  causes are 137 margin-only rejections (28,267 frame observations) and 360
  additional local tracks without a four-way assignment slot (16,005 frame
  observations). Fragmentation is likely but not verified for every track.
- Recorded a separate coverage regression: the strict view gate accepted no
  shots after Shot 313 (~41:10), leaving the last ~13 minutes without overlays.
  The final MP4 also has one fewer video frame than the source.
- Wrote `docs/audits/match-reid-rerun-2026-09-19.md`. No matcher thresholds,
  identity files, or video outputs were modified during diagnosis.

## 2026-09-13 — Match ReID output-quality improvements

- Replaced the single-midpoint court check with three-frame verification at 25%,
  50%, and 75% of each candidate. A shot now requires a complete in-frame court,
  plausible wide-view scale and perspective, and less than 5% corner drift before
  tracking or persistent-gallery updates. The shot tracker also refuses partial
  grids when selecting the geometry used for court filtering.
- A full candidate preflight retained 70 of 245 coarse candidates. Wide examples
  18, 23, 75, 120, and 168 remain accepted; audited zoom/side views 216, 270,
  373, and 472 are rejected.
- Added collision-aware label placement shared by the match and single-shot
  identity visualizations. A Shot 75 replay confirmed that four centrally
  stacked labels render in separate readable slots.
- The final renderer now uses FFmpeg to create controlled-bitrate H.264 and copy
  optional source audio. A real 5.27-second smoke output contained both H.264
  video and AAC audio.
- Added court-completeness, camera-drift, label-layout, and final-encoding tests.
  Verification: 69 tests pass. The existing full-match MP4 was not regenerated;
  run the complete match experiment to apply the new shot selection and ReID.

## 2026-09-13 — Full-match ReID output audit

- Audited the generated full-match MP4 and `frame_identities.json`; recorded the
  results in `docs/audits/match-reid-output-2026-09-13.md` without changing the
  inference or rendering pipeline.
- The video fully decodes and matches the source's 97,501 frames, 1280×720
  resolution, 30 FPS, and 3,250.03-second duration. Saved boxes and identity keys
  are structurally consistent, and sampled frames contain masks and court points.
- Recorded blocking quality gaps: 61.50% of observations are unresolved, only
  105 frames contain the complete `P1/P2/P3/P4` set, non-wide views are admitted
  as `MAIN_COURT`, and court points become invalid on those views.
- Recorded delivery defects: source audio is absent, overlapping labels are hard
  to read, and `mp4v` encoding expands the 331 MiB source to a 2.2 GiB output.

## 2026-09-13 — Match video segmentation and court overlays

- Added `scratchpad/render_match_reid.py`, a render-only command that reuses
  the completed `frame_identities.json`. It replays verified MAIN_COURT shots
  to draw court keypoints, player segmentation masks/contours, bounding boxes,
  logical P1–P4 labels, and raw ByteTrack IDs without rerunning DINOv2 ReID.
- Updated the full match experiment to use the same renderer. Rendering writes
  to a temporary MP4 and replaces the previous output only after success, so a
  stopped long render preserves the existing video.
- Added tests for JSON round-tripping, shot-scoped identity lookup, and visible
  court/player overlays. Verification: 62 tests pass.

## 2026-09-12 — Experimental match-level persistent ReID

- Added reusable cross-shot gallery models and pure scoring/assignment helpers
  under `src/badminton_analysis/appearance/`. A track gallery retains sampled
  normalized DINOv2 and HSV observations; a bounded persistent player gallery
  accumulates only accepted local tracks. Hungarian assignment enforces one
  target-track-to-player proposal per shot, while configurable absolute-score
  and positive-margin gates keep weak or ambiguous matches unresolved.
- Added `scratchpad/test_cross_shot_reid.py` for configurable two-shot
  diagnostics and `scratchpad/test_match_reid.py` for chronological
  MAIN_COURT-shot processing. The latter resets ByteTrack per shot, retains
  match-level P1/P2/P3/P4 galleries, writes `frame_identities.json`, and
  renders an MP4 with both logical and raw track IDs.
- Verified the initial camera-cut experiment with shot 18 -> shot 23. It
  accepts track 1 -> P1 (0.884, margin 0.156) and track 4 -> P2 (0.866,
  margin 0.192); tracks 2 and 3 are deliberately unresolved because their
  margins are 0.020 and -0.034.
- Reused court and segmentation models across shots while keeping a fresh
  ByteTrack instance for every cut. Segmentation processes ordered bounded
  frame batches before the sequential tracker update. Court tracking now finds
  the first valid court-grid frame when a shot begins on a transition; shot 47
  verifies this behavior. Verification: 59 tests pass.
- Added `docs/modules/cross-shot-reid.md` in plain language. It explains the
  gallery data objects, each reusable scoring/assignment helper, scratchpad
  tools, outputs, two-shot result, and the current limitation that a completed
  full-match MP4 has not yet been generated.
- Fixed empty appearance handling in `test_match_reid.py`. A court-view shot
  with no valid player masks now prints a skip message before archive writing,
  rather than passing an empty list to `numpy.stack()`. The standalone archive
  writer also rejects direct empty calls with a clear error.

## 2026-09-12 — Anonymous identities within one MAIN_COURT shot

- Added `PlayerIdentity` and `resolve_player_identities()`. The resolver maps
  raw local tracks to deterministic `P1`, `P2`, … identities using only likely,
  time-valid, non-overlapping association candidates. It supports association
  chains and refuses any component merge that would group overlapping tracks.
- Added `scratchpad/test_player_identities.py`, which first resolves identities
  and then renders stable labels with raw IDs retained for debugging. Shot 18
  maps `1 -> P1`, `2 -> P2`, `5 -> P2`, `3 -> P3`, and `4 -> P4` in
  `data/player_predictions/player_identities/sample2_shot_0018_player_identities.mp4`.
- Added resolver coverage for accepted, rejected, overlapping, unrelated, and
  chained tracks; deterministic numbering; and raw-ID preservation.
  Verification: 48 tests pass.

## 2026-09-12 — Association module cleanup

- Moved `AssociationPolicy`, `TrackAssociationProfile`, and
  `AssociationCandidate` to `src/badminton_analysis/appearance/models.py`.
- Moved cosine similarity, frame-gap, court-distance, and temporal-overlap
  calculations to `src/badminton_analysis/appearance/utils.py`. The association
  module now only evaluates and ranks candidates from those models and helpers.
- Verification: 42 tests pass.

## 2026-09-12 — Local track-association experiment implemented

- Added pure, model-free association scoring in
  `src/badminton_analysis/appearance/association.py`. `TrackAssociationProfile`
  keeps each existing local ID separate, while `AssociationCandidate` reports
  DINOv2/HSV cosine similarity, elapsed frames, temporal overlap/validity, and
  court-plane distance. No function returns a replacement ID or updates the
  tracker.
- Added image-to-court homography helpers to `scratchpad/court_players.py` and
  `scratchpad/test_track_association.py`. The script replays the existing
  court-filtered ByteTrack path for raw first/last endpoints and reads final
  descriptors from the saved archive; it does not run DINOv2 again.
- On the sample shot, local Track 2 -> Track 5 is the only time-valid ordered
  pair: DINOv2 `0.8888`, HSV `0.9907`, 25 elapsed frames, and `0.26 m` between
  endpoint foot positions. It is reported as likely, while IDs 2 and 5 remain
  independent. All other ordered pairs are rejected for overlap or reverse
  time.
- Added deterministic association tests for overlap rejection, frame gaps,
  cosine similarity, court distance, policy rejection, and Track 2 -> Track 5
  evaluation without identity merging. Verification: 42 tests pass.
- The descriptor archive used for the smoke run predates the full-mask HSV
  change. Its offline regeneration was stopped because CPU-only DINOv2
  extraction was still running after several minutes. Regenerate it with
  `uv run python -m scratchpad.test_track_appearance` before using the printed
  HSV score for a downstream decision.

## 2026-09-12 — Local track-association experiment planned

- Added `TRACK_ASSOCIATION_PLAN.md` before implementation. The experiment will
  retain local ByteTrack IDs and only report potential predecessor-successor
  links from existing final DINOv2/HSV descriptors, temporal order, and
  foot-point positions projected to court metres.
- The planned reusable association calculations are pure and model-free under
  `src/badminton_analysis/appearance/`; the scratchpad script will own video
  replay, descriptor-archive loading, court projection, and printed output.
- Initial diagnostic policy is explicit and conservative: DINOv2 >= 0.85, HSV
  >= 0.90, at most 60 elapsed frames, and at most 4.0 m of court movement.
  It will rank candidates but will not merge Track 2 with Track 5.

## 2026-09-11 — Full-mask HSV descriptor

- Removed the manual `TORSO_HEIGHT_RATIO` cutoff. HSV extraction now uses every
  pixel selected by the player segmentation mask, including valid lower-body
  pixels, while continuing to exclude the background.

## 2026-09-11 — Appearance model cleanup

- Kept `AppearanceInput` separate from feature-processing code in
  `src/badminton_analysis/appearance/models.py`.
- Moved the complete appearance package out of `model/` because it owns domain
  feature preparation and track aggregation rather than a DL model adapter.
- Removed `AppearanceQuality`, `minimum_crop_size`, and `minimum_mask_pixels` from
  the API. Appearance extraction now rejects only structurally invalid inputs:
  mismatched mask dimensions, invalid or empty boxes, and empty masks.

## 2026-09-10 — Offline track appearance descriptors

- Added reusable appearance extraction under
  `src/badminton_analysis/appearance/`. Each sampled
  `AppearanceObservation` retains its source frame and exact
  `TrackedSegmentation`, normalized DINOv2 embedding, mask-aware HSV descriptor,
  and representative masked RGB crop.
- Added `TrackAppearanceCollector` with per-track sampling (default every five
  frames), structural input checks, and normalized mean aggregation.
  Local ByteTrack IDs remain independent; no identity matching or team labels are
  performed.
- Refactored the original masked crop and upper-body HSV histogram into one
  production feature path shared by the appearance collector and the existing
  one-image similarity experiment.
- Added `scratchpad/tracked_shot.py` as the shared MAIN_COURT iterator used by both
  tracking and appearance scripts. `TrackedSegmentation` preserves the exact
  track ID, detection, and mask association without tuple-position conventions.
- Added `scratchpad/test_track_appearance.py`. Command:
  `uv run python -m scratchpad.test_track_appearance`.
- Saved final descriptors to
  `data/player_predictions/track_appearance/shot_0018_descriptors.npz` and up to
  four representative masked crops per track under `shot_0018_crops/`.
- Sample result with interval five: track 1 = 30 observations, track 2 = 3,
  track 3 = 32, track 4 = 32, and track 5 = 22. Every DINOv2 descriptor has 768
  dimensions and every HSV descriptor has 96 dimensions. IDs 2 and 5 remain
  separate as required.
- Verification: 36 tests pass. Saved DINOv2 and HSV aggregate norms are 1.0 within
  float32 precision.

## 2026-09-10 — ByteTrack player IDs

- Implemented stateful tracking contracts, an Ultralytics `ByteTrack` adapter,
  and the client-facing `Tracker("bytetrack")` strategy façade.
- Added and locked `lap>=0.5.12`, loading Ultralytics' bundled
  `bytetrack.yaml` through Pydantic settings.
- Added `scratchpad/test_player_tracking.py`. The default run processes shot 18
  (`170.5`–`175.733` seconds), segments people at confidence `0.1`, filters to an
  expanded court playing area, keeps the four strongest player masks, tracks
  their detections, and renders `Player {track_id}` using an ID-stable mask color.
- Moved shared court polygon, keypoint, foot-point, and player-filtering logic to
  `scratchpad/court_players.py`; court points remain visible without generated
  court lines.
- Generated a 1280x720, 30 FPS, 158-frame result at
  `data/player_predictions/bytetrack/sample2_shot_0018_tracked.mp4` and a preview
  image beside it. Command: `uv run python -m scratchpad.test_player_tracking`.
- Verification: 31 tests pass, including adapter conversion, stateful updates,
  ROI filtering, exact mask pairing, façade delegation, and track-ID rendering.
- Observed IDs were `[1, 2, 3, 4, 5]`: the near-left player changed from ID 2 to
  ID 5 during fast movement. Appearance-based re-identification remains the next
  step for repairing this kind of motion-only ID switch.

## 2026-09-10 — Player segmentation and appearance grouping

- Added generic segmentation contracts under
  `src/badminton_analysis/model/segmentation/`. A `Segmentation` pairs the existing
  `Detection` metadata with a full-resolution boolean mask.
- Added `YOLOSegmenter` using `yolo11n-seg.pt` and the client façade
  `Segmenter("yolo")` in `model/segmentations.py`.
- Added Pydantic settings and environment examples for segmentation model and
  confidence selection.
- Added `scratchpad/test_players_in_court.py`. It uses detected court geometry and
  each person's bottom-centre foot point to exclude officials and spectators. The
  sample frame contains four accepted court players.
- Added `scratchpad/test_player_similarity_segmentation.py`. For one image, it
  creates masked player crops, combines normalized DINOv2 features with a
  mask-aware HSV uniform descriptor, and uses KMeans with two clusters. Players
  `P1/P2` share Group 1 and `P3/P4` share Group 2, with matching mask colours.
- Preserved court keypoints in the visualization without drawing generated court
  lines. Output: `data/player_predictions/test_player_similarity/shot_0018_75.jpg`.
- Installed and validated the global Codex skill `thermonuclear-implementation`.
- Added `.gitignore` rules for datasets, model checkpoints, runs, caches, and local
  environments. Verification: 25 tests pass.

Decisions and follow-ups:

- `P1`–`P4` are deterministic per-image labels, not persistent YOLO track IDs.
- Re-identification and multi-frame tracking are intentionally deferred.
- The two-cluster appearance grouping is a one-image experiment, not an identity
  model. A future re-identification model should replace this heuristic.

## 2026-09-04 — Generic object detection strategy

- Replaced the person-specific detector with generic `YOLODetector`; results now
  retain each YOLO class ID and class name without hard-coded `"person"` filtering.
- Added the tag-based `Detector("yolo")` façade in `model/detections.py`, matching
  the clustering and embedding client pattern.
- Centralized YOLO checkpoint and confidence defaults in Pydantic settings.

## 2026-09-03 — Embedding and clustering pipeline

- Added image embedding strategies for DINOv2, CLIP, and ResNet with shared image
  loading and device selection.
- Added KMeans, DBSCAN, and HDBSCAN clustering strategies plus tag-based
  `Embedding` and `Cluster` façades.
- Added `scratchpad/test_clustering_approach.py` for 1,431 extracted shot frames.
  It caches DINOv2 embeddings, averages three frames per shot, compares KMeans
  candidates, and materializes assignments. The recorded run selected six visual
  classes (`silhouette=0.207529`).
- Added `transnetv2-pytorch` to the uv-managed project dependencies and created the
  repository contributor guide in `AGENTS.md`.

## 2026-08-28 — Court reconstruction prototype

- Added `scratchpad/test_court.py` using a YOLO pose checkpoint with 30 court
  keypoints.
- Reconstructed a regulation badminton grid through homography and rendered a
  custom overlay without the model's default item box. Documented the geometry and
  commands in `COURT_VISUALIZATION.md`.

## 2026-08-22 — Shot-frame extraction

- Added `scratchpad/get_image.py` to read `data/sample2_scenes.csv` and extract
  frames at 25%, 50%, and 75% of every shot into `data/shot_frames/`.
- The current generated dataset contains 1,431 shot-frame images.
