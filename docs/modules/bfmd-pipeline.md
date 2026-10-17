# Run the BFMD match preparation pipeline

## Inputs and source rights

BFMD is a CVPR 2026 workshop full-match badminton dataset. Download its
**annotation package** from the [official BFMD repository](https://github.com/Ning-D/BFMD).
The published data package has 12 singles and 7 doubles broadcast matches.
It **does not redistribute the actual videos** for copyright reasons. Follow
the provider's `BFMD_data/download_youtube.py --list` and `--match` instructions
to obtain the relevant official BWF TV source locally, subject to rights and
platform terms. Never commit or redistribute those recordings.

The root is expected to look like this:

```text
BFMD_data/
  download_youtube.py
  videos/<tournament>/<match>.mp4       # singles
  videos_doubles/<match>.mp4             # doubles
  annotations/
    court_perframe/<match>.json          # optional 1-Hz court hints
    pose/<match>.json                    # optional singles pose-box prelabels
```

**Important**: BFMD documents video frame coordinates as **zero-based** and
annotations aligned to **1280x720 at 30 fps**. Do not silently resize,
trim, change FPS, or remux with frame dropping. This script refuses mismatched
video dimensions/timing by default. Override only for exploratory clustering,
with BFMD frame/box/court annotations disabled.

## Main pipeline

```text
Official BFMD match MP4
   |
   +--> TransNetV2 shot-boundary detector --> camera-shot [start,end) frames
   |                                             |
   |                       sample at 25%, 50%, 75% of each shot
   |                                             |
   |                        existing DINOv2 image embedding
   |                                             |
   |                        mean embedding for each camera shot
   |                                             |
   |                        existing KMeans visual clustering
   |                                             |
   |                   review cluster contact sheets and map views
   |                                             |
   +--> sampled original full-size frames for box/mask annotation
   |
   +--> optional BFMD court_perframe hints --> court ROI *previews only*
   |
   +--> optional BFMD singles pose boxes --> unverified prelabels only
```

**Do not crop to the court before DINOv2 clustering.** The goal of these
embeddings is to separate *camera views*, for which scoreboard, zoom, player,
and wider scene context are useful. The court ROI is used as a later
*preview/diagnostic* for court-player filtering; it is never substituted for the
original 1280x720 frame in the benchmark.

## Run

From the repository root:

```bash
uv sync
python BFMD_data/download_youtube.py --list
python BFMD_data/download_youtube.py --match "<unique-match-fragment>"

uv run python scripts/prepare_bfmd.py \
  --bfmd-root BFMD_data \
  --match "<unique-match-fragment>" \
  --transnet-weights /path/to/transnetv2-pytorch-weights.pth \
  --clusters 6 \
  --frames 150
```

The weights are the appropriate pretrained TransNetV2 PyTorch state dict;
do not supply the DINOv2 or YOLO checkpoint. The `transnetv2-pytorch==1.0.5`
package's documented `detect_scenes` method is used with `threshold=0.5`
(default). It returns inclusive end indices; we explicitly normalize to
`[start_frame,end_frame)` for this repository.

**Already have shot boundaries?** Reuse them instead of rerunning TransNetV2:

```bash
uv run python scripts/prepare_bfmd.py \
  --bfmd-root BFMD_data \
  --match "<unique-match-fragment>" \
  --shot-ranges data/benchmark/my_shot_ranges.csv \
  --frames 150
```

The CSV must have `shot_id,start_frame,end_frame` with **end exclusive**
and absolute video frame indices.

After reviewing `cluster_review/*.jpg`, map cluster IDs to semantic views in
`cluster_camera_views.json`. To prepare just the **wide-court region/view**
for player segmentation, rerun in a **separate output directory**
(preventing an older mixed-view review sheet from being mistaken for current):

```bash
uv run python scripts/prepare_bfmd.py \
  --bfmd-root BFMD_data \
  --match "<unique-match-fragment>" \
  --shot-ranges data/benchmark/bfmd/<match>/shot_ranges.csv \
  --camera-view wide \
  --frames 150 \
  --output data/benchmark/bfmd/<match>_wide
```

Copy the already-reviewed `cluster_camera_views.json` to that output
directory first. To reuse embeddings without recomputation, also copy the
validated `shot_embeddings.npz` for the same full shot ranges. The
`--camera-view` option may be repeated. Because a shot-level view
classification is not a verified court homography, BFMD court hints and
manual review still determine whether a court ROI is suitable.

For a smoke test use `--max-shots 12 --clusters 3 --frames 20`.
Rerun with `--reembed` when input shots/video or embedding setup changes,
and `--redetect` to replace cached boundaries. Avoid reusing an output
directory between different videos.

## Output

```text
data/benchmark/bfmd/<match>/
  shot_ranges.csv                  # TransNetV2, end exclusive
  shot_embeddings.npz              # 25/50/75 samples + DINOv2 means
  shot_clusters.csv                # labels + exact shot/sample provenance
  cluster_camera_views.json        # all needs_review until inspected
  candidate_frames.csv             # selected paths, shot/cluster, view, split, court hint
  images/frame_XXXXXXXX.png        # lossless full frames for manual masks
  court_roi_previews/              # only when time-nearby usable court corners
  cluster_review/cluster_XX.jpg    # contact sheets for reviewing clusters
  annotations_PRELABELS_NOT_GROUND_TRUTH.json
  summary.json
```

BFMD's `court_perframe` flags are **valid/refit/no_court**. This script uses
only a nearby observation (within 15 source frames), and only valid/refit
corners may produce an ROI preview; the hint is not certified frame-exact
geometry, especially if a camera cut occurs between samples.

For singles, BFMD may have per-rally pose-detection bounding boxes;
these are copied as **unreviewed prelabels** when present. For doubles,
the released BFMD package does not provide matching four-player
bounding-box/mask annotations: those must be created manually.
Empty `players` in this preparation artifact means *unlabelled*, **not**
that zero players are present. Never use this JSON directly as benchmark ground
truth or report model accuracy against it.

Review cluster contact sheets and replace
`cluster_camera_views.json` values with actual semantic labels
(`wide`, `partial_court`, `closeup`, `side_angle`, `transition`,
`other`, `needs_review`). Rerunning the script reuses this reviewed map.
For any visually mixed cluster, create `shot_view_overrides.json` in
the same output directory, keyed by shot IDs; the runner applies the
shot-level overrides on rerun. See the format in
[shot-camera-view-clustering.md](shot-camera-view-clustering.md).

The script marks ~20% of unique sampled **shot IDs** as `holdout`, so
multiple frames from one shot cannot end up in different splits. The
remaining frames are `development`. This is for annotation/evaluation
splitting, not a claim of independent BFMD match-level generalization.

Then manually correct player boxes and draw segmentation masks in your
annotation tool, separately maintain a shot-held-out validation set, and
use the existing `evaluation.cli evaluate` command. Without that manual
verification, **Milestone 1 detector accuracy cannot be claimed**.

## Prelabel players and export a CVAT annotation task

After extracting BFMD candidate PNG frames, use the existing YOLO11
segmentation adapter to propose each visible person and its mask. These
predictions have **not** been court-filtered and will include occasional
spectators/officials, duplicate players, and missing players. They must be
corrected in CVAT; never evaluate YOLO against its own prelabels.

```bash
# Full candidate frame set, including every sampled image:
MATCH_DIR=data/benchmark/bfmd/<match>

uv run python -m badminton_analysis.evaluation.cli predict \
  --video BFMD_data/videos_doubles/<match>.mp4 \
  --annotations "$MATCH_DIR/annotations_PRELABELS_NOT_GROUND_TRUTH.json" \
  --output "$MATCH_DIR/yolo_predictions.json" \
  --confidence 0.1

# Export the frames and proposed boxes/masks as CVAT COCO 1.0:
uv run python -m badminton_analysis.evaluation.cli export-cvat \
  --annotations "$MATCH_DIR/annotations_PRELABELS_NOT_GROUND_TRUTH.json" \
  --predictions "$MATCH_DIR/yolo_predictions.json" \
  --images "$MATCH_DIR/images" \
  --output "$MATCH_DIR/cvat_UNVERIFIED_prelabels.zip"
```

Replace the <match> placeholder with the actual BFMD video stem and match
path. For singles videos, use the actual nested path under `videos/`.
The YOLO `predict` command runs from the full source video using exact
absolute frame indices, matching the previously extracted PNG frame names.

Import this ZIP into a CVAT task using the COCO 1.0 dataset import workflow.
The `player` category is a **candidate** for manual review. Review
**every** image, remove off-court people, add missing players, correct tight
XYXY boxes and visible-body masks, and confirm genuine negative frames.
The polygons are generated from external mask contours; holes may not
survive COCO polygon conversion and must be manually corrected. Keep the
separate frame/shot-based holdout split when evaluating.

After manually correcting all images in CVAT, export the reviewed task
as COCO 1.0 (JSON or ZIP). Then run:

```bash
uv run python -m badminton_analysis.evaluation.cli import-cvat \
  --coco "$MATCH_DIR/cvat_REVIEWED.zip" \
  --annotations "$MATCH_DIR/annotations_PRELABELS_NOT_GROUND_TRUTH.json" \
  --output "$MATCH_DIR/annotations_GROUND_TRUTH.json" \
  --confirm-reviewed

uv run python -m badminton_analysis.evaluation.cli evaluate \
  --annotations "$MATCH_DIR/annotations_GROUND_TRUTH.json" \
  --predictions "$MATCH_DIR/yolo_predictions.json" \
  --output "$MATCH_DIR/player_metrics.json"
```

The converter uses exact full-resolution pixels, creates instance-mask PNGs
from reviewed polygons or COCO RLE (including masks with holes), and preserves frame indices, camera-view labels,
shot IDs, and development/holdout splits. It rejects dropped frames,
unrecognized classes, out-of-bounds boxes and malformed masks, rather
than silently corrupting ground truth. If you only corrected boxes, mask-IoU
is unavailable for those instances. The `--confirm-reviewed` flag is a
declaration from the annotator, not automatic verification. Never pass a
machine-generated prelabel ZIP as reviewed ground truth.

## Status

This PR contains an executable adapter with pure data handling tests.
The real BFMD match inference has **not** been executed here: neither the
copyrighted source MP4, downloaded BFMD annotation package, nor pretrained
TransNetV2/DINOv2 weights are present in the working environment.
