# Reuse existing shot-boundary and visual-clustering pipeline

## Purpose

Use **existing** badminton-video shot classification and clustering as
camera-view *prelabels* for Milestone 1. This is a reusable part of the
player-detection benchmark, not a new supervised camera classifier.

The initial experimental implementation is described in
[shot-clustering.md](../experiments/shot-clustering.md): 1,431 sampled frames,
three interior frames per shot, DINOv2 and KMeans. Its recorded run found six
**visual clusters** with silhouette 0.207529. That result is an unsupervised
visual partition, not validated camera-angle classification accuracy.

## Exact algorithm

1. **Scene/shot boundaries.** Use the existing TransNetV2 shot-boundary model
   to detect camera cuts and produce ordered `[start_frame, end_frame)`
   intervals. This model detects *cuts*, not semantic main-court/close-up views.
   The boundary-detection experiment currently lives in ignored `scratchpad/`
   files; the new tracked code takes its **exported** intervals.
2. **Representative frames.** Sample each shot at 25%, 50%, and 75% of the
   inclusive range of valid frame indices. For extremely short shots, duplicated
   sample indices are deduplicated rather than decoding the same frame twice.
3. **Embedding.** Reuse `Embedding("dinov2")` to generate one RGB embedding per
   sampled frame. Average those embeddings into **one feature vector per shot**.
   Keep per-frame sample indices and shot boundaries in an NPZ cache.
4. **Visual clustering.** Reuse the project's `Cluster("kmeans", ...)` facade.
   Default to six clusters to reproduce the old experiment, but treat that
   as a configurable experiment, not a fixed semantic class count.
   Print silhouette for diagnostics.
5. **Human semantic interpretation.** Review a contact sheet of representative
   shots from **each** cluster, then map cluster IDs to `wide`,
   `partial_court`, `closeup`, `side_angle`, `transition`,
   `other`, or `needs_review`. A low silhouette or mixed membership calls for
   `needs_review` or a per-shot override, not an invented label.
6. **Annotate actual court players.** On selected frames, review and correct
   YOLO-generated boxes and instance masks. No persistent P1–P4 ID is required
   for this detection/segmentation benchmark.

Court-keypoint geometry should independently validate full-court usability.
Being assigned to a cluster associated with wide views is **not sufficient**
evidence of a correct court grid. Cuts can be missed and zoom/pan can occur
inside an otherwise continuous shot.

## Step 1 — Export existing shot intervals

Prepare `data/benchmark/shot_ranges.csv` from your existing TransNetV2
metadata (do not rerun boundary detection just to create a second pipeline):

```csv
shot_id,start_frame,end_frame
18,5115,5272
23,6200,6350
```

These rows are **illustrative**. Provide real zero-based absolute frame indices
for the actual match. `start_frame` is inclusive, `end_frame` exclusive.
Do not feed shot timestamps in seconds directly; convert to frame indices
using the source video's frame timing.

If your existing clustering experiment already has the shot-level
`shot_id,cluster_id,start_frame,end_frame` assignments, **skip steps 2 and
3** and use that CSV directly.

## Step 2 — Reuse the existing DINOv2 embedding model

```bash
uv run python -m badminton_analysis.evaluation.cli embed-shots \
  --video data/sample2.mp4 \
  --shot-ranges data/benchmark/shot_ranges.csv \
  --output data/benchmark/shot_embeddings.npz
```

The NPZ caches the averaged embeddings and original sampled frame indices.
It does **not** contain the frames or inference video. The existing embedding
model is reused for every shot, avoiding per-shot weight reloads.

## Step 3 — Cluster shots from the cached embeddings

```bash
uv run python -m badminton_analysis.evaluation.cli cluster-shots \
  --cache data/benchmark/shot_embeddings.npz \
  --clusters 6 \
  --output data/benchmark/shot_clusters.csv
```

The CSV contains `shot_id,cluster_id,start_frame,end_frame` plus the three
sample frame indices. The clustering is deterministic for the same cache and
KMeans seed, but there is no guarantee that cluster 0 is always the main court.

## Step 4 — Review cluster semantics once

```bash
uv run python -m badminton_analysis.evaluation.cli camera-view-template \
  --shot-clusters data/benchmark/shot_clusters.csv \
  --output data/benchmark/cluster_camera_views.json
```

Initially every cluster is `needs_review`:

```json
{
  "0": "needs_review",
  "1": "needs_review"
}
```

After visually reviewing representative shots per cluster, update each entry.
For mixed clusters, keep the cluster label conservative and optionally supply
`data/benchmark/shot_view_overrides.json`:

```json
{
  "216": "side_angle",
  "270": "closeup"
}
```

Those IDs/views are **examples** only; no such labels have been verified.

## Step 5 — Attach reviewed views to the player ground truth

The input `annotations.json` must already contain manually verified player
boxes, and masks if available. The output must be in the *same directory*
because mask paths are relative.

```bash
uv run python -m badminton_analysis.evaluation.cli camera-views \
  --shot-clusters data/benchmark/shot_clusters.csv \
  --cluster-map data/benchmark/cluster_camera_views.json \
  --shot-overrides data/benchmark/shot_view_overrides.json \
  --annotations data/benchmark/annotations.json \
  --output data/benchmark/annotations_with_views.json
```

Omit `--shot-overrides` if there are none. The output contains:
`shot_id`, `visual_cluster_id`, `camera_view`, and
`camera_view_source` (`reviewed_cluster` or `shot_override`).

Existing player boxes/masks remain unchanged. Already-labelled conflicting
camera views and uncovered frames fail explicitly instead of being overwritten.

## Validation status / limits

The reader, percentile sampler, cache/export logic and cluster metadata tests
are deterministic and do not require model downloads. A small synthetic video
validates decoding, embedding averaging, and cache round-tripping. The complete
TransNetV2 → DINOv2 → KMeans → annotated-video experiment has **not** been run
here because `data/sample2.mp4` and the existing TransNetV2 scratchpad
artifacts are not available in this environment.

Camera-view accuracy still requires review: the silhouette score alone does not
establish semantic correctness. Keep all raw shot assignments for auditability.
