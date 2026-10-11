# Match ReID Rerun Diagnosis — 2026-09-19

## Result

The clean rerun finished on 2026-09-13 and saved:

- `data/player_predictions/match_reid/sample2_match_persistent_identities.mp4`
- `data/player_predictions/match_reid/frame_identities.json`
- 70 shot-level descriptor archives under `data/player_predictions/match_reid/galleries/`

The MP4 fully decodes, contains H.264 video and source AAC audio, and is 801 MB.
However, the persistent-identity goal is **not met**: most tracked observations
still show `unresolved`.

## Measured coverage

| Measure | New rerun |
| --- | ---: |
| Court shots processed | 70 |
| Frames with tracking records | 19,532 |
| Player observations | 71,134 |
| `P1` / `P2` / `P3` / `P4` observations | 9,053 / 7,597 / 5,356 / 4,105 |
| Unresolved observations | 45,023 (63.29%) |
| Annotated frames with at least one unresolved player | 19,323 (98.93%) |
| Four-detection frames with all four IDs | 105 / 13,507 (0.78%) |
| Unresolved shot-local tracks | 505 / 641 |

The previous output had 61.50% unresolved observations, but it processed 226
shots rather than 70; the percentages are not a like-for-like accuracy
comparison. Both outputs fall far short of stable four-player labelling.

## Why tracks remain unresolved

Replaying the saved descriptor archives through the existing matcher reproduced
the JSON assignments exactly. The 45,023 unresolved frame observations break
down as follows:

| Cause | Local tracks | Frame observations |
| --- | ---: | ---: |
| Proposed match failed only the `margin > 0.05` rule | 137 | 28,267 |
| No assignment slot in the four-player one-to-one matcher | 360 | 16,005 |
| Failed both score and margin rules | 7 | 750 |
| Failed only the `score >= 0.70` rule | 1 | 1 |

The matcher proposes at most one local track per persistent player in each
shot. Shots often contain more than four local tracks, so additional tracks
receive no proposal at all. Track fragmentation is a likely contributor, but
some short tracks may also be false detections; the saved data alone cannot
distinguish every case. Of the proposals it does make, the margin gate rejects
about half because the assigned player's score is not clearly better than the
alternatives. Simply lowering the margin would increase labels but risks
incorrect player IDs; accuracy cannot be measured without manually labelled
examples.

## Separate coverage issue

The stricter court-view gate removed the audited close-ups and side angles, but
it accepted no shots after Shot 313, ending at 00:41:09.57. The remaining ~13
minutes of the 54:10 source therefore have no player overlays. The output also
contains 97,500 video frames versus 97,501 in the source, apparently because
the final FFmpeg command uses `-shortest`.

## Next corrective experiment

1. Group non-overlapping ByteTrack fragments into shot-local player identities
   **before** the four-way cross-shot assignment, reusing the existing
   association/identity utilities. Preserve raw track IDs in the JSON.
2. Build a small manually labelled set of wide shots, then measure false matches
   and unresolved coverage before adjusting the margin or appearance model.
3. Inspect post-Shot-313 camera geometry separately; admit only genuinely valid
   courts rather than relaxing the wide-view gate globally.
4. Correct the one-frame export loss and verify source/output frame counts.

No identity thresholds or saved results were changed during this diagnosis.

## Follow-up: conservative local-fragment recovery

The match script now collects each track's first/last court position during its
existing tracking pass. It reuses the established DINOv2/HSV, time-gap, and
court-distance policy to find likely non-overlapping fragments **within a
shot**. Cross-shot Hungarian assignments and the `0.70` score / `0.05` margin
gates are unchanged. After those decisions, an accepted P-ID is copied to a
linked, unresolved local fragment only when the group has exactly one accepted
P-ID. Groups with conflicting accepted IDs are not propagated. Raw ByteTrack
IDs and persistent appearance galleries remain unchanged by propagation.

The read-only replay command is
`uv run python -m scratchpad.audit_reid_grouping`. On the saved 70-shot data,
it linked 24 local fragments and raised resolved observations from 26,111 to
27,558 (+1,447). Estimated unresolved observations fell from 45,023 (63.29%)
to 43,576 (61.26%). It changed zero previously accepted labels and lost zero
previously accepted labels. This is a **coverage** measure, not identity
accuracy: no hand-labelled truth set exists, and the replay estimates each
shot's court transform from a middle frame. The existing JSON and MP4 were not
overwritten; a full inference run is required to produce a new video.

An earlier experiment that grouped galleries *before* Hungarian matching was
rejected: although it gained labels overall, it made 6,044 previously accepted
observations unresolved. This is why the implemented change propagates only
after the original cross-shot decisions.

## Remaining issues and research direction

1. The strict margin remains the largest rejection cause (28,267 baseline
   observations). Similar player clothing makes DINOv2/HSV scores close; this
   is an inference from the measured score margins, not a verified identity
   accuracy claim. [DINOv2](https://arxiv.org/abs/2304.07193) was designed as
   a general visual feature encoder, whereas [BoT-SORT](https://arxiv.org/abs/2206.14651)
   explicitly combines motion and appearance for tracking.
2. Short ByteTrack fragments still exceed the four cross-shot assignment
   slots (16,005 baseline observations had no proposal). This local recovery
   only helps fragments supported by the conservative association policy.
   [ByteTrack](https://arxiv.org/abs/2110.06864) addresses low-confidence
   detections, and [StrongSORT](https://arxiv.org/abs/2202.13514) studies
   offline tracklet linking; both are relevant avenues, not evidence that a
   threshold change is safe for this video.
3. Low combined appearance score accounts for 751 baseline observations,
   including seven tracks also failing the margin rule. A calibrated,
   person-specific ReID model may help, but should first be tested on manually
   labelled pairs from this match.
4. The court-view gate still stops after Shot 313. Contact-sheet inspection
   shows later views include close-ups and alternate wide camera angles. The
   current fixed-view court geometry does not safely cover those angles, so
   this is a separate *coverage* issue rather than an unresolved-ID match.
