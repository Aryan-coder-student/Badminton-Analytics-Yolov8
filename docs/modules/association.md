# Module — Within-Shot Track Association

## Purpose

When ByteTrack fragments one physical player into two local tracks, the association layer evaluates whether a later track is a plausible continuation of an earlier lost track.

## Inputs

The current association logic uses:

- final DINOv2 appearance descriptors
- final HSV descriptors
- source/end frame information
- temporal overlap checks
- court foot positions projected to metres

## What it does not do

The association module does not replace IDs and does not update ByteTrack state. It only evaluates/ranks candidates.

## Verified example

For the sample shot:

```text
Track 2 -> Track 5
DINOv2 similarity: 0.8888
HSV similarity:    0.9907
Elapsed frames:    25
Court distance:    0.26 m
Result:             likely association
```

All other ordered pairs in the smoke run were rejected because they overlapped in time or occurred in reverse temporal order.
