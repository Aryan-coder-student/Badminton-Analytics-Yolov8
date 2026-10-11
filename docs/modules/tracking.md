# Module — Local Player Tracking

## What it does

The tracking contract is stateful. `Tracker("bytetrack")` wraps an Ultralytics ByteTrack adapter and is updated frame by frame.

## Meaning of a local track

A track is a sequence of detections that ByteTrack believes belong to the same physical object across consecutive frames.

Example:

```text
Frame 100 -> Track 2
Frame 101 -> Track 2
Frame 102 -> Track 2
```

## Verified failure case

In the sample shot, five local IDs were observed even though there were four physical players. The near-left player was initially Track 2 and later became Track 5 during fast movement.

This is the reason a separate appearance/association layer was added.
