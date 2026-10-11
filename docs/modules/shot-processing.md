# Module — Shot Processing

## What it does

The recorded broadcast is handled as a sequence of camera shots rather than one uninterrupted tracking stream.

Representative frames are extracted at 25%, 50%, and 75% of each shot from the scene metadata.

## Why

Broadcast badminton contains camera cuts, close-ups, replays, and other views. Sampling several interior points of a shot reduces the chance that experiments use only a transition frame.

## Verified artifact

The current extracted shot-frame dataset contains 1,431 images.
