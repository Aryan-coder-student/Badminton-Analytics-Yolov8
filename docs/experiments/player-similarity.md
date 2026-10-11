# Experiment — One-Image Player Appearance Grouping

## Setup

For one court image:

1. segment the four accepted players
2. create masked player crops
3. extract normalized DINOv2 features
4. extract mask-aware HSV descriptors
5. combine the appearance information
6. cluster into two groups with KMeans

## Recorded result

The test grouped `P1/P2` together and `P3/P4` together.

## Limitation

This is a one-image appearance-grouping experiment. The `P1`–`P4` labels are deterministic per-image labels, not persistent match identities.
