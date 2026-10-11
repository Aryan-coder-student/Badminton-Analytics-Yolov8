# ADR 0004 — Represent Appearance at Track Level, Not One Frame Only

## Status

Accepted and implemented for within-shot experiments.

## Context

One frame can be blurry, partially occluded, or show an unhelpful player view. ByteTrack already provides multiple observations for a local track.

## Decision

Sample multiple observations from each local track, extract DINOv2 and HSV features, and aggregate them into normalized track descriptors.

## Consequences

- appearance comparison uses evidence from multiple frames
- each track can be represented more robustly than with a single crop
- short tracks, such as Track 2 in the sample, naturally have fewer observations and therefore weaker evidence
