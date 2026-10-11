# ADR 0006 — Use Strategy/Factory Facades for Replaceable Model Components

## Status

Accepted in the current model layer.

## Context

Several stages may need alternative implementations during experimentation, including embeddings, clustering, detection, segmentation, and tracking.

## Decision

Expose common interfaces plus client-facing tag-based facades such as `Embedding(...)`, `Cluster(...)`, `Detector(...)`, `Segmenter(...)`, and `Tracker(...)`.

## Consequences

- scratchpad experiments can switch implementations without changing downstream code
- model-specific output is normalized behind project contracts
- unused candidate implementations can later be removed without redesigning the consumers
