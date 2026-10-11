# ADR 0003 — Keep Appearance Processing Outside `model/`

## Status

Accepted and implemented.

## Context

Appearance code combines masks, crops, embeddings, color descriptors, observations, track aggregation, and association evidence. It is domain feature-processing logic rather than a single DL model adapter.

## Decision

Keep reusable appearance logic under `src/badminton_analysis/appearance/` instead of `model/`.

## Consequences

- model adapters remain focused on inference contracts
- appearance aggregation can combine outputs from several models/algorithms
- track association logic stays independent from any single embedding implementation
