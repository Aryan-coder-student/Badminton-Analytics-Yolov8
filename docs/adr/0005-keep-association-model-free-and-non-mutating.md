# ADR 0005 — Keep Track Association Model-Free and Non-Mutating

## Status

Accepted and implemented.

## Context

The project already has appearance descriptors, timing information, and court coordinates. The immediate need is to evaluate whether two local tracks plausibly belong to the same player.

## Decision

Implement association as pure scoring over existing measurements:

- DINOv2 cosine similarity
- HSV cosine similarity
- frame gap
- temporal overlap
- court distance

The association layer reports/ranks candidates but does not rewrite tracker IDs.

## Consequences

- association behavior is deterministic and unit-testable
- raw tracking remains untouched
- persistent identity can be added later as a separate layer
