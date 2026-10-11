# ADR 0001 — Separate Local Track IDs from Persistent Player Identity

## Status

Accepted for the current architecture.

## Context

ByteTrack IDs are local tracking artifacts. A physical player may receive a new ID after fast motion or missed detections. The sample shot demonstrated this when the near-left player changed from Track 2 to Track 5.

## Decision

Do not treat `track_id` as the final player identity.

Keep raw ByteTrack IDs unchanged and place any identity/association logic above the tracker.

## Consequences

- tracker output remains debuggable and faithful to ByteTrack
- appearance logic can reason about Track 2 and Track 5 without mutating tracking history
- later persistent identity can be introduced without coupling it to one tracker implementation
