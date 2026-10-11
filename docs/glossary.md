# Glossary

## Detection
A frame-level object prediction containing information such as bounding box, confidence, and class.

## Segmentation
A detection plus a pixel mask identifying the image pixels belonging to that object.

## Foot point
The bottom-centre point of a person's bounding box, used as an approximation of where the player touches the court.

## Homography
A projective transformation mapping image coordinates to the canonical badminton-court plane.

## Track
A sequence of frame-level detections that a tracker believes belong to the same moving object over consecutive frames.

## Track ID
The temporary/local identifier assigned by ByteTrack. It is not a persistent real-player identity.

## TrackedSegmentation
The project object that keeps a local track ID paired with the exact detection and segmentation mask.

## Appearance observation
One sampled tracked-player observation containing its frame, tracked segmentation, masked crop, DINOv2 embedding, and HSV descriptor.

## Track appearance descriptor
An aggregated visual representation built from multiple appearance observations belonging to one local track.

## Association candidate
A possible predecessor/successor relationship between two local tracks, scored using appearance, timing, and court-distance evidence.

## Persistent player identity
An experimental anonymous label such as `P1` that can represent the same
physical player across several MAIN_COURT shots. It is separate from the raw
ByteTrack ID and is assigned only when the appearance evidence is confident.

## Appearance gallery
A collection of several normalized DINOv2 and HSV appearance samples. A local
track has a track gallery; an accepted anonymous player has a persistent gallery
containing samples from more than one shot.

## Assignment margin
The difference between a proposed player's combined appearance score and the
track's second-best possible player. A small or negative margin means the
result is ambiguous, so the track remains unresolved.
