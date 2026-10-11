# Model and Algorithm Catalog — Implemented Components

This catalog lists only models/algorithms already present in the project.

## Shot processing

### TransNetV2 / shot-boundary data

**Purpose:** split recorded broadcast video into continuous camera shots so downstream tracking is not expected to survive arbitrary broadcast cuts.

**Current use:** shot metadata is consumed by scratchpad tooling, and representative frames are extracted at 25%, 50%, and 75% of each shot.

---

## Court detection

### YOLO pose court-keypoint model

**Purpose:** detect 30 badminton-court keypoints from a broadcast frame.

**Why it is used:** the keypoints provide correspondences between image coordinates and a regulation badminton court.

**Current output:** court keypoints and a homography used to reconstruct/interpret court geometry.

---

## Player detection / segmentation

### Ultralytics YOLO detector

**Purpose:** generic object detection through the project detection strategy interface.

**Current status:** implemented and retained as a generic detector adapter. It is not the only player path used in experiments.

### YOLO11 segmentation (`yolo11n-seg.pt`)

**Purpose:** detect players and obtain a pixel mask for each person.

**Why it is useful here:** the mask lets appearance extraction ignore the court/background and use only player pixels.

**Current output:** detection metadata plus a full-resolution boolean mask.

---

## Local tracking

### ByteTrack

**Purpose:** maintain local object IDs across consecutive frames inside one continuous shot.

**Why it is used:** it links frame-level player detections into short-lived tracks without requiring a persistent identity system.

**Important limitation already observed:** local IDs can fragment. In the verified sample shot, the near-left player changed from local Track 2 to Track 5 during fast movement.

---

## Appearance embeddings

### DINOv2

**Purpose:** convert masked player crops into high-dimensional visual descriptors.

**Current use:** player appearance comparison and track-level appearance aggregation.

**Current descriptor size:** 768 dimensions.

### CLIP

**Purpose:** alternate image embedding strategy behind the shared embedding interface.

**Current status:** implemented as an experimental candidate.

### ResNet

**Purpose:** alternate image embedding strategy behind the shared embedding interface.

**Current status:** implemented as an experimental candidate.

---

## Color appearance

### Mask-aware HSV descriptor

**Purpose:** explicitly capture player color information using only pixels selected by the segmentation mask.

**Current behavior:** all valid masked player pixels are used; the earlier torso-only cutoff was removed.

**Current descriptor size:** 96 dimensions.

---

## Clustering

### KMeans

**Purpose:** unsupervised grouping of embeddings.

**Current uses:** shot-frame clustering experiments and one-image two-group player appearance experiment.

### DBSCAN / HDBSCAN

**Purpose:** alternate clustering strategies behind the shared clustering interface.

**Current status:** implemented for experimentation.

---

## Track association

### Model-free association scoring

This is not a neural network. It combines already-computed measurements:

- DINOv2 cosine similarity
- HSV cosine similarity
- frame gap
- temporal overlap validity
- court-plane distance

The association code ranks predecessor/successor track candidates but does not rewrite ByteTrack IDs.
