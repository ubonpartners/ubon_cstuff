# Architecture

This document describes the tracking subsystem located under `src/track/`.

## Overview

The tracking pipeline consumes video frames or compressed streams and produces
per‑object track data plus optional auxiliary outputs such as embeddings or
JPEG snapshots.  The subsystem is made of several cooperating modules:

- **track_stream** – entry point for video data.  It accepts raw frames,
  compressed H26x/JPEG inputs, RTP packets or video files.  The stream module
  decodes input, applies motion tracking and rate limiting, then dispatches
  frames to inference and tracking stages.
- **track_shared** – shared context for multiple streams.  It manages worker
  threads, CUDA initialisation, inference threads and a registry of active
  streams.  Configuration is loaded from YAML and reused when possible.
- **track_aux** – optional post‑processing for tracks.  It generates JPEG
  thumbnails and runs auxiliary models (face, CLIP, FIQA) to attach embeddings
  or quality metrics to tracked objects.
- **trackset** – utility for loading ground‑truth tracking data from YAML files
  for evaluation or replay.
- **Algorithms** – two trackers are provided:
  - `bytetracker/` implements the ByteTrack algorithm.
  - `utrack/` provides a lightweight tracker that predicts positions and
    integrates motion tracking.

## Data Flow

```text
Input (RTP/H26x/JPEG/frames)
          ↓
    track_stream
      ├─ decode & rate limit
      ├─ motion tracker
      ├─ main inference
      └─ tracker (ByteTrack or utrack)
              ↓
        track_aux (embeddings/JPEGs)
              ↓
        application callback
```

Frames are processed asynchronously through work queues so that decoding,
inference and auxiliary tasks can overlap.  `track_shared` initialises a thread
pool and CUDA context once and allows multiple `track_stream` instances to run
concurrently.
