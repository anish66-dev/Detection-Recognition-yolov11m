# Detection

## Purpose

The Detection Worker is the first analytics service in the pipeline. It consumes incoming camera frames, performs real-time person detection and multi-object tracking, and publishes structured detection events for downstream services.

It is responsible only for **detecting and tracking people**. It performs **no face recognition, business rule evaluation, alert generation, or database writes**.

---

## Responsibilities

- Subscribe to one or more camera frame streams.
- Retrieve frames using the reference contained in `FrameEvent`.
- Perform person detection using YOLO11m.
- Track detected people across frames using ByteTrack.
- Generate stable track IDs.
- Filter low-confidence detections.
- Publish `DetectionEvent`s.
- Acknowledge processed Redis messages.

---

## Non-Responsibilities

The Detection Worker **does not**:

- Recognize faces.
- Perform face tokenization.
- Detect restricted persons.
- Monitor virtual zones.
- Count people.
- Estimate crowd density.
- Generate alerts.
- Store detection history in PostgreSQL.
- Make business decisions.

Those responsibilities belong to downstream services.

---

# Technology Stack

| Component | Purpose |
| --- | --- |
| YOLO11m | Person detection |
| ByteTrack | Multi-object tracking |
| OpenCV | Image decoding and preprocessing |
| PyTorch | Model inference |
| Redis Streams | Event transport |
| Redis / Frame Provider | Frame retrieval |
| Pydantic | Event validation |

---

# Input

Consumes:

```
frames:{camera_id}
```

Message type:

```python
FrameEvent
```

Example:

```
FrameEvent
 ├── camera_id
 ├── frame_reference
 ├── frame_provider
 ├── frame_shape
 ├── timestamp
 └── frame_seq
```

---

# Output

Publishes:

```
events:detections
```

Message type:

```python
DetectionEvent
```

Each detected person generates one DetectionEvent.

---

# Processing Pipeline

```
FrameEvent
      │
      ▼
Validate Event
      │
      ▼
Retrieve Frame
      │
      ▼
Decode Image
      │
      ▼
YOLO11m Inference
      │
      ▼
Filter Person Class
      │
      ▼
ByteTrack
      │
      ▼
Create DetectionEvents
      │
      ▼
Publish Events
      │
      ▼
ACK Redis Message
```

---

# Detailed Workflow

## Step 1 — Consume FrameEvent

The worker subscribes to one or more camera streams.

Upon receiving a `FrameEvent`:

- Validate schema.
- Extract frame reference.
- Extract camera metadata.

---

## Step 2 — Retrieve Frame

Using the `frame_provider` and `frame_reference`:

- Retrieve the frame.
- Decode into an OpenCV image.

The retrieval mechanism is abstracted and independent of the storage backend.

---

## Step 3 — Preprocessing

Perform minimal preprocessing required by the detection model.

Examples:

- Resize
- Color conversion
- Tensor conversion

No business logic is applied.

---

## Step 4 — YOLO11m Detection

Run inference.

Only detections of class:

```
person
```

are retained.

Each detection contains:

- bounding box
- confidence score
- class

---

## Step 5 — ByteTrack

The detections are passed into ByteTrack.

ByteTrack associates detections with existing tracks and assigns persistent track IDs.

Output:

```
Track ID 17
Track ID 31
Track ID 44
...
```

Track IDs remain stable while a person remains visible.

---

## Step 6 — Detection Filtering

Discard detections that fail configured thresholds.

Possible filters:

- confidence threshold
- minimum bounding box size
- invalid coordinates

---

## Step 7 — Publish DetectionEvent

For every tracked person:

```
DetectionEvent
```

is published to

```
events:detections
```

The event contains detection metadata only.

No image data is embedded.

---

## Step 8 — ACK

After all DetectionEvents are successfully published,

the consumed Redis message is acknowledged.

---

# Scaling Strategy

Detection is GPU-bound.

Scaling is achieved by:

- multiple GPU-enabled containers
- assigning different camera streams to different workers

Future optimization:

- batch frames from multiple cameras into a single inference batch

to improve GPU utilization.

---

# Failure Handling

## Invalid FrameEvent

Discard and log.

ACK the message to avoid infinite retries.

---

## Frame Retrieval Failure

Retry according to retry policy.

If retrieval repeatedly fails:

- log error
- publish failure metric
- move to dead-letter queue (future enhancement)

---

## YOLO Failure

Log exception.

Do not publish DetectionEvents.

Retry according to retry policy.

---

## ByteTrack Failure

Treat as processing failure.

DetectionEvent must never contain inconsistent tracking information.

---

## Redis Failure

Reconnect automatically using the shared BaseStreamConsumer.

---

# Performance Considerations

The worker should:

- keep YOLO model loaded in GPU memory
- keep ByteTrack state in memory
- minimize image copies
- batch inference where possible
- avoid unnecessary CPU-GPU transfers

---

# Metrics

Expose metrics such as:

- Frames processed/sec
- Detection latency
- Average inference time
- GPU utilization
- Detection count/sec
- Active track count
- Failed frames
- Queue lag

---

# Dependencies

Consumes:

```
frames:{camera_id}
```

Publishes:

```
events:detections
```

Depends on:

- Redis Streams
- Frame Provider (Redis cache/MinIO/etc.)
- YOLO11m weights
- ByteTrack configuration
- Shared event schemas

---

# Sequence Diagram

```
FrameEvent
    │
    ▼
Detection Worker
    │
    ├── Retrieve Frame
    │
    ├── Decode
    │
    ├── YOLO11m
    │
    ├── ByteTrack
    │
    ├── DetectionEvent(s)
    │
    ▼
events:detections
```

*Note:  **"retrieves frames using the `frame_provider` and `frame_reference` from the `FrameEvent`."** Keeping the service independent of whether the hot path ultimately uses Redis, shared memory, or another mechanism while preserving the same event contract.*