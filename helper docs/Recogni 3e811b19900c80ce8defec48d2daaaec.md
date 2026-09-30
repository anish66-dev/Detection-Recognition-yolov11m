# Recogni

## Purpose

The Recognition Worker consumes person detection events, detects visible faces within each detected person, extracts facial embeddings using ArcFace, performs identity matching against the enrolled face database, and publishes structured recognition events for downstream analytics.

It is responsible only for **face recognition and identity resolution**. It performs **no business rule evaluation, alert generation, zone monitoring, or crowd analytics**.

---

## Responsibilities

- Subscribe to the detection event stream.
- Retrieve the associated frame using the frame reference.
- Locate the detected person using the bounding box.
- Perform face detection (SCRFD).
- Crop and align detected faces.
- Generate 512-dimensional ArcFace embeddings.
- Compare embeddings against the enrolled face database using pgvector.
- Classify identities as:
    - Enrolled Person
    - Visitor
    - Unknown
- Perform face deduplication.
- Publish `RecognitionEvent`s.
- Write audit entries for enrollment or recognition operations where required.
- Acknowledge processed Redis messages.

---

## Non-Responsibilities

The Recognition Worker **does not**:

- Detect people.
- Track people.
- Generate alerts.
- Detect intruders.
- Monitor zones.
- Count people.
- Estimate crowd density.
- Store business events.

Those responsibilities belong to downstream event-processing workers.

---

# Technology Stack

| Component | Purpose |
| --- | --- |
| SCRFD (InsightFace) | Face detection |
| ArcFace (InsightFace) | Face embedding generation |
| pgvector | Nearest-neighbor embedding search |
| PostgreSQL | Enrolled face database |
| Redis Streams | Event transport |
| Pydantic | Event validation |
| shared/schemas | Shared event contracts |
| shared/audit | Audit logging |

---

# Input

Consumes:

```
events:detections
```

Message type:

```python
DetectionEvent
```

Each DetectionEvent represents a tracked person.

---

# Output

Publishes:

```
events:recognitions
```

Message type:

```python
RecognitionEvent
```

One RecognitionEvent is produced for each successfully processed DetectionEvent.

---

# Processing Pipeline

```
DetectionEvent
        │
        ▼
Validate Event
        │
        ▼
Retrieve Frame
        │
        ▼
Crop Person Region
        │
        ▼
SCRFD Face Detection
        │
        ▼
Face Alignment
        │
        ▼
ArcFace Embedding
        │
        ▼
pgvector Similarity Search
        │
        ▼
Identity Classification
        │
        ▼
Deduplication
        │
        ▼
Publish RecognitionEvent
        │
        ▼
Audit (where applicable)
        │
        ▼
ACK Redis Message
```

---

# Detailed Workflow

## Step 1 — Consume DetectionEvent

The worker subscribes to the detection event stream.

Upon receiving a `DetectionEvent`:

- Validate schema.
- Extract camera metadata.
- Extract person bounding box.
- Extract frame reference.
- Extract track ID.

---

## Step 2 — Retrieve Frame

Using the `frame_provider` and `frame_reference`:

- Retrieve the corresponding frame.
- Decode into an OpenCV image.

---

## Step 3 — Crop Person Region

Crop the detected person's bounding box from the frame.

This reduces the search area for face detection.

---

## Step 4 — Face Detection

Run SCRFD on the cropped person image.

If no face is detected:

- Recognition is skipped.
- A `RecognitionEvent` may still be published with status indicating that no face was available.

---

## Step 5 — Face Alignment

Detected faces are aligned according to the InsightFace preprocessing pipeline.

This improves embedding consistency across different poses.

---

## Step 6 — Embedding Generation

Run ArcFace inference.

Generate a fixed-length **512-dimensional facial embedding** representing the detected face.

---

## Step 7 — Identity Search

Query the enrolled embedding database using pgvector.

Perform nearest-neighbor similarity search against enrolled identities.

Retrieve:

- nearest identity
- similarity score

---

## Step 8 — Identity Classification

Based on configurable similarity thresholds:

- **Enrolled Person** – embedding matches an enrolled identity.
- **Visitor** – recognized from a temporary visitor enrollment.
- **Unknown** – no valid match found.

Confidence scores are included in the recognition result.

---

## Step 9 — Face Deduplication

Prevent duplicate recognition of the same individual across consecutive frames.

Typical strategies include:

- Track ID association.
- Recognition cooldown windows.
- Confidence-based update rules.

---

## Step 10 — Publish RecognitionEvent

Publish the recognition result to:

```
events:recognitions
```

The event includes:

- identity classification
- recognized person (if any)
- confidence
- track ID
- frame reference
- timestamp

---

## Step 11 — Audit Logging

Where applicable (for example, successful enrollment or identity assignment), create an audit entry using the shared Audit Writer within the same database transaction.

Routine recognition events generally do not require audit records unless specified by system policy.

---

## Step 12 — ACK

After successful publication (and any required audit logging), acknowledge the consumed Redis message.

---

# Scaling Strategy

Recognition is GPU-bound.

Scaling is achieved by:

- multiple recognition worker containers
- GPU resource limits
- batching multiple face crops into a single ArcFace inference batch

Face embedding generation is typically the most GPU-intensive stage.

---

# Failure Handling

## Invalid DetectionEvent

Discard and log.

ACK the message.

---

## Frame Retrieval Failure

Retry according to retry policy.

If retrieval repeatedly fails:

- log error
- publish failure metric
- future enhancement: dead-letter queue

---

## No Face Detected

Not considered an error.

Publish a RecognitionEvent indicating that recognition could not be performed due to the absence of a detectable face.

---

## Recognition Failure

Log exception.

Do not publish invalid RecognitionEvents.

Retry according to retry policy.

---

## Database Failure

Retry database operation.

If retries fail:

- log error
- do not acknowledge until retry policy is exhausted

---

## Redis Failure

Reconnect automatically through the shared BaseStreamConsumer.

---

# Performance Considerations

The worker should:

- keep SCRFD and ArcFace models loaded in GPU memory
- batch face crops whenever possible
- reuse database connections
- minimize image copies
- cache frequently accessed embeddings if appropriate
- avoid redundant recognition of the same track

---

# Metrics

Expose metrics such as:

- Detection events processed/sec
- Recognition latency
- Face detection success rate
- Recognition accuracy (offline evaluation)
- Average embedding generation time
- Database query latency
- Recognition confidence distribution
- Unknown identity rate
- Queue lag

---

# Dependencies

Consumes:

```
events:detections
```

Publishes:

```
events:recognitions
```

Depends on:

- Redis Streams
- Frame Provider
- SCRFD
- ArcFace
- PostgreSQL
- pgvector
- Shared event schemas
- Shared audit library

---

# Sequence Diagram

```
DetectionEvent
       │
       ▼
Recognition Worker
       │
       ├── Retrieve Frame
       ├── Crop Person
       ├── SCRFD
       ├── Face Alignment
       ├── ArcFace
       ├── pgvector Search
       ├── Deduplication
       ├── RecognitionEvent
       ▼
events:recognitions
```

*Note: Separate **enrollment** from the Recognition Worker. The Recognition Worker should only **consume embeddings** from the database for matching. A dedicated enrollment workflow (API + background task) should be responsible for capturing faces, generating reference embeddings, and inserting them into pgvector.*