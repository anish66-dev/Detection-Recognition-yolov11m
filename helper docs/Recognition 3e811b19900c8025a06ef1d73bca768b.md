# Recognition

## 1. Deterministic Recognition

- “likely Rakshit with confidence 0.81”

| Confidence | Action |
| --- | --- |
| >0.90 | auto-identify |
| 0.75–0.90 | probable match |
| 0.60–0.75 | low confidence |
| <0.60 | unknown |

---

# 2. Use Temporal Confirmation

Recognize multiple frames

```
Track ID 42
Frame 1 → similarity 0.71
Frame 2 → similarity 0.83
Frame 3 → similarity 0.86
Frame 4 → similarity 0.84
```

Aggregate confidence across time using weighted average.

Then identify only after:

- N consistent matches
- within a time window

---

# 3. Multi-Embedding Enrollment

Store:

- multiple embeddings
- multiple lighting conditions
- slight angle variation
- different expressions

Then:

- cluster them
- average them
- or use nearest-neighbor among the set

Rejection Criteria:

- If submitted enrollment images cluster too loosely.

---

# 4. Introduce Quality Gating Before Recognition

No recognition of garbage images

Reject:

- tiny faces
- blurred crops
- extreme angles
- partial occlusions
- low brightness

Gates:

- minimum face resolution
- Laplacian blur threshold
- pose angle threshold
- confidence threshold from detector

---

# 5. Add Unknown Persistence

If identity flapping:

```
unknown → unknown → weak match → identified
```

Instead:

- maintain identity state per track
- require confidence stability before switching identities

Example:

- once identified, require strong contradictory evidence to change
- once unknown, require repeated evidence to identify

---

# 6. Separate Recognition from Authorization

Recognition confidence and authorization confidence should be different.

Example:

- similarity = 0.72
- restricted zone

check for:

- “possible unauthorized individual”
- severity proportional to confidence

---

# 7. Human Verification Path

Example:

- low-confidence match
- operator receives snapshot
- operator confirms/rejects

Then:

- feedback can improve thresholds
- you collect evaluation data

Potentially:

- Online learning using operator’s confirmation.

---

# 8. Thresholds Should Be Per-Camera

Different cameras have:

- different lighting
- angles
- compression
- focal lengths

A universal threshold is usually suboptimal.

You may eventually want:

- global baseline threshold
- per-camera adjustment

---

# 9. Benchmark Early

Early testing on:

- real RTSP feeds
- poor lighting
- multiple people
- motion
- occlusion

---

# 10. Recognition Sampling

Recognizing every frame is not efficient.

Instead:

- detect every frame
- track every frame
- recognize periodically

Example:

- every 10th frame (adaptive)
- on track appearance
- when face quality improves

---

# 11. Liveness Detection (Anti-Spoofing)

- Current pipeline can be fooled by photo held up to camera.
- Add liveness check between quality gating and embedding extraction
    - MiniVAS / Silent-Face-Anti-Spoofing

---

# 12. Face Re-identification under occlusion

- When face is masked / partially occluded, confidence drops sharply.
- Rather than rejecting, add body re-id fallback
    - Use person’s full body appearance embedding to maintain identity continuity
    - Use BoT-SORT

---

# 13. Embedding Drift Handling

- A person enrolled 6 months ago may look diff today.
- Multi-embedding handled partially, but needed periodic re-enrollment prompt.

---

# 14. False Positive Rate Tracking Per Camera

- Track false positive and false negative rates per camera and surface them in dahboard.
- Operators can adjust thresholds accordingly.

---

# The Most Important Architectural Improvement

Change:

```
Detection → Embedding → Lookup → Identity
```

To:

```
Detection (every frame)
→ ByteTrack / BoT-SORT tracking (every frame)
→ Quality Filter (face size, blur, pose, confidence)
→ Liveness Check (anti-spoofing)
→ Embedding Extraction (sampled, adaptive rate)
→ Multi-Embedding Match (weighted nearest-neighbor)
→ Temporal Aggregation (weighted recent-biased window)
→ Confidence Scoring (per-camera threshold applied)
→ Identity State Machine (flap prevention)
→ Body Re-ID Fallback (if face occluded)
→ Authorization Logic (zone + enrollment + blocklist)
→ Severity Scaling (confidence-weighted alerts)
→ Human Verification Path (for uncertain matches)
→ Feedback Loop (operator confirmations update embeddings)
```