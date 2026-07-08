# Smart-Vision-Shoe
 A browser-based obstacle scanner for a shoe-mounted "vision shoe" concept. Uses your device's camera and real-time computer vision to detect obstacles in your path, classify what they are, and warn you by voice — left, center, or right — before you reach them.
# PATHSCAN — Vision Shoe Obstacle Scanner

A single-page, browser-based obstacle detection system that turns any camera-equipped device into a real-time path hazard scanner. Built around a "vision shoe" concept — a low, forward/downward-mounted camera that watches the ground ahead of a walking user and calls out obstacles by zone, distance, and identity.

Runs 100% client-side. No backend, no video upload, no build step — one HTML file.

---

## 1. Overview

PATHSCAN answers three questions about the path ahead, continuously, in real time:

| Question | How it's answered |
|---|---|
| **Where** is the obstacle? | LEFT / CENTER / RIGHT zone |
| **How far** is it? | Estimated distance in meters, from a calibrated ground-plane model |
| **What** is it? | Object class label, from a neural object detector |

These are fused into a single spoken alert (`"Backpack ahead, left, 1.4 meters"`) plus a live visual HUD and a scrollable alert log.

---

## 2. System Architecture

The app runs **two independent perception pipelines in parallel**, on every frame, and fuses their outputs at the zone level. This dual-pipeline design is deliberate: each pipeline covers the other's blind spot.

```
                         ┌─────────────────────────┐
                         │   getUserMedia() stream   │
                         │   (rear camera, mirrored  │
                         │    for on-screen display) │
                         └────────────┬─────────────┘
                                      │
                    ┌─────────────────┴─────────────────┐
                    │                                     │
                    ▼                                     ▼
    ┌───────────────────────────────┐   ┌───────────────────────────────┐
    │  PIPELINE A — Geometric        │   │  PIPELINE B — Neural           │
    │  Edge / Distance Estimator     │   │  Object Classifier             │
    │  (runs every rendered frame,   │   │  (runs every 600ms)            │
    │   ~30–60 Hz)                   │   │                                 │
    │                                 │   │                                 │
    │  1. Downscale frame to 160×120 │   │  1. Full-res frame → COCO-SSD  │
    │  2. Sobel-style gradient pass  │   │     (TensorFlow.js, MobileNet   │
    │     over bottom 55% of frame   │   │     backbone)                   │
    │  3. Per-zone edge accumulation │   │  2. Confidence filter           │
    │  4. Nearest edge row per zone  │   │     (0.6 for "person", 0.45     │
    │  5. Row → meters via pinhole   │   │     for everything else)        │
    │     ground-plane projection    │   │  3. Cross-class NMS (removes    │
    │  6. Threshold → CLEAR/NEAR/    │   │     duplicate overlapping boxes)│
    │     CLOSE per zone             │   │  4. Self/operator filter        │
    │                                 │   │     (drops boxes covering       │
    │  Output: distance + severity   │   │     >55% of frame area or       │
    │  per zone, always available,   │   │     >85% of frame height)       │
    │  works on ANY surface texture  │   │  5. Zone assignment by bbox     │
    │  (curbs, steps, poles, walls — │   │     horizontal span (not just   │
    │  things with no COCO class)    │   │     center point)               │
    │                                 │   │                                 │
    │                                 │   │  Output: object class label     │
    │                                 │   │  per zone (nearest by distance, │
    │                                 │   │  or by confidence if off-       │
    │                                 │   │  horizon)                       │
    └────────────────┬───────────────┘   └────────────────┬────────────────┘
                      │                                     │
                      └──────────────────┬──────────────────┘
                                          ▼
                         ┌───────────────────────────────┐
                         │        FUSION LAYER             │
                         │  Per zone: distance/severity     │
                         │  from Pipeline A + label from    │
                         │  Pipeline B (if any) = final      │
                         │  zone state                       │
                         └────────────────┬──────────────────┘
                                          ▼
                    ┌─────────────────────────────────────┐
                    │            OUTPUT LAYER                │
                    │  • Canvas overlay (zone tint,           │
                    │    distance label, object label,        │
                    │    live detection boxes)                 │
                    │  • Web Speech API (cooldown-gated         │
                    │    spoken alert on worst zone)            │
                    │  • Alert log (timestamped, color-coded)   │
                    └─────────────────────────────────────┘
```

**Why two pipelines instead of one?** A neural object detector only knows ~80 COCO classes — it has no idea what a curb, pothole, step, or trailing cable is. The geometric edge detector doesn't know *what* anything is, but it reacts to *any* high-contrast surface change, so it catches real-world hazards the object model would silently miss. Neither pipeline is trusted alone for the final alert; the geometric pipeline decides *whether and how urgently* to alert, and the object model decides *what to call the thing* if it recognizes it.

---

## 3. AI / ML Pipeline Detail

### 3.1 Pipeline A — Geometric edge & ground-plane distance estimator

This is a classical computer vision pipeline (no neural network), chosen deliberately because it needs to run every animation frame with near-zero latency and works regardless of lighting/object type.

**Step 1 — Downscale & mirror**
The live video is drawn into an offscreen `160×120` canvas, horizontally flipped to match the mirrored on-screen preview. Downscaling keeps the per-pixel gradient pass cheap enough to run at full frame rate.

**Step 2 — Region of interest**
Only the bottom 55% of the frame is analyzed (`roiTop = 0.45 × height`). This is the region where the ground plane and near-field obstacles actually appear; the sky/upper background is discarded.

**Step 3 — Edge energy (simple Sobel-like gradient)**
For every pixel in the ROI:
```
gray = (R + G + B) / 3
gx = |gray(x+1) − gray(x−1)|
gy = |gray(y+1) − gray(y−1)|
edge_energy = gx + gy
```
Pixels with `edge_energy > 40` are treated as an "edge" (a likely object/ground boundary). Each edge pixel is weighted more heavily the closer it is to the bottom of the frame (`proximityWeight = 1 + (y / roiHeight) × 1.5`), since rows near the bottom correspond to points physically closer to the camera.

**Step 4 — Per-zone aggregation**
The ROI is split into three equal-width columns (LEFT / CENTER / RIGHT). For each column, the algorithm tracks the lowest (= nearest) row containing a qualifying edge, and requires at least 6 edge pixels in that zone before trusting the reading (`MIN_EDGE_PIXELS`), to reject single-pixel noise.

**Step 5 — Ground-plane distance projection**
The nearest edge row per zone is converted into a real-world distance using a pinhole/ground-plane model parameterized by three user-calibrated values:

```
v        = (row − frameHeight/2) / (frameHeight/2)     // −1 (top) to +1 (bottom)
rayAngle = v × (verticalFOV / 2)
φ        = mountTiltAngle + rayAngle                     // ray angle below horizontal
distance = mountHeight / tan(φ)                           // meters
```
If `φ ≤ 0.02` rad, the ray points at or above the horizon and has no ground intersection → distance is `Infinity` (nothing detected in that zone).

**Step 6 — Zone state classification**
`SENSITIVITY` sets the meter range for state transitions:
```
nearThreshold  = 0.5 + (sensitivity / 100) × 2.5   // ≈0.75 m .. 3.0 m
closeThreshold = nearThreshold × 0.4                // inner 40% of that range
```
Each zone becomes `CLEAR`, `NEAR`, or `CLOSE` accordingly, and the worst zone (by severity, then distance) becomes the candidate for the next spoken alert.

### 3.2 Pipeline B — Neural object classifier

**Model:** [COCO-SSD](https://github.com/tensorflow/tfjs-models/tree/master/coco-ssd) (Single Shot MultiBox Detector, MobileNet backbone) via **TensorFlow.js**, loaded once on page load and cached in-browser after first download.

**Inference cadence:** every 600ms (`setInterval(detectObjects, 600)`), decoupled from the render loop — object identity doesn't need to update at 60fps, and this keeps the neural pass from starving the main thread.

**Post-processing pipeline (per inference pass):**

1. **Confidence filtering** — class-dependent threshold. `person` requires ≥0.6 confidence (this class produces the most false positives against clothing, shadows, and reflections); all other classes use ≥0.45.
2. **Cross-class Non-Max Suppression** — COCO-SSD only suppresses duplicate boxes *within* the same class. This pipeline adds a second NMS pass across *all* classes (IoU > 0.5 → keep only the highest-confidence box) so the same physical object doesn't produce multiple conflicting labels.
3. **Self/operator filtering** — any box covering more than 55% of the frame area or 85% of the frame height is excluded and logged as a warning rather than treated as an obstacle. A shoe/waist-mounted camera aimed at the path ahead should never see a person filling the frame — when it happens, it's almost always the camera wearer's own body, not a genuine near-field obstacle.
4. **Zone assignment by span, not centroid** — each surviving box is mapped to *every* zone column its bounding box actually overlaps (not just the column under its center point), so wide objects near a zone boundary aren't mis-assigned or dropped.
5. **Per-zone winner selection** — within each zone, the object with the lowest effective distance wins:
   ```
   effectiveDistance = groundPlaneDistance(bbox.bottom)   // if bbox intersects the ground plane
                      = 1000 − confidenceScore              // fallback ranking if off-horizon
   ```

### 3.3 Fusion & alerting

- The **geometric pipeline decides severity and distance** for each zone (CLEAR / NEAR / CLOSE + meters).
- The **object pipeline decides the label** to speak/display for that zone, if any class was recognized there.
- If no object was recognized, the alert falls back to the generic term `"Obstacle"`.
- Alerts are cooldown-gated (`ALERT COOLDOWN` slider, 1–5s) so the system doesn't speak continuously.
- Every alert is spoken via the **Web Speech API** (`SpeechSynthesisUtterance`) and written to the on-screen log, color-coded by severity.

---

## 4. Live Visual Feedback (HUD)

The canvas overlay draws, every frame:
- A dashed horizon line marking the top of the analysis ROI
- Tinted zone rectangles (green/amber/red) matching current severity
- Distance and object-label text per zone
- **Raw detection boxes** with class + confidence (e.g. `chair 82%`) directly on the video feed — this makes any misclassification visually debuggable in real time instead of being a black box.

---

## 5. Controls & Calibration

| Control | Effect |
|---|---|
| **Sensitivity** | Sets the NEAR/CLOSE distance thresholds (higher = alerts trigger further away) |
| **Alert Cooldown** | Minimum time between spoken alerts (1–5s) |
| **Mount Height** | Camera height off the ground, in cm — required for the ground-plane distance formula |
| **Mount Tilt** | Downward camera angle from horizontal, in degrees |
| **Camera FOV** | Vertical field of view of the camera, in degrees |
| **Switch Camera** | Cycles through enumerated camera devices; auto-prefers a rear/back-labeled lens on load |
| **Voice toggle** | Enables/disables spoken alerts (visual HUD and log continue regardless) |

Accurate distance estimates depend on Mount Height / Tilt / FOV being set to match how the camera is actually worn — this is a calibrated geometric estimate, not a depth sensor.

---

## 6. Tech Stack

- **Vanilla JavaScript** (no framework, no build step)
- **HTML5 Canvas** — offscreen analysis buffer + live overlay rendering
- **WebRTC `getUserMedia()` / `enumerateDevices()`** — camera access and device selection
- **TensorFlow.js 4.20** + **COCO-SSD 2.2.2** (loaded via CDN, cached after first load)
- **Web Speech API** (`SpeechSynthesisUtterance`) — text-to-speech alerts
- Single self-contained `.html` file — CSS and JS inline, no external assets besides the two CDN model scripts

---

## 7. Project Structure

```
pathscan.html      ← everything: markup, styles, and application logic
```

There is intentionally no build pipeline, package.json, or server component. The entire app is one file that can be opened, hosted, or shared as-is.

---

## 8. Running It

Camera access (`getUserMedia`) requires a **secure context** — either `https://` or `http://localhost`. It will **not** work opened directly as a `file://` path in most browsers.

**Local testing:**
```bash
# any static server works, e.g.:
python3 -m http.server 8000
# then open http://localhost:8000/pathscan.html
```

**Deploying for real-world / mobile use:**
- Drag-and-drop deploy: [Netlify Drop](https://app.netlify.com/drop), [Vercel](https://vercel.com)
- Static hosting: GitHub Pages
- On mobile, use **Add to Home Screen** (iOS Safari share menu / Android Chrome menu) — the page includes PWA-style meta tags so it launches full-screen like a native app.

---

## 9. Known Limitations

- Distance is a **geometric estimate**, not a measured depth — accuracy depends entirely on correct Mount Height/Tilt/FOV calibration.
- Object labels are limited to COCO-SSD's ~80 trained classes; unlabeled hazards (curbs, steps, poles, uneven ground, cables) are only caught by the geometric edge pipeline, with no name attached.
- COCO-SSD inference runs client-side; performance depends on device GPU/CPU — older phones may see the 600ms inference cycle run slower.
- `facingMode: 'environment'` is a hint, not a guarantee — device camera enumeration and manual switching are provided as a fallback for devices that don't honor it.

## 10. Possible Future Improvements

- Replace the classical edge detector with a lightweight monocular depth model for more robust distance estimates on non-uniform ground
- Add haptic (vibration) alerts as a silent alternative/complement to voice
- Persist calibration settings (height/tilt/FOV) across sessions
- Multi-object-per-zone reporting instead of single nearest-object winner
- Offline model caching via Service Worker for true no-connectivity operation after first load

---

## License

Prototype / demonstration project. No license specified — add one appropriate to your intended use (MIT is a common default for open prototypes).
