# LANESIGHT — Hybrid Lane Detection System

**Core vision engine + Streamlit visualization UI (thin client).**

LANESIGHT is a modular lane perception project that runs multiple lane-detection approaches on the **same input frame** to make **comparison, debugging, and future fusion** straightforward.

---

## Repository branches

| Branch | Purpose |
|---|---|
| `main` | Core lane detection engine (models, inference, mask generation, fusion-ready outputs) |
| `lanesight_ui` | Streamlit UI for visualization & video preview (no vision logic) |

> The UI intentionally stays thin: it **does not reimplement** the vision pipeline. It only calls the engine.

---

## What the engine produces

For each frame, the engine returns **four image-space binary masks** (one per method):

1. **Hybrid ViT–UNet (semantic segmentation)**  
   Pixel-wise lane probability → thresholded binary mask  
   Robust to lighting changes and partial occlusions

2. **UFLD (Ultra-Fast Lane Detection)**  
   Lane polylines → rasterized lane mask  
   Very fast and stable on highways

3. **Traditional Computer Vision**  
   Canny + ROI + Hough → polygon lane mask  
   Deterministic baseline

4. **YOLO Object Detection**  
   Vehicle bounding boxes → vehicle mask  
   Intended for masking/rejecting lane hypotheses that intersect obstacles

---

## Vision engine architecture (high level)

```text
Input frame
   |
   +--> ViT–Hybrid UNet  --> lane probability --> binary lane mask
   |
   +--> UFLD ------------> lane polylines ----> lane mask
   |
   +--> Traditional CV --> Hough/ROI ---------> polygon mask
   |
   +--> YOLO ------------> vehicle boxes -----> vehicle mask
```

**Unified interface:** one call → returns the 4 masks for the current frame.

---

## Streamlit UI (`lanesight_ui` branch)

### Purpose

Interactive visualization layer to:

- preview results in (near) real time
- compare all approaches side-by-side
- process uploaded dashcam videos without modifying the engine

**Important:** the UI contains **no vision logic**.  
All inference runs inside: `LANESIGHT/video/infer_video.py`.

### Features

**Input**
- Upload video: `mp4`, `avi`, `mov`, `mkv`
- No dataset formatting required

**Toggles**
- Enable/disable each module independently:
  - ViT segmentation
  - UFLD
  - Traditional processing
  - YOLO vehicles

**Preview**
- 4 independent previews updated every *N* frames:
  1) ViT–Hybrid  
  2) YOLO  
  3) UFLD  
  4) Traditional polygon

**Output**
Final 2×2 split video:

```text
+------------------+------------------+
| 1) ViT–Hybrid    | 2) YOLO          |
+------------------+------------------+
| 3) UFLD          | 4) Polygon       |
+------------------+------------------+
```

**Export**
- Download processed video
- Optional server-side export

---

## UI architecture (thin by design)

```text
Streamlit UI
   |
   |-- video upload / sliders / checkboxes
   |
   |-- VisionMaskGenerator (cached)
           |
           +--> LANESIGHT engine
```

Key design choices:
- Models are loaded **once per session**
- UI only handles:
  - video I/O
  - frame sampling
  - visualization
- Engine can evolve without rewriting the UI

---

## Quick start (UI)

```bash
# Switch to UI branch
git checkout lanesight_ui

# Install dependencies
pip install -r requirements.txt

# Launch Streamlit
streamlit run app.py
```

---

## Intended use cases

- Academic project (PFE / research)
- Method comparison & ablation studies
- Demonstration / oral defense
- Debugging lane detection failure modes
- Teaching hybrid perception pipelines

---

## Design philosophy

- **Separation of concerns:** Engine ≠ UI  
- **Comparability over chasing a single score:** multiple methods > single black box  
- **Explainability-first:** visual, interpretable outputs

---

## Roadmap / future extensions

- Temporal fusion across frames
- Confidence-based lane ranking
- Real-time deployment (TensorRT)
- Multi-lane instance separation
- Online calibration