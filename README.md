# SEM → SAM → Parametric Fit → Spatial Compare

Self-contained copy of the SEM analysis pipeline.

## Contents

| File | What it is |
|------|------------|
| `sem_pipeline.py` | The whole pipeline in one heavily-commented module (all 5 stages + orchestrator + CLI). |
| `example.ipynb` | Worked example that runs each stage on a real SEM photo with visualizations. |
| `data/SEM_results.tif` | Example SEM image (1024×768, 300 nm scale bar). |
| `data/my_layout.gds` | Example existing design layout, for the Stage-4 spatial comparison. |
| `data/distortion_X_raw.npy`, `data/distortion_Y.npy` | Example training data for Stage 5 — 27 measured devices' (x,y) positions and per-parameter fitted/nominal factors. |
| `sam_vit_b_01ec64.pth` | Symlink to the SAM checkpoint (358 MB) in `../old/`. |
| `requirements.txt` | Python dependencies. |

## The 5 stages

1. **SEM → scale** — OCR the scale-bar label + measure its pixel length → `nm/px`.
2. **SEM → cutout (SAM)** — Segment Anything turns click points into a mask → contour in nm.
3. **Optimize fit** — fit the parametric waveguide+taper: search shape params (outer DE),
   and for each candidate find the best translation **and rotation/angle** for overlap (inner DE).
4. **Compare vs existing** — spatially register the fitted outline onto an existing design
   GDS by maximizing overlap area; write a combined GDS.
5. **Distortion prediction** — fab distortion (etch loading, proximity effects, dose
   variation, ...) varies smoothly across a chip. Given many previously-measured devices'
   `(x,y)` position + Stage-3 `fitted/nominal` ratio, fit a Gaussian Process
   (`SpatialDistortionModel`) that predicts the expected distortion factor at a new `(x,y)`
   — e.g. to pre-bias a design before fabricating it there.

## Setup

```bash
pip install -r requirements.txt          # plus the system `tesseract` binary
# If the SAM symlink is broken, download the checkpoint here:
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
```

## Run

Notebook: open `example.ipynb` and run top to bottom.

One call (headless):
```python
from sem_pipeline import run_full_pipeline
r = run_full_pipeline("data/SEM_results.tif",
                      nominal_params={"W1":300,"W2":400,"W3":500,"W4":600,"L1":200,"L2":300},
                      sam_points=[[400,190],[900,190]],   # None -> interactive clicking
                      design_gds="data/my_layout.gds",     # None -> skip Stage 4
                      out_prefix="results/run1")
print(r["complete_params"], r["best_hausdorff"])
```

CLI:
```bash
python sem_pipeline.py data/SEM_results.tif --points "400,190;900,190" --design-gds data/my_layout.gds
```

## Notes

- Default scale-bar ROIs are tuned for `SEM_results.tif`. For other magnifications pass new
  `text_box`/`scale_box` rectangles, or supply a known `nm_per_px` and skip Stage 1.
- Two parametric models ship in `sem_pipeline.py`, selected via the `shape_fn` argument to
  `fit_waveguide_to_contours` / `optimize_shape_params`:
  - `generate_waveguide_shape` — straight waveguide + one-sided taper (`[W1..L2, length_waveguide]`).
  - `generate_diamond_device` — symmetric diamond taper + guide with a step edge
    (`[w_tip, w_max, w_neck, w_guide, L_up, L_down, L_guide]`): widens tip→peak, narrows to a neck,
    then a vertical step drops to the guide width. Use `estimate_diamond_nominal(contour)` to auto-seed
    params from the contour and `diamond_bounds(nominal)` for the search box.
  Add your own generator (any `params -> (N,2) closed ring`) and pass it as `shape_fn` to fit other shapes.
- The full outer optimization runs a placement search per candidate, so it takes minutes; the
  notebook demonstrates a fast single inner fit and gates the full search behind a flag.
- Stage 5 is a *separate question* from Stage 3: Stage 3 tells you how one device fabricated
  relative to its own nominal params; Stage 5 predicts what distortion to *expect* at a
  location you haven't measured yet, based on a scatter of previously-measured devices. Build
  training rows with `compute_distortion_factors(nominal, fitted)` as you process more devices.
