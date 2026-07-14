# SEM → SAM → Parametric Fit → Spatial Compare

Self-contained copy of the SEM analysis pipeline.

## Contents

| File | What it is |
|------|------------|
| `sem_pipeline.py` | The whole pipeline in one heavily-commented module (all 4 stages + orchestrator + CLI). |
| `example.ipynb` | Worked example that runs each stage on a real SEM photo with visualizations. |
| `data/SEM_results.tif` | Example SEM image (1024×768, 300 nm scale bar). |
| `data/my_layout.gds` | Example existing design layout, for the Stage-4 spatial comparison. |
| `sam_vit_b_01ec64.pth` | Symlink to the SAM checkpoint (358 MB) in `../old/`. |
| `requirements.txt` | Python dependencies. |

## The 4 stages

1. **SEM → scale** — OCR the scale-bar label + measure its pixel length → `nm/px`.
2. **SEM → cutout (SAM)** — Segment Anything turns click points into a mask → contour in nm.
3. **Optimize fit** — fit the parametric waveguide+taper: search shape params (outer DE),
   and for each candidate find the best translation **and rotation/angle** for overlap (inner DE).
4. **Compare vs existing** — spatially register the fitted outline onto an existing design
   GDS by maximizing overlap area; write a combined GDS.

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
- The built-in shape template (`generate_waveguide_shape`) is a straight waveguide + one-sided
  taper. Swap it for your device geometry to fit other structures.
- The full outer optimization runs a placement search per candidate, so it takes minutes; the
  notebook demonstrates a fast single inner fit and gates the full search behind a flag.
