# ARGUS tree detector – fine-tuned DeepForest model

This package contains a tree-crown detector fine-tuned on ARGUS drone imagery
(DJI, 5280 × 3956 px, power-line corridors), together with a script that runs it on a
folder of images and writes the detections as CSV files and annotated pictures.

**Model:** `argus_trees.ckpt` (≈ 130 MB) – DeepForest RetinaNet (ResNet-50 FPN), fine-tuned for 10 epochs
on 52 annotated images (9,931 trees). Validation: box recall 0.73, box precision 0.54 (F1 0.62), versus
F1 0.12 for the pretrained DeepForest model on the same images.

## Contents

| File | Purpose |
|---|---|
| `argus_trees.ckpt` | The fine-tuned model weights |
| `detect_trees.py` | Batch inference script (folder of images → CSVs + overlays) |
| `requirements.txt` | Python dependencies (for the pip route) |
| `Dockerfile`, `docker-compose.yml` | Containerised alternative (for the Docker route) |
| `input/`, `output/` | Put images in `input/`, results appear in `output/` |

## Setup – choose one route

### Route A: plain Python (Windows, macOS or Linux; GPU optional)

Requires Python 3.10–3.12. From a terminal in this folder:

```bash
python -m venv .venv
# Windows:            .venv\Scripts\activate
# macOS / Linux:      source .venv/bin/activate
pip install -r requirements.txt
```

`pip install` downloads PyTorch (~2–3 GB). On a machine with an NVIDIA GPU, PyTorch's default
Linux/Windows wheel already includes CUDA support and the script will use the GPU automatically.
Without a GPU it runs on the CPU – fine for tens of images, roughly 1–3 minutes each.

Test:

```bash
python detect_trees.py --input input --output output
```

### Route B: Docker (Linux server with NVIDIA GPU)

Requires Docker and the NVIDIA Container Toolkit
(`docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi` must print the GPU).

```bash
docker compose build          # first time only, ~10 min
docker compose run --rm detector
```

Images are read from `./input` and results written to `./output`. Extra options are passed after
the service name, e.g. `docker compose run --rm detector --input /data/input --output /data/output --threshold 0.5`.
To run on a CPU-only machine delete the `deploy:` block from `docker-compose.yml`.

## Using it

Copy the drone photographs into `input/` (JPEG, PNG or TIFF; any size, the script tiles large
images automatically) and run:

```bash
python detect_trees.py --input input --output output --threshold 0.3
```

For each image you get `<name>_trees.csv` and `<name>_overlay.jpg`; `all_detections.csv` combines
every image and `summary.csv` lists the tree count per image.

CSV columns: `image_path, xmin, ymin, xmax, ymax, score, label`. Coordinates are pixels in the
original image (origin top-left, x to the right, y downwards); `score` is the model's confidence (0–1).

### Choosing the threshold

`--threshold` is the minimum confidence to keep a detection.

* 0.2–0.3 – favours completeness: finds nearly every tree, some extra boxes on shrubs. Recommended
  when missing a tree is worse than a false alarm (corridor risk screening, pre-annotation).
* 0.5 – favours precision: fewer boxes, almost all real trees; some small or partly hidden crowns are missed.

Overlays let you judge quickly: red boxes are detections, the count is printed in the corner.

## Using the model from your own Python code

```python
from deepforest import main
m = main.deepforest.load_from_checkpoint("argus_trees.ckpt")
m.config["score_thresh"] = 0.3
boxes = m.predict_tile("DJI_xxx.JPG", patch_size=800, patch_overlap=0.1)   # pandas DataFrame
```

`predict_tile` handles the tiling; use `m.predict_image(path=...)` for images ≤ ~1500 px.

## Notes and limitations

* Trained on nadir (straight-down) RGB drone photos of Slovenian forest at 5280 × 3956 px; images
  at very different scales, angles or seasons may need re-tuning.
* Single class ("Tree"): species and health are not predicted by this model.
* Rotated or overlapping crowns in dense canopy may be merged into one box.
* The model detects many small trees and shrubs that were not in the training annotations; at low
  thresholds expect boxes on woody vegetation of any size.
* Keep `argus_trees.ckpt` next to `detect_trees.py`; the script looks for it there.

## Troubleshooting

* `Model file not found` – `argus_trees.ckpt` is missing from the folder.
* `CUDA out of memory` – add `--patch-size 600`, or run on CPU.
* Very slow – check the first line printed by the script: it says whether it is on GPU or CPU.
* `ImportError: libGL` on Linux – `sudo apt install libgl1 libglib2.0-0`.
* Windows path problems – put the folder somewhere without spaces or non-ASCII characters.

## License and credit

Model and scripts: MIT, © 2026 Local AI (local-ai.gr). Built on DeepForest (MIT), Weinstein et al. 2020,
*Methods in Ecology and Evolution*, https://doi.org/10.1111/2041-210X.13472.
