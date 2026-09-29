# ARGUS tree detection – fine-tuning DeepForest on drone imagery

Code and reproducible GPU environment for fine-tuning the open-source
[DeepForest](https://github.com/weecology/DeepForest) tree-crown detector on
UAV photographs annotated in [Label Studio](https://labelstud.io/).

Developed for the ARGUS project by [Local AI](https://local-ai.gr), Kalamata, Greece.

On 52 annotated DJI images (5280 × 3956 px, 9,931 tree boxes) ten epochs of
fine-tuning raised validation F1 from **0.12** (pretrained model) to **0.62**:

| Model | Box recall | Box precision | F1 |
|---|---|---|---|
| Pretrained `weecology/deepforest-tree` | 0.285 | 0.077 | 0.121 |
| Fine-tuned on ARGUS (10 epochs) | 0.730 | 0.537 | 0.619 |

## What is in the repository

| File | Purpose |
|---|---|
| `export_labelstudio.py` | Pulls annotations from the Label Studio API and writes the DeepForest CSV (`image_path,xmin,ymin,xmax,ymax,label`), converting percent coordinates to pixels and rotated boxes to axis-aligned ones |
| `prepare_and_train_deepforest.py` | Tiles the full-resolution images, splits train/validation **by image**, fine-tunes from the pretrained weights, saves `argus_trees.ckpt` and prints validation metrics |
| `compare_baseline.py` | Evaluates pretrained vs fine-tuned model on the same validation tiles and draws a side-by-side picture |
| `predict_and_draw.py` | Runs the fine-tuned model on one full-resolution image and draws predictions (red) over annotations (green) |
| `Dockerfile`, `docker-compose.yml`, `.env.example`, `run.sh` | Reproducible GPU training environment (PyTorch 2.4 / CUDA 12.4 / DeepForest) |
| `sample_annotations.csv` | Three example rows showing the expected CSV format |

Data, tiles, checkpoints and `.env` are deliberately excluded by `.gitignore`.

## Requirements

* A Linux machine with an NVIDIA GPU (tested on a GTX 1660 Ti, 6 GB), driver ≥ 550
* Docker with the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
  – check with `docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi`
* A Label Studio project using `RectangleLabels` (one label, e.g. `Tree`; an optional per-region `Choices` for species is supported)
* The original full-resolution images on the training machine (Label Studio may only hold downscaled previews)
* Python 3 with `pandas` and `requests` for the export script (everything else runs inside the container)

## Step by step

### 1. Clone and configure

```bash
git clone https://github.com/<your-user>/argus-tree-detection.git
cd argus-tree-detection
cp .env.example .env          # set ORIG_DIR to the folder with the original images
```

### 2. Export the annotations from Label Studio

Get an API token from Label Studio (*Account & Settings → Access Token*), then:

```bash
export LABEL_STUDIO_URL=http://<label-studio-host>:8080
export LABEL_STUDIO_TOKEN=<token>
pip install pandas requests
python export_labelstudio.py --project <project-id> --out deepforest_annotations.csv
```

Add `--label species` to use the per-box species choice as the class instead of a single `Tree` class.
The `image_path` column contains the original file names, so every file listed must exist in `ORIG_DIR`.

### 3. Train

```bash
bash run.sh                        # builds the image (first time ~10 min), checks the GPU, starts training
docker logs -f argus-deepforest    # follow progress (Ctrl-C only stops watching)
```

Training writes `tiles/` (with `train.csv` / `val.csv`), `argus_trees.ckpt` and the validation
metrics at the end of the log. Ten epochs on 52 images take about an hour on a GTX 1660 Ti.

Tunable constants at the top of `prepare_and_train_deepforest.py`: `PATCH_SIZE` (800 px; use
larger for bigger crowns), `EPOCHS` (10; 30 once labels are complete), `VAL_FRACTION` (0.2),
and `batch_size` (2 for 6 GB of GPU memory).

### 4. Evaluate against the pretrained baseline

```bash
docker compose run --rm deepforest python compare_baseline.py [images/<validation-image>.JPG]
```

Prints recall / precision / F1 for both models, writes `comparison.csv` and
`compare_<image>.jpg` (baseline left, fine-tuned right, annotations in green).

### 5. Run inference on an image

```bash
docker compose run --rm deepforest python predict_and_draw.py images/<file>.JPG 0.3
```

The last argument is the score threshold. Output: `preview_<file>_<thr>.jpg` and `predictions_<file>.csv`.

In your own code:

```python
from deepforest import main
m = main.deepforest.load_from_checkpoint("argus_trees.ckpt")
m.config["score_thresh"] = 0.3
boxes = m.predict_tile("DJI_xxx.JPG", patch_size=800, patch_overlap=0.1)   # pandas DataFrame
```

## How the container is organised

* `Dockerfile` starts from `pytorch/pytorch:2.4.0-cuda12.4-cudnn9-runtime` and installs DeepForest.
* `docker-compose.yml` gives the container one GPU, 4 GB of shared memory for data-loader workers, and three mounts:
  the repository as `/work` (read-write – all outputs land here on the host), `ORIG_DIR` as `/work/images` (read-only),
  and `./hf_cache` as the model cache so pretrained weights are downloaded once.
* `run.sh` chains build → GPU check → detached training. Training survives SSH disconnects.
* `docker compose run --rm deepforest <command>` runs any one-off command in a fresh container from the same image.

## Notes and known issues

* Label Studio stores boxes as percentages; the export script uses `original_width/height` from each
  annotation, so it is correct even if Label Studio itself serves downscaled images.
* Boxes drawn with a rotation are replaced by their axis-aligned bounding box.
* `deepforest.evaluate()` ignores `score_thresh`; for a real precision–recall curve filter predictions by score yourself.
* Precision is under-estimated when annotations are incomplete: the fine-tuned model detects trees the annotators
  missed. A human-in-the-loop round (model predictions as Label Studio pre-annotations, then retrain) is the
  recommended next step.

## Citation

DeepForest: Weinstein, B.G. et al. (2020). *DeepForest: A Python package for RGB deep learning tree crown delineation.*
Methods in Ecology and Evolution, 11, 1743–1751. https://doi.org/10.1111/2041-210X.13472

## License

MIT – see [LICENSE](LICENSE). DeepForest is also MIT-licensed.
