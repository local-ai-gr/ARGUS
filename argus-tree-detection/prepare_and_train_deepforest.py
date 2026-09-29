"""
Tile the full-resolution DJI images with the Label Studio-derived annotations
and fine-tune DeepForest's pretrained tree model.

Usage:
    pip install deepforest
    python prepare_and_train_deepforest.py

Expected layout before running:
    images/                      <- the 52 original DJI_*.JPG files (5280x3956)
    deepforest_annotations.csv   <- exported from Label Studio (full-res pixel coords)
"""
from pathlib import Path
import pandas as pd
from deepforest import main, preprocess, get_data  # noqa: F401

IMAGE_DIR = Path("images")
ANNOTATIONS = Path("deepforest_annotations.csv")
TILE_DIR = Path("tiles")
PATCH_SIZE = 800         # px; 800 keeps memory modest on a 6 GB GPU (trees ~100-200 px)
PATCH_OVERLAP = 0.10
VAL_FRACTION = 0.2       # hold out ~20% of the *images* for validation
EPOCHS = 10

TILE_DIR.mkdir(exist_ok=True)
df = pd.read_csv(ANNOTATIONS)

# sanity: every image in the CSV must exist
missing = [p for p in df.image_path.unique() if not (IMAGE_DIR / p).exists()]
if missing:
    raise SystemExit(f"{len(missing)} images referenced in the CSV are missing from {IMAGE_DIR}: {missing[:5]}")

# 1) tile each raster, writing crops + a per-image CSV into TILE_DIR
tiled = []
for img in sorted(df.image_path.unique()):
    part = preprocess.split_raster(
        annotations_file=df[df.image_path == img],
        path_to_raster=str(IMAGE_DIR / img),
        root_dir=str(IMAGE_DIR),
        patch_size=PATCH_SIZE,
        patch_overlap=PATCH_OVERLAP,
        save_dir=str(TILE_DIR),      # older DeepForest versions call this base_dir
        allow_empty=False,
    )
    tiled.append(part)
tiles = pd.concat(tiled, ignore_index=True)

# 2) split by source image so tiles of one photo never leak across train/val
images = sorted(df.image_path.unique())
n_val = max(1, int(len(images) * VAL_FRACTION))
val_images = set(images[-n_val:])
tiles["src"] = tiles.image_path.str.extract(r"^(DJI_\d+_\d+_D)")[0] + ".JPG"
train = tiles[~tiles.src.isin(val_images)].drop(columns="src")
val = tiles[tiles.src.isin(val_images)].drop(columns="src")
train.to_csv(TILE_DIR / "train.csv", index=False)
val.to_csv(TILE_DIR / "val.csv", index=False)
print(f"{len(train)} train boxes, {len(val)} val boxes, {tiles.image_path.nunique()} tiles")

# 3) fine-tune from the pretrained tree model
m = main.deepforest()
m.load_model("weecology/deepforest-tree")   # on DeepForest < 1.4 use m.use_release()
m.config["train"]["csv_file"] = str(TILE_DIR / "train.csv")
m.config["train"]["root_dir"] = str(TILE_DIR)
m.config["validation"]["csv_file"] = str(TILE_DIR / "val.csv")
m.config["validation"]["root_dir"] = str(TILE_DIR)
m.config["train"]["epochs"] = EPOCHS
m.config["gpus"] = 1          # set 0 for CPU
m.config["batch_size"] = 2          # 6 GB VRAM; drop to 1 if you see CUDA out of memory
m.config["workers"] = 4
m.create_trainer()
m.trainer.fit(m)
m.trainer.save_checkpoint("argus_trees.ckpt")
print(m.evaluate(str(TILE_DIR / "val.csv"), root_dir=str(TILE_DIR)))
