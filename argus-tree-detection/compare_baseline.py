"""
Compare the pretrained DeepForest tree model with the fine-tuned checkpoint on
the same validation set (tiles/val.csv) and on one full-res image.

Usage inside the container:
    python compare_baseline.py [images/DJI_20260511173316_0105_D.JPG]
Outputs (in /work):
    comparison.csv                  metrics per model and threshold
    compare_<image>.jpg             baseline (left) vs fine-tuned (right)
"""
import sys, warnings, logging
from pathlib import Path
import cv2, numpy as np, pandas as pd
from deepforest import main

warnings.filterwarnings("ignore")
logging.getLogger("lightning.pytorch").setLevel(logging.ERROR)

VAL_CSV, TILE_DIR = "tiles/val.csv", "tiles"
img_path = sys.argv[1] if len(sys.argv) > 1 else "images/DJI_20260511173316_0105_D.JPG"
THRESHOLDS = [0.1, 0.3, 0.5]

def load_baseline():
    m = main.deepforest(); m.load_model("weecology/deepforest-tree"); return m

def load_finetuned():
    return main.deepforest.load_from_checkpoint("argus_trees.ckpt")

rows = []
previews = {}
for name, loader in [("baseline", load_baseline), ("fine-tuned", load_finetuned)]:
    m = loader()
    for t in THRESHOLDS:
        m.config["score_thresh"] = t
        r = m.evaluate(VAL_CSV, root_dir=TILE_DIR)
        rec, prec = float(r["box_recall"]), float(r["box_precision"])
        f1 = 2 * rec * prec / (rec + prec) if rec + prec else 0.0
        rows.append({"model": name, "score_thresh": t, "box_recall": round(rec, 3),
                     "box_precision": round(prec, 3), "f1": round(f1, 3)})
        print(f"{name:10s} thresh={t:.1f}  recall={rec:.3f}  precision={prec:.3f}  f1={f1:.3f}", flush=True)
    m.config["score_thresh"] = 0.3
    previews[name] = m.predict_tile(img_path, patch_size=800, patch_overlap=0.1)

df = pd.DataFrame(rows)
df.to_csv("comparison.csv", index=False)
print("\n" + df.to_string(index=False))

# side-by-side picture: green = annotations, red = predictions (thresh 0.3)
gt = pd.read_csv("deepforest_annotations.csv")
gt = gt[gt.image_path == Path(img_path).name]
base = cv2.imread(img_path)
panels = []
for name in ["baseline", "fine-tuned"]:
    img = base.copy()
    for _, b in gt.iterrows():
        cv2.rectangle(img, (int(b.xmin), int(b.ymin)), (int(b.xmax), int(b.ymax)), (0, 255, 0), 2)
    for _, b in previews[name].iterrows():
        cv2.rectangle(img, (int(b.xmin), int(b.ymin)), (int(b.xmax), int(b.ymax)), (0, 0, 255), 3)
    cv2.putText(img, f"{name}: {len(previews[name])} boxes (green = {len(gt)} annotations)",
                (40, 120), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 255), 8)
    panels.append(img)
out = f"compare_{Path(img_path).stem}.jpg"
cv2.imwrite(out, np.hstack(panels), [cv2.IMWRITE_JPEG_QUALITY, 80])
print(f"\nSide-by-side written to {out}")
