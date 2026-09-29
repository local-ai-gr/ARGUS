"""
Run the fine-tuned model on one full-res image and save a picture with the boxes.
Usage inside the container:
    python predict_and_draw.py images/DJI_20260511173316_0105_D.JPG 0.3
Output: preview_<image>_<thresh>.jpg and predictions_<image>.csv in /work
"""
import sys
from pathlib import Path
import cv2
import pandas as pd
from deepforest import main

img_path = sys.argv[1]
thresh = float(sys.argv[2]) if len(sys.argv) > 2 else 0.3

m = main.deepforest.load_from_checkpoint("argus_trees.ckpt")
m.config["score_thresh"] = thresh
boxes = m.predict_tile(img_path, patch_size=800, patch_overlap=0.1)
boxes.to_csv(f"predictions_{Path(img_path).stem}.csv", index=False)

img = cv2.imread(img_path)
for _, b in boxes.iterrows():
    cv2.rectangle(img, (int(b.xmin), int(b.ymin)), (int(b.xmax), int(b.ymax)), (0, 0, 255), 3)

# ground truth from the annotation CSV in green, if the image is annotated
gt = pd.read_csv("deepforest_annotations.csv")
gt = gt[gt.image_path == Path(img_path).name]
for _, b in gt.iterrows():
    cv2.rectangle(img, (int(b.xmin), int(b.ymin)), (int(b.xmax), int(b.ymax)), (0, 255, 0), 2)

out = f"preview_{Path(img_path).stem}_{thresh}.jpg"
cv2.imwrite(out, img, [cv2.IMWRITE_JPEG_QUALITY, 85])
print(f"{len(boxes)} predictions (red) vs {len(gt)} annotations (green) -> {out}")
