"""
ARGUS tree detector – batch inference with the fine-tuned DeepForest model.

Detects tree crowns in full-resolution drone photographs (tested on DJI 5280x3956 JPEGs).

Usage:
    python detect_trees.py --input <folder or image> --output <folder> [--threshold 0.3] [--no-overlay]

Outputs, per image, in the output folder:
    <name>_trees.csv      one row per tree: image_path,xmin,ymin,xmax,ymax,score,label
    <name>_overlay.jpg    the image with detections drawn (unless --no-overlay)
plus all_detections.csv with every image combined, and summary.csv with tree counts per image.
"""
import argparse
import sys
import time
from pathlib import Path

import cv2
import pandas as pd
import torch

IMAGE_EXT = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
CKPT = Path(__file__).with_name("argus_trees.ckpt")


def load_model(threshold: float):
    from deepforest import main
    if not CKPT.exists():
        sys.exit(f"Model file not found: {CKPT}. Place argus_trees.ckpt next to this script.")
    m = main.deepforest.load_from_checkpoint(str(CKPT))
    m.config["score_thresh"] = threshold
    if torch.cuda.is_available():
        m.config["gpus"] = 1
        device = torch.cuda.get_device_name(0)
    else:
        m.config["gpus"] = 0
        device = "CPU (slower: ~1-3 min per 5280x3956 image)"
    m.create_trainer()
    print(f"Model loaded. Device: {device}. Score threshold: {threshold}")
    return m


def draw(image_path: Path, boxes: pd.DataFrame, out_path: Path):
    img = cv2.imread(str(image_path))
    for _, b in boxes.iterrows():
        cv2.rectangle(img, (int(b.xmin), int(b.ymin)), (int(b.xmax), int(b.ymax)), (0, 0, 255), 3)
    cv2.putText(img, f"{len(boxes)} trees", (40, 110), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 255), 8)
    cv2.imwrite(str(out_path), img, [cv2.IMWRITE_JPEG_QUALITY, 85])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True, help="image file or folder of images")
    ap.add_argument("--output", required=True, help="output folder (created if missing)")
    ap.add_argument("--threshold", type=float, default=0.3,
                    help="minimum confidence score to keep a detection (0-1). Lower = more trees, more false alarms. Default 0.3")
    ap.add_argument("--patch-size", type=int, default=800, help="tile size in pixels (model was trained at 800)")
    ap.add_argument("--no-overlay", action="store_true", help="skip writing the annotated JPEGs")
    args = ap.parse_args()

    src = Path(args.input)
    images = [src] if src.is_file() else sorted(p for p in src.iterdir() if p.suffix.lower() in IMAGE_EXT)
    if not images:
        sys.exit(f"No images found in {src}")
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    model = load_model(args.threshold)
    all_rows, summary = [], []
    for i, img_path in enumerate(images, 1):
        t0 = time.time()
        boxes = model.predict_tile(str(img_path), patch_size=args.patch_size, patch_overlap=0.1)
        if boxes is None or len(boxes) == 0:
            boxes = pd.DataFrame(columns=["xmin", "ymin", "xmax", "ymax", "label", "score", "image_path"])
        boxes = boxes[["image_path", "xmin", "ymin", "xmax", "ymax", "score", "label"]].copy()
        boxes["image_path"] = img_path.name
        boxes[["xmin", "ymin", "xmax", "ymax"]] = boxes[["xmin", "ymin", "xmax", "ymax"]].round().astype(int)
        boxes["score"] = boxes["score"].round(3)
        boxes.to_csv(out / f"{img_path.stem}_trees.csv", index=False)
        if not args.no_overlay:
            draw(img_path, boxes, out / f"{img_path.stem}_overlay.jpg")
        all_rows.append(boxes)
        summary.append({"image": img_path.name, "trees": len(boxes), "seconds": round(time.time() - t0, 1)})
        print(f"[{i}/{len(images)}] {img_path.name}: {len(boxes)} trees ({summary[-1]['seconds']} s)")

    pd.concat(all_rows, ignore_index=True).to_csv(out / "all_detections.csv", index=False)
    pd.DataFrame(summary).to_csv(out / "summary.csv", index=False)
    print(f"Done. Results in {out.resolve()}")


if __name__ == "__main__":
    main()
