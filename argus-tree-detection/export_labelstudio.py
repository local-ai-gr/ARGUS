"""
Export Label Studio RectangleLabels annotations to the DeepForest CSV format.

    image_path,xmin,ymin,xmax,ymax,label

Label Studio stores boxes as percentages of the image size (plus an optional
rotation); DeepForest wants absolute pixel coordinates of axis-aligned boxes.
This script converts them, replaces rotated boxes by their axis-aligned
bounding box, clips to the image bounds and strips Label Studio's upload
prefix from the file names so they match the original image files.

Usage:
    export LABEL_STUDIO_URL=http://10.10.15.16:8080
    export LABEL_STUDIO_TOKEN=<your API token, Account & Settings > Access Token>
    python export_labelstudio.py --project 12 --out deepforest_annotations.csv
    python export_labelstudio.py --project 12 --label species   # use per-box species as label
"""
import argparse
import math
import os
import re
import sys
import urllib.parse

import pandas as pd
import requests


def axis_aligned(x, y, w, h, rotation_deg):
    """Corners of a rectangle rotated (clockwise, y-down) about its top-left corner."""
    th = math.radians(rotation_deg)
    c, s = math.cos(th), math.sin(th)
    pts = [(x + px * c - py * s, y + px * s + py * c) for px, py in ((0, 0), (w, 0), (w, h), (0, h))]
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    return min(xs), min(ys), max(xs), max(ys)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", type=int, required=True, help="Label Studio project id")
    ap.add_argument("--out", default="deepforest_annotations.csv")
    ap.add_argument("--label", choices=["fixed", "species"], default="fixed",
                    help="'fixed' writes --label-name for every box; 'species' uses the per-region Choices if present")
    ap.add_argument("--label-name", default="Tree")
    ap.add_argument("--min-size", type=int, default=2, help="drop boxes narrower/shorter than this many pixels")
    args = ap.parse_args()

    url = os.environ.get("LABEL_STUDIO_URL")
    token = os.environ.get("LABEL_STUDIO_TOKEN")
    if not url or not token:
        sys.exit("Set LABEL_STUDIO_URL and LABEL_STUDIO_TOKEN in the environment.")

    r = requests.get(f"{url.rstrip('/')}/api/projects/{args.project}/export",
                     params={"exportType": "JSON", "download_all_tasks": "false"},
                     headers={"Authorization": f"Token {token}"}, timeout=300)
    r.raise_for_status()
    tasks = r.json()

    rows, rotated, dropped = [], 0, 0
    for t in tasks:
        name = urllib.parse.unquote(t["data"]["image"].split("/")[-1])
        name = re.sub(r"^[0-9a-f]{8}-", "", name)          # strip Label Studio upload prefix
        anns = [a for a in t.get("annotations", []) if not a.get("was_cancelled")]
        if not anns:
            continue
        results = anns[-1]["result"]                         # latest annotation wins
        species = {r["id"]: r["value"]["choices"][0] for r in results
                   if r["type"] == "choices" and r["value"].get("choices")}
        for res in results:
            if res["type"] != "rectanglelabels":
                continue
            W, H, v = res["original_width"], res["original_height"], res["value"]
            x, y = v["x"] / 100 * W, v["y"] / 100 * H
            w, h = v["width"] / 100 * W, v["height"] / 100 * H
            rot = v.get("rotation", 0) or 0
            if abs(rot % 360) > 0.01:
                rotated += 1
                xmin, ymin, xmax, ymax = axis_aligned(x, y, w, h, rot)
            else:
                xmin, ymin, xmax, ymax = x, y, x + w, y + h
            xmin, ymin = max(0, round(xmin)), max(0, round(ymin))
            xmax, ymax = min(W, round(xmax)), min(H, round(ymax))
            if xmax - xmin < args.min_size or ymax - ymin < args.min_size:
                dropped += 1
                continue
            label = args.label_name
            if args.label == "species":
                label = species.get(res["id"], args.label_name)
            rows.append((name, xmin, ymin, xmax, ymax, label))

    df = pd.DataFrame(rows, columns=["image_path", "xmin", "ymin", "xmax", "ymax", "label"])
    df.to_csv(args.out, index=False)
    print(f"{len(tasks)} tasks, {df.image_path.nunique()} images, {len(df)} boxes "
          f"({rotated} rotated -> axis-aligned, {dropped} dropped) -> {args.out}")
    print(df.label.value_counts().to_string())


if __name__ == "__main__":
    main()
