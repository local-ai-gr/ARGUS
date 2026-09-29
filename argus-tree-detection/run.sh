#!/usr/bin/env bash
# Usage: bash run.sh            -> builds the image (first time) and trains in the background
#        docker logs -f argus-deepforest   -> follow progress
set -e
cd "$(dirname "$0")"
[ -f .env ] || { echo 'Copy .env.example to .env and set ORIG_DIR'; exit 1; }
grep -q '/path/to/original' .env && { echo "Edit .env first: set ORIG_DIR to the folder with the DJI originals"; exit 1; }
[ -f deepforest_annotations.csv ] || { echo "deepforest_annotations.csv not found: run export_labelstudio.py first"; exit 1; }
docker compose build
docker compose run --rm deepforest python -c "import torch;print('CUDA:',torch.cuda.is_available(),torch.cuda.get_device_name(0))"
docker compose up -d
echo "Training started. Follow with:  docker logs -f argus-deepforest"
