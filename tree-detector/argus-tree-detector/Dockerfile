# Inference image for the ARGUS fine-tuned DeepForest tree detector.
# Works on a GPU host (with NVIDIA Container Toolkit) and falls back to CPU otherwise.
FROM pytorch/pytorch:2.4.0-cuda12.4-cudnn9-runtime
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt
WORKDIR /app
COPY detect_trees.py argus_trees.ckpt /app/
ENTRYPOINT ["python", "detect_trees.py"]
