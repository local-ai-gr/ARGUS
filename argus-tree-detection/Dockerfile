# CUDA 12.4 matches the host driver 550.x
FROM pytorch/pytorch:2.4.0-cuda12.4-cudnn9-runtime
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir deepforest
WORKDIR /work
CMD ["python", "prepare_and_train_deepforest.py"]
