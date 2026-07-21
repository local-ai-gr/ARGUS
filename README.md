# ARGUS
Deep learning pipeline and multi-sensor visualization tool for power line corridor vegetation risk assessment (RGB / multispectral / LiDAR).

ARGUS is a vegetation-assessment system for power line corridors in Slovenia. It detects and characterizes trees near transmission infrastructure from high-resolution aerial data (RGB imagery, multispectral imagery, and LiDAR point clouds) to help identify individuals that may pose a risk to the lines.

What it does
The core is an end-to-end tree analysis pipeline built on a RetinaNet detector (ResNet-50 + FPN backbone) with ResNet-based follow-up classifiers. Given aerial RGB frames, the pipeline:

detects individual tree crowns across whole image batches using a tiled prediction approach;
flags each detected tree as alive or dead (dead trees are a priority corridor risk), producing color-coded overlays;
classifies each crown by species, and exports a per-tree table (bounding box, health status, species, confidence).

The repository also documents the Drone Footage Data Space, a browser-based visualization tool with a georeferenced 2D map view (RGB orthophoto, multispectral bands, vegetation indices such as NDVI, and elevation products) and an interactive 3D LiDAR point-cloud view with a power line overlay and measurement tools for tree height and line-clearance analysis.
Status
Active development. Pretrained models are being fine-tuned on locally annotated imagery (via Label Studio) to adapt detection and species classification to Slovenian forest conditions.

