# Carried-sack detection

[Download the original Roboflow YOLO export](https://github.com/kiran25y/sack_detection_benchmark_counting/releases/download/sack-benchmark-v1/Sackdetection.v1i.yolo26.zip). It contains 24,007 images and 16,400 carried/handled-sack boxes. Native splits contain 21,006 training, 2,001 validation and 1,000 test images. The training export contains three versions of 7,002 filename stems. Preserve recording and augmentation dependence when using these splits.

A missing box describes the carried-sack annotation. A loaded truck can contain zero annotated carried sacks. For whole-image inventory supervision, join the 300 reviewed totals by image hash and source archive path.

Original dataset: Sackdetection v1 (Roboflow, sack-detection workspace). [Original dataset and attribution](https://universe.roboflow.com/sack-detection/sackdetection/dataset/1) · [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) · [Export notice](upstream/README.dataset.txt). The original ZIP is distributed byte-for-byte. Its upstream scope and license apply to this dataset.
