# TSMMR

This project hosts the code for implementing the TSMMR algorithm of The Visual Computer Journal paper:


# TSMMR: A Mixed Mamba Network with Cross-Drone Template Fusion and Text-Guided Redetection for Robust Aerial Tracking
![image](https://github.com/JerryBryant24/TSMMR/blob/main/12.jpg)

This repository contains the implementation of a novel Template Sharing-enhanced Mixed Mamba Network with Redetection strategy (TSMMR) for robust multi-drone single object tracking. The framework addresses key challenges such as viewpoint inconsistency, occlusion, and target loss in collaborative aerial systems. Its core innovations include a Dynamic Template Interaction Adapter (DTIA) for cross-drone feature complementation and a Text-Guided Cross-Drone Redetection strategy for robust recovery.

[**[Models and Raw Results (Baidu)]**](https://pan.baidu.com/s/1wbNr4GIy9gX8-9Ywj8TJUQ?pwd=7536 提取码: 7536)

### Acknowledgement
The code based on the [PyTracking](https://github.com/visionml/pytracking), [PySOT](https://github.com/STVIR/pysot) , [SiamBAN](https://github.com/hqucv/siamban) ,
[Biformer](https://ieeexplore.ieee.org/document/10203555).
We would like to express our sincere thanks to the contributors.

# System Requirements
Ubuntu 20.04 LTS
CUDA Version: 11.3 or higher (for GPU acceleration)
Python: 3.8

# Create and Activate Conda Environment

conda create -n tsmmr python=3.8 -y
conda activate tsmmr

# Train
Start training using train.py

[![DOI](https://zenodo.org/badge/1135137750.svg)](https://doi.org/10.5281/zenodo.18290398)
