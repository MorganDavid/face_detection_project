# Face Detection on NVIDIA Jetson TX2 using Deep Convolutional Neural Networks.
This project uses Deep Convolutional Artificial Neural Networks (CNN) to carry out face detection through a web-cam on the NVIDIA JETSON TX2 embedded computer (windows version also available). Face detection updates on the TX2 run around 2FPS, while modern windows machines run much faster. 

The architecture is made up of two CNNs called P-net and R-net and is inspired by the architecture developed in "A Convolutional Neural Network Cascade for Face Detection by Haoxiang Li, Zhe Lin et al.". Some changes are made in this version to improve performance on low-power embeded systems. Our CNNs were trained on 25k annotated face images from the WIDER dataset and tested on Caltech-101 to yield test accuracy of 95%. 

The Keras models are all included and fully trained, so no training is required before running the system.
To run on Windows from root project directory use (requires Python 3 compiler):
```
python3 system/image_reader_windows.py
```

## Example output:

![example output of face detection](https://i.imgur.com/3RVufAV.png)

## Overview of code
Scripts for this project are seperated into three directories:

1. `data_extraction` : Extracts training and test data from the WIDER and Caltech-101 datasets respectively. Also includes a script to perform hard-negative mining using the .h5 model file defined in the `_model_dir` variable. Datasets are not included in this repo - WIDER and Caltech-101 required for training.
2. `system`          : Contains everything needed for running real time face detection on windows or the TX2 Linux distro; ready-to-run with trained models included. Main methods can be found in `image_reader_windows.py` and `image_reader.py` (for linux).
3. `training_etc`    : Handles training the Keras models. data_extraction scripts must all be run before running this.
