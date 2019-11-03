# Face Detection on NVIDIA Jetson TX2 using Deep Convolutional Neural Networks.
This project uses Deep Convolutional Artificial Neural Networks (DCNN) to carry out face detection through a web-cam on the NVIDIA JETSON TX2 embedded computer including a version for Windows. Face detection updates on the TX2 run around 2FPS, while modern windows machines should be much faster. 

The architecture is made up of two CNNs called P-net and R-net and is heavily inspired by the architecture developed in "A Convolutional Neural Network Cascade for Face Detection by Haoxiang Li, Zhe Lin et al." These CNNs were trained on 25k annotated face images from the WIDER dataset and tested on Caltech-101 to yield test accuracy of 95%. 

To run on Windows from root project directory (requires Python 3 compiler):
```
python3 system/image_reader_windows.py
```

Example output:
![example output of face detection](https://i.imgur.com/3RVufAV.png)
