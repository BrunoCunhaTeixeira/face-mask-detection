# face-mask-detection

As part of a university project, a MobileNetV2 classifier was trained to classify faces into "faces with mask" and "faces without mask". Faces in which the mask is not worn correctly are classified as "faces without mask". Reatinaface was used for the localization of the faces.

Datasets:

MaskedFace-Net: https://doi.org/10.1016/j.smhl.2020.100144

and

https://github.com/chandrikadeb7/Face-Mask-Detection

## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install retinaface and Tensorflow.

```bash
$ pip install retina-face

$ pip install tensorflow
```

## Usage
Run:
```python
python maskdetector.py
```
Make sure you have a webcam connected.
Close the window with q.