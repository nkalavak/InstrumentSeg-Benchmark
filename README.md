# Bechmarking for Semantic Segmentation of Neurosurgical Instruments

## Dataset

Label Me annotation tool: http://labelme.csail.mit.edu/Release3.0/

### Classes of instruments

### Color assignments

## Code
1. getFrames.py: Get the frames from video.
2. getMasks.py: Generate masks from .xml files.

## Methods

* Hand-crafted features (Done - Nivii)
* Gaussian Mixture Models (Done - Shan)

* Random Forests (Semi-done - Shan)
* Superpixel Segmentation (w/ pooling) (Shan)
  * Conditional Random Fields
* U-net (w/ pre-trained weights) (Shan and Nivii)
* Mask R-CNN (Nivii)
* DeepLabV3+ (Nivii)
* ToolNet-C (Shan)
* Y-Net (Nivii)
* NiftyNet (Shan)

* Naive Bayesian classifiers

# Loss functions for training

# Metrics for inference
