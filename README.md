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
* BiSeNet (Nivii)
* AdapNet (Nivii)
* FRRN (Nivii)

# Loss functions for training
* Binary Cross-entropy

# Metrics for inference
1. Precision
2. Recall
3. F1 score
4. Average accuracy
5. Per-class accuracy
6. Mean IoU
7. Jaccard Index
