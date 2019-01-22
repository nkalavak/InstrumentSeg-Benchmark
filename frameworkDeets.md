## Framework outline

0. Creating a dataset locator and download functionality.
1. separation of data into training, testing and validation sets;
  * Train and test split is usually available
2. randomized sampling during training;
3. image data loading and sampling;
4. data augmentation;
  * Online
  * Offline
5. a network architecture defined as the composition of many simple functions;
  * ModelZoo - Architectures defined in the README.md file
  * Transfer learning
6. a fast computational framework for optimization and inference;
7. metrics for evaluating performance during training and inference.

* Data size - 720 x 480
* Data format - JPEG or PNG

TASK:
* Shan and Nivii try NiftyNet -- Evaluate what is good and what is missing
* What applications can be integrated and how do we set it up for easier integration

TECHNICAL DETAILS:

* Python 3.6
* OpenCV 3.3/3.4
* TensorFlow 1.12 (Nivii)/ TensorFlow <version> (Shan)
* LabelMe for annotation
* TensorBoard
