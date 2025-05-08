# Classification of house numbers

## Overview

Copmarison of Convolutional Neural Networks and Vision Transformers on house numbers classification. For data I used SVHN dataset [2] with some simple preprocessing. As for neural networks, for CNN I closely followed the solution presented in "Multi-digit Number Recognition from Street View Imagery using Deep Convolutional Neural Networks," while for ViT I used Keras' example
This repository contains a Jupyter Notebook tutorial and all the necessary code, data, and documentation to reproduce the analysis and predictions.

There are two experiments presented here, each in their own notebook:

1) Preprocessing SVHN data and classifier training:
   - This experiment is based off of original SVHN dataset [1], first part is my preprocessing of dataest and then definition and training of classifier. After training there is evaluation part with confusion matrix.
2) Training visual transformer:
   - This experiment is based off of Keras' example for training of Vision Transformers on small datasets [3], data used is same as in previous experiment, as well as evaluation and confusion matrix.
  
## Conclusion

Both experiments are successful, with comparable results. While convolutional neural network reaches 91.37% accuracy, vision transformer reaches 89.53% accuracy.

## References
1. Multi-digit Number Recognition from Street View Imagery using Deep Convolutional Neural Networks: [link](https://arxiv.org/abs/1312.6082)
2. The Street View House Numbers (SVHN) Dataset: [link](http://ufldl.stanford.edu/housenumbers/)
3. Train a Vision Transformer on small datasets: [link](https://keras.io/examples/vision/vit_small_ds/)
