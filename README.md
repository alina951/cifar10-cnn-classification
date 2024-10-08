# CIFAR-10 CNN Classification
A convolutional neural network (CNN) classification model on the CIFAR-10 dataset using TensorFlow and Keras.


## What the project does
This project trains a convolutional neural network (CNN) to classify images from the CIFAR-10 dataset into one of 10 classes: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, and truck. The model is implemented using TensorFlow and Keras, and achieves a test accuracy of approximately 85%.

## Why the project is useful
This project is useful for several reasons:

* It provides a simple and easy-to-understand implementation of a CNN for image classification tasks.
* It serves as a baseline model for more complex image classification tasks.
* It demonstrates the use of TensorFlow and Keras for building and training deep learning models.


## Overview
The CIFAR-10 dataset consists of 60,000 32x32 color images in 10 classes, with 6,000 images per class. This project trains a CNN model to classify these images into their respective classes.

## Requirements
* TensorFlow 2.x
* Keras 2.x
* Python 3.x
* NumPy
* Matplotlib

## Running the Code
To run the code, simply execute the `cifar10_cnn.py` script in your terminal or command prompt.

## Model Architecture
The CNN model architecture consists of the following layers:

* Conv2D (64 filters, kernel size 3x3, activation='relu')
* BatchNormalization
* MaxPooling2D (pool size 2x2)
* Dropout (0.2)
* Conv2D (128 filters, kernel size 3x3, activation='relu')
* BatchNormalization
* MaxPooling2D (pool size 2x2)
* Dropout (0.2)
* Conv2D (256 filters, kernel size 3x3, activation='relu')
* BatchNormalization
* MaxPooling2D (pool size 2x2)
* Dropout (0.2)
* Flatten
* Dense (128 units, activation='relu')
* BatchNormalization
* Dropout (0.2)
* Dense (10 units, activation='softmax')

## Training and Evaluation
The model is trained on the training set using the Adamax optimizer and categorical cross-entropy loss. The model is evaluated on the test set using accuracy as the evaluation metric.

## Results
The model achieves a test accuracy of approximately 85%.

## Contributing
If you'd like to contribute to this project, please fork the repository and submit a pull request.

## License
This project is licensed under the MIT License. See the `LICENSE` file for details.

test
