# Neural Network from Scratch using Numpy for MNIST Dataset

This repository contains Python code for implementing a simple feedforward neural network from scratch using only NumPy. The neural network is trained on the MNIST dataset for digit classification.


## Table of Contents

- [Overview](#overview)
- [ReLU (Rectified Linear Activation)](#relu-rectified-linear-activation)
- [Softmax Activation](#softmax-activation)
- [Forward Propagation](#forward-propagation)
- [Backward Propagation](#backward-propagation)
- [Loss Function](#loss-function)
- [Preprocessing](#preprocessing)
- [Results](#results)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [References](#references)
- [License](#license)
- [Contributors](#contributors)

---

## Overview

The neural network architecture consists of an input layer, a hidden layer with ReLU activation, and an output layer with softmax activation. It's trained using stochastic gradient descent (SGD) with backpropagation.

## ReLU (Rectified Linear Activation)

ReLU (Rectified Linear Unit) is a popular activation function used in neural networks. It introduces non-linearity by outputting the input directly if it is positive, otherwise, it outputs zero. ReLU has become the preferred choice for many neural network architectures due to its simplicity and effectiveness in mitigating the vanishing gradient problem.

## Softmax Activation

Softmax activation is commonly used in the output layer of neural networks for multi-class classification problems. It converts the raw output scores of the network into probabilities, ensuring that they sum up to one. Softmax is particularly useful when dealing with mutually exclusive classes, as it provides a probability distribution over all classes.

## Forward Propagation

Forward propagation is the process of computing the output of a neural network given an input. It involves passing the input through each layer of the network, applying the activation functions, and generating the final output. In this implementation, forward propagation computes the output of the neural network given an input image.

## Backward Propagation

Backward propagation is the process of updating the weights of a neural network based on the computed gradients of the loss function with respect to the weights. It involves propagating the error backward from the output layer to the input layer, adjusting the weights using gradient descent. Backward propagation enables the network to learn from the training data by updating its parameters to minimize the loss.

## Loss Function

The loss function measures the difference between the predicted output of the neural network and the true labels. It quantifies how well the network is performing during training and provides feedback for adjusting the model parameters. In this implementation, the cross-entropy loss function is used, which is commonly employed for multi-class classification tasks.

## Preprocessing

Preprocessing is an essential step in preparing the input data for training a neural network. It involves transforming the raw data into a format that is suitable for the network architecture and learning algorithm. In this implementation, the MNIST images are flattened and normalized to ensure that pixel values are within the range [0, 1]. Additionally, the labels are converted to one-hot encoding to represent the target classes.

## Results
After training for 5 epochs, the model achieved an accuracy of approximately 90.83% on the test set.

---

## Requirements

- Python 3.x
- NumPy
- Matplotlib

## Installation

Clone the repository to your local machine:

```bash
git clone https://github.com/your_username/Neural-Network-from-Scratch.git
```

Install the required dependencies:

```bash
pip install numpy matplotlib
```

## Usage

1. Run all the cells to train the neural network.
2. After training, the output will display the training loss and accuracy for each epoch, as well as plots showing the training loss and accuracy trends.

## References

- [MNIST Dataset](http://yann.lecun.com/exdb/mnist/)
- [NumPy Documentation](https://numpy.org/doc/)
- [Matplotlib Documentation](https://matplotlib.org/stable/contents.html)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributors

- ![Rohan Agarkar](https://media.licdn.com/dms/image/D4D03AQGNwhF1H5O7Kg/profile-displayphoto-shrink_400_400/0/1705987860347?e=1719446400&v=beta&t=-8F6kpQ_4ooI5QLJbIHkoGOVys4jLPvNRyV4vnyVmRs) [Rohan Agarkar](https://github.com/RohanAgarkar)
- ![Abhiraj Chaudhuri](https://avatars.githubusercontent.com/u/117913120?v=4) [Abhiraj Chaudhuri](https://github.com/abhie7)

---
