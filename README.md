# MNIST Convolutional Neural Network (CNN) with PyTorch

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Architecture](#architecture)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Training and Evaluation](#training-and-evaluation)
- [Results](#results)
- [License](#license)
- [Acknowledgements](#acknowledgements)

## Introduction

This project implements a Convolutional Neural Network (CNN) using PyTorch to classify handwritten digits from the MNIST dataset. The model is designed to achieve high accuracy by leveraging multiple convolutional layers, batch normalization, and pooling layers. It includes functionalities for training, evaluating, and visualizing the model's performance over epochs.

## Features

- **Deep CNN Architecture**: Utilizes multiple convolutional layers with increasing filter sizes to capture complex patterns in the data.
- **Batch Normalization**: Applied after each convolutional layer to stabilize and accelerate training.
- **Max Pooling**: Reduces spatial dimensions, helping to manage computational complexity and control overfitting.
- **Flexible Training Parameters**: Adjustable learning rate, number of epochs, and batch size.
- **Performance Tracking**: Records loss and accuracy over epochs and visualizes them using Matplotlib.
- **GPU Acceleration**: Automatically utilizes CUDA-enabled GPUs if available for faster computation.

## Architecture

The CNN model consists of the following components:

1. **Convolutional Layers**:
   - **Conv1**: 1 input channel, 32 output channels, kernel size 3
   - **Conv2**: 32 input channels, 32 output channels, kernel size 3
   - **Conv3**: 32 input channels, 32 output channels, kernel size 3
   - **Conv4**: 32 input channels, 64 output channels, kernel size 3
   - **Conv5**: 64 input channels, 64 output channels, kernel size 3
   - **Conv6**: 64 input channels, 64 output channels, kernel size 3

2. **Batch Normalization Layers**: Applied after each convolutional layer to normalize the output and improve training stability.

3. **Max Pooling Layers**:
   - **MaxPool1**: Kernel size 2
   - **MaxPool2**: Kernel size 2

4. **Fully Connected Layer**:
   - Connects the flattened output of the convolutional layers to 10 output classes representing the digits 0-9.

## Requirements

- Python 3.7+
- PyTorch 1.7+
- torchvision
- NumPy
- Matplotlib

