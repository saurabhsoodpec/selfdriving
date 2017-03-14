# CarND-P3-Simulator

## Overview

* [Introduction](#introduction)
* [Model Selection](#model-selection)
* [Model Diagram](#model-diagram)
* [Network Architecture](#network-architecture)
* [Augmenation And Recovery](#augmenation-and-recovery)
* [Dataset](#dataset)
* [Training](#training)
* [Prediction](#prediction)

## Introduction

This project is using all the machine learning skills learnt till and and use it to drive a self driving car in a simulator. The technique used to train the network is - Behaviour cloning.

The approach is to first manually drive the car in a simulator and then computer and learn the same by following the information learnt during the training. 

The challenge is to train the computer to learn driving in a generalized way using the limited training data. 


## Model Selection and Network Architecture

The Model used is a standard CNN network. The network includes three 3x3 convolution layers, and each convolution followed by 2x2 max pooling and dropout. This is followed by few Fully connected and output layers.

## Model Diagram
Here are the layer details - 

1) 3x3 Convolution
2) Activation Unit (ELU)
3) Max pooling 2x2
4) Dropout (50%)
5) 3x3 Convolution
6) Activation Unit (ELU)
7) Max pooling 2x2
8) Dropout (50%)
9) 3x3 Convolution
10) Activation Unit (ELU)
11) Max pooling 2x2
12) Dropout (50%)
13) Flatten
14) FC (1024)
15) Dropout (50%)
16) FC (128)
17) Dropout (50%)
18) FC (64)
19) Dropout (50%)
20) FC - Output (1)

## Augmenation And Recovery

The center angle camera is adjusted by +/- 0.15 to bring car back to center.

The following augmentations and data manipulations are perfomed

1. Resize images to 64*64
2. Apply AffineTransform on images to shift output and generated augmented data. 
3. Recovery images are ignored from the transform.

## Dataset

1) Some data was already provided along with the project.
2) Manually collected some data through test runs on simulator.

## Training

No. of Epochs = 50
Batch Size = 64
Optimizer = Adam
Learning Rate = 0.0001

## Prediction

Weights and model JSON is saved during each epoch.

