This repository contians **Source Code** and **Note** that i took during the learning process.

## What Does each Directory contains?

- [micrograd](micrograd/): Andrej Karpathy's first lecture "Backpropagation and NeuralNetwork". This Directory contains Notes and Python implementations with comments from the introduction to neural networks and backpropagation. The material builds a small automatic differentiation engine and a neural network from scratch. The lecture contains A comprehensive introduction to neural networks, **backpropagation**, and **gradient descent**. Covers **derivatives**, the **chain rule**, **computational graphs**, **automatic differentiation**, and **gradient propagation**, with step-by-step mathematical explanations and Python implementations. Demonstrates how to build a neural network and train it from scratch using gradient-based optimization.

- [makemore](makemore/):
  - [Part1](makemore/makemore_part1_bigrams.ipynb): [Notes](<makemore/makemore_part1_Character-Level Language Modeling - Makemore (Bigram Models).md>) and a [Jupyter notebook](makemore/makemore_part1_bigrams.ipynb) from _The Spelled-Out Intro to Language Modeling: Building Makemore_ made by Andrej Karpathy. This section develops a character-level bigram language model that learns from a dataset of names and generates new, name-like sequences. Topics covered:
    - **normalization, broadcasting, and add-one smoothing**
    - **likelihood, log-likelihood, and negative log-likelihood**
    - **one-hot encoding and matrix multiplication**
    - **softmax probabilities and regularization**
    - **training with backpropagation and gradient descent**
    - **comparing the count-based and neural-network approaches**.
  - Part 2: [Notes](<makemore/makemore_part2_Character-Level language modeling -Makemore (MLP).md>) and a [Jupyter notebook](makemore/makemore_part2_mlp.ipynb) covering the transition from bigram models to a multilayer perceptron (MLP) character-level language model. Topics include
    - **context windows, character embeddings, embedding lookup tables, tensor reshaping**
    - **hidden layers and `tanh` activations**
    - **logits and cross-entropy loss**
    - **mini-batch training, learning-rate selection, train/dev/test splits, model-capacity tuning**
  - Part 3: [Notes](<makemore/makemore_part3_Neural Network Initialization, Gradient Flow, BatchNormalization.md>) and a [Jupyter notebook](makemore/makemore_part3.ipynb) covering neural-network initialization, gradient flow, and batch normalization in deeper MLPs. Topics include:
    - **calibrating the initial loss and preventing saturated `tanh` activations**
    - **fan-in-aware/Kaiming initialization and activation-specific gain**
    - **Batch Normalization, running statistics, and training versus inference behavior**
    - **PyTorch-style `Linear`, `BatchNorm1d`, and `Tanh` layers for a modular deep MLP**
    - **activation, gradient, gradient-to-data, and update-to-data diagnostics**
