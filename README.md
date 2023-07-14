
[![KimYoung_Planetoid](https://img.shields.io/badge/Planetoid-blue)](https://github.com/kimiyoung/planetoid)


# Graph Convolutional Networks (GCN) for Node Classification

This repository includes a Python implementation of a Graph Convolutional Network (GCN) for node classification on the Cora dataset.

The Cora dataset is a citation network dataset where nodes represent scientific papers, and edges denote citations. Node features are binary word vectors indicating the presence of corresponding words from a dictionary. The aim is to classify these nodes into one of seven classes, each corresponding to a different research field.

## Table of Contents
- [1. Getting Started](#1-getting-started)
  - [1.1 Installation](#11-installation)
  - [1.2 Dataset](#12-dataset)
- [2. The Model](#2-the-model)
  - [2.1 GCN Architecture](#21-gcn-architecture)
  - [2.2 Training](#22-training)
  - [2.3 Evaluation](#23-evaluation)
- [3. Visualization](#3-visualization)
  - [3.1 Loss and Accuracy Plots](#31-loss-and-accuracy-plots)
  - [3.2 Graph Structure](#32-graph-structure)
  - [3.3 Embedding Visualization](#33-embedding-visualization)
- [4. Contributing](#4-contributing)

## 1. Getting Started

### 1.1 Installation
- Install Python 3.7 or higher
- Install PyTorch 1.0 or higher
- Install Torch Geometric library
- Install NetworkX library
- Install Matplotlib library
- Install scikit-learn library

### 1.2 Dataset
The Cora dataset can be loaded using the Planetoid class in the Torch Geometric library. The dataset will be automatically downloaded and stored in the specified root directory.

## 2. The Model

### 2.1 GCN Architecture
The implemented GCN consists of two graph convolutional layers, followed by a ReLU nonlinearity and dropout after the first layer.

### 2.2 Training
The model is trained using the Adam optimizer, with a learning rate of 0.01 and weight decay of 5e-4. The training process uses the negative log likelihood loss.

### 2.3 Evaluation
After training, the model is evaluated based on how well it classifies the nodes in the test set. The test accuracy is printed for each training epoch.

## 3. Visualization

### 3.1 Loss and Accuracy Plots
Loss and test accuracy are plotted against epochs to visualize the model's performance over time.

### 3.2 Graph Structure
The graph structure of the Cora dataset is visualized using the NetworkX library.

### 3.3 Embedding Visualization
The hidden embeddings of the nodes are retrieved, reduced using t-SNE (or PCA), and then plotted to visualize the learned representations.

## 4. Contributing
If you have any ideas, feel free to open an issue and discuss it.

