
[![KimYoung_Planetoid](https://img.shields.io/badge/Planetoid-blue)](https://github.com/kimiyoung/planetoid)


# Graph Convolutional Networks (GCN) on Cora Dataset
This project demonstrates how to use Graph Convolutional Networks (GCNs) for node classification in a citation network using the Cora dataset. GCNs are a powerful method for handling graph structured data.

# Table of Contents
- [1. Getting Started](#1-getting-started)
    - [1.1 Dataset](#11-dataset)
    - [1.2 Prerequisites](#12-prerequisites)
- [2. Code Overview](#2-code-overview)
    - [2.1 Model Definition](#21-model-definition)
    - [2.2 Training](#22-training)
    - [2.3 Testing](#23-testing)
    - [2.4 Visualization](#24-visualization)
- [3. Contributing](#3-contributing)


# 1. Getting Started
## 1.1 Dataset
The Cora dataset is a prominent citation network dataset used for node classification tasks. It represents a scientific collaboration network, where nodes correspond to scientific papers, and edges represent citations between the papers. Node features are binary word vectors that encode the presence of a corresponding word from a dictionary which describes the paper. There are seven different classes, each one corresponding to a different research field.

## 1.2 Prerequisites
You need to have PyTorch and PyTorch Geometric (PyG) installed. PyG is a geometric deep learning extension library for PyTorch dedicated to processing irregularly structured input data such as graphs, point clouds, and manifolds.

# 2. Code Overview
## 2.1 Model Definition
A two-layer GCN model is defined using PyG's GCNConv module. The GCN layers are followed by a ReLU activation function and dropout, with the last layer followed by a log_softmax activation function for the multi-class classification.

The model is trained using the nodes that have known classifications, and the quality of the model is evaluated based on how well it classifies the remaining nodes. This is typical of semi-supervised learning scenarios in machine learning, where a small amount of the data is labeled.

## 2.2 Training
The model is trained using negative log-likelihood loss. The training loop involves a forward pass of the model, calculating the loss, performing backpropagation to compute gradients, and updating the model parameters.

After training, the code evaluates the model on the testing data and prints the test accuracy for each epoch. The code also collects the loss and test accuracy history for each epoch, which are then plotted to visualize the performance of the model over time.

Furthermore, the code also visualizes the graph structure of the Cora dataset using the NetworkX library, which helps understand the complex relationships in the data.

## 2.3 Testing
The trained model is tested on masked test data. Predictions are made using the model's forward pass and compared with actual labels to compute the test accuracy.

## 2.4 Visualization
Loss and test accuracy are plotted against epochs to visualize the model performance during training. Additionally, the final graph structure is visualized using the NetworkX library. Furthermore, the hidden layer embeddings are retrieved, dimensionally reduced using t-SNE or PCA, and plotted to visualize the clustering effect of the GCN model.

# 3. Contributing
Feel free to submit pull requests or propose changes. For major changes, please open an issue first to discuss the change you wish to make.

