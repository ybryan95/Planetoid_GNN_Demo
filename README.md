![image](https://github.com/ybryan95/Planetoid_GNN_Demo/assets/123009743/50cb5f5f-2e4b-410c-863f-e36276799eb8)

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
The Cora dataset consists of 2708 scientific publications classified into one of seven classes. The citation network consists of 5429 links. Each publication in the dataset is described by a 0/1-valued word vector indicating the absence/presence of the corresponding word from the dictionary.

## 1.2 Prerequisites
You need to have PyTorch and PyTorch Geometric (PyG) installed. PyG is a geometric deep learning extension library for PyTorch dedicated to processing irregularly structured input data such as graphs, point clouds, and manifolds.

# 2. Code Overview
## 2.1 Model Definition
A two-layer GCN model is defined using PyG's GCNConv module. The GCN layers are followed by a ReLU activation function and dropout, with the last layer followed by a log_softmax activation function for the multi-class classification.

## 2.2 Training
The model is trained using negative log-likelihood loss. The training loop involves a forward pass of the model, calculating the loss, performing backpropagation to compute gradients, and updating the model parameters.

## 2.3 Testing
The trained model is tested on masked test data. Predictions are made using the model's forward pass and compared with actual labels to compute the test accuracy.

## 2.4 Visualization
Loss and test accuracy are plotted against epochs to visualize the model performance during training. Additionally, the final graph structure is visualized using the NetworkX library. Furthermore, the hidden layer embeddings are retrieved, dimensionally reduced using t-SNE or PCA, and plotted to visualize the clustering effect of the GCN model.

# 3. Contributing
Feel free to submit pull requests or propose changes. For major changes, please open an issue first to discuss the change you wish to make.

