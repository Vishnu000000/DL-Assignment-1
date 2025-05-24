# Deep Learning Assignment 1

This repository contains the implementation and experiments for the deep learning assignment. The project is structured into several Python files:

## File Structure

- **network.py**  
  Contains all activation functions and the full implementation of the `NeuralNetwork` class. This module must be imported first by any training or analysis script.

- **sweep.py**  
  Contains the sweep configuration and the `sweep_train()` function. This file is used to run hyperparameter sweeps using WandB.

- **visualize.py**  
  Contains functions to retrieve and visualize artifacts (e.g., the best confusion matrix) from WandB. Also includes a (placeholder) function for loss function comparison.

- **main.py**  
  The main entry point for the project. You can choose a mode to run:
  - `train`: Run a single training run using a fixed configuration.
  - `sweep`: Run the hyperparameter sweep agent.
  - `visualize`: Visualize the best confusion matrix artifact from the runs.
  - `loss_comparison`: Compare loss functions based on logged metrics.

## Features
- Supports **multiple hidden layers** with customizable sizes.
- Implements **SGD, Momentum, NAG, RMSprop, Adam, and Nadam** optimizers.
- Supports **ReLU, Sigmoid, Tanh, and Identity** activation functions.
- Supports **Random and Xavier** weight initialization.
- Allows **L2 regularization (Weight Decay)** to prevent overfitting.
- Trains on **Fashion-MNIST** or **MNIST** datasets.

---

## Installation
Ensure you have Python 3 installed, then install dependencies:
```bash
pip install numpy matplotlib wandb
```
Also I have imported datasets from keras.datasets

---

## Training the Model
To train the model, use the `train` function with the following parameters:

### **Function Signature:**
```python
train(inputSize, hiddenLayers, outputSize, sizeOfHiddenLayers, batchSize, learningRate,
      initialisationType, optimiser, epochs, activationFunc, weightDecay, lossFunc,
      dataset, beta, beta1, beta2, epsilon)
```

### **Example Usage:**
```python
train(inputSize=784, hiddenLayers=4, outputSize=10, sizeOfHiddenLayers=64,
      batchSize=32, learningRate=0.001, initialisationType="xavier", optimiser="nadam",
      epochs=10, activationFunc="relu", weightDecay=0.0001, lossFunc="cross_entropy",
      dataset="fashion_mnist", beta=0.9, beta1=0.9, beta2=0.999, epsilon=1e-8)
```

---

## Optimizers
The model supports the following optimization algorithms:
- **SGD** (Stochastic Gradient Descent)
- **Momentum-Based Gradient Descent**
- **Nesterov Accelerated Gradient Descent (NAG)**
- **RMSprop**
- **Adam**
- **Nadam** (Nesterov-accelerated Adam)

---

## Activation Functions
Supported activation functions:
- `tanh`
- `sigmoid`
- `relu`
- `identity`

---

## Weight Initializers
Supported weight initialization methods:
- `random`
- `xavier`

---

## Best Observations
From experimentation, the best configuration achieved **88.23% validation accuracy**:
- **Epochs:** 10
- **Hidden Layers:** 4 (each with 64 neurons)
- **Learning Rate:** 0.001
- **Optimizer:** aadam
- **Batch Size:** 32
- **Weight Initialization:** Xavier
- **Activation Function:** ReLU
- **Loss Function:** Cross-Entropy

For better performance, **data augmentation** can be used to reach **95% accuracy**.

---
## Wandb Report Link:
https://wandb.ai/cs24m022-iit-madras-foundation/Deep_Learning_Assignment1_cs24m022/reports/DA6401_Assignment1--VmlldzoxMTgzMDM4NQ?accessToken=2uisv4p86papc3tw1t3dso655xj0y97jps42y74abc8t9ofqgtpy2xtp35g8ovwi

---
## Github Repository Link
 https://github.com/Vishnu000000/DL-Assignment-1

---

## Way to run train.py
python train.py  -we "manasdeshpande4902-iit-madras" -wp "Trial" -o "adam" -lr 0.001 -l "cross_entropy" -e 10 -w_i "xavier" -w_d 0.005 -a "tanh"

