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