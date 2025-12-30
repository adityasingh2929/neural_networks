# Perceptron and Neural Networks From Scratch to MNIST

This repository contains simple and progressively advanced implementations of perceptrons and neural networks, starting from a hand coded logical AND gate, moving to a trained perceptron using scikit learn, and finally a fully connected neural network trained on the MNIST dataset using TensorFlow and Keras.

The focus of this repository is learning and understanding how neural networks work, not just using high level libraries.

---

## Contents

### 1. AND Gate using a Perceptron (From Scratch)

A perceptron is implemented manually using basic Python functions to simulate the behavior of an AND logic gate.

#### Concepts covered
- Weighted sum of inputs
- Bias term
- Step activation function
- How specific weights and bias model logical operations

The perceptron outputs:
- `1` only when both inputs are `1`
- `0` for all other input combinations

---

### 2. Training a Perceptron using Scikit Learn

A perceptron model is trained on a synthetic binary classification dataset generated using `make_classification`.

#### Concepts covered
- Dataset generation
- Train test split
- Gradient descent based learning
- Model evaluation using accuracy

---

### 3. Neural Network for Digit Classification (MNIST)

A neural network is built using TensorFlow and Keras to classify handwritten digits from the MNIST dataset.

#### Model architecture
- Input layer for 28×28 images
- Flatten layer to convert 2D images into a 1D array
- Dense hidden layer with 128 neurons
- Dense output layer with 10 neurons

---

## Activation Functions Used

### ReLU (Rectified Linear Unit)

Introduces non linearity and helps the model learn complex patterns.


---

### Softmax

Converts raw outputs into probabilities, ensuring values lie between 0 and 1 and sum to 1.


---

## Technologies Used

- Python
- NumPy
- Scikit learn
- TensorFlow
- Keras
- MNIST Dataset

---

## Purpose of This Repository

- Learn perceptrons from scratch
- Understand neural networks step by step
- Practice machine learning workflows
- Maintain a genuine learning based GitHub repository

---

## Future Improvements

- Train a perceptron from scratch without libraries
- Visualize decision boundaries
- Implement backpropagation manually
- Extend to deeper neural networks

---

## Note

This repository is for educational and learning purposes.
