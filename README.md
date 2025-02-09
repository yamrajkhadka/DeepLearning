# DeepLearning
Deep Learning Journey 🚀 This repo covers key concepts, code implementations, and projects on deep learning. It includes Jupyter notebooks, explanations, real-world case studies, and optimization techniques. Topics include neural networks, CNNs, RNNs, and model tuning. Explore and learn together! 🚀
## Day 1: Perceptron - A Simple Neural Network Model
On Day 1 of my Deep Learning journey, I explored the **Perceptron**, a foundational algorithm for binary classification. In this session, I learned how to:

- Build a Perceptron model from scratch.
- Visualize decision boundaries.
- Train a Perceptron on a dataset.
- Evaluate model accuracy.
- Understand the Perceptron loss function.
- Identify the limitations of the Perceptron.

---

### Key Learnings

#### 1. What is a Perceptron?
- A **Perceptron** is a single-layer neural network that performs binary classification.
- Unlike Perceptrons, **Neurons** in multi-layer networks are part of more complex models with non-linear activation functions.

#### Key Difference:
A Perceptron is a simplified mathematical model of a neuron used in machine learning, while a Human Neuron is a complex biological cell involved in brain functions.

![Screenshot 2025-02-05 005546](https://github.com/user-attachments/assets/6b4653ff-476e-4019-8f76-6a5128fb44c5)
---
#### 2. Building and Visualizing a Perceptron
I implemented a **Perceptron** in Python and visualized its decision boundary between two classes.
![image](https://github.com/user-attachments/assets/5d8a4a5c-cd6c-4b1f-9f94-54bedbc220b6)
![image](https://github.com/user-attachments/assets/b16f620c-8a1e-4b56-ad33-00183b60b7f3)

#### 3.Limitation of perceptron
Inability to Solve Non-Linearly Separable Problems. The Perceptron struggles with datasets where classes cannot be separated by a straight line. A classic example is the XOR problem, which it cannot solve due to its linear decision boundary. 
!![image](https://github.com/user-attachments/assets/e8e57bf7-6078-4f2c-8eac-9da87b173cfb)




## 🚀 Day 2: Multi-Layer Perceptron (MLP) & Forward Propagation:

Today, I explored **Multi-Layer Perceptrons (MLP)** and **Forward Propagation**, key concepts in deep learning. Below is a summary of my learning with **visualizations, intuition, and code implementation**.  

---

### 📌 What is Multi-Layer Perceptron (MLP)?  
MLP is a type of artificial neural network consisting of multiple layers:  
✔ **Input Layer** – Accepts the raw input data  
✔ **Hidden Layer(s)** – Applies transformations using weights & activation functions  
✔ **Output Layer** – Produces the final prediction  

MLP intution:
![image](https://github.com/user-attachments/assets/80bd9c1d-3b8b-45ec-b1d4-332057d34b1f)

MLP visualization in tensorflowbackground:
![image](https://github.com/user-attachments/assets/570624b1-ae35-42c1-b3c0-5873c3682666)



---

### 🔁 Understanding Forward Propagation  
Forward propagation is the process where input data moves through the network:  
1️⃣ Compute weighted sum of inputs in each neuron  
2️⃣ Apply activation function (e.g., Sigmoid, ReLU)  
3️⃣ Pass the result to the next layer until reaching the output  

Selfmade NN for forward propagation implementation:
![image](https://github.com/user-attachments/assets/a9728eb0-7252-4547-a9be-e4979463c06c)


---

### 📝 Implementation in Python (Without DL Framework)  
Here’s a simple **NumPy implementation** of forward propagation in an MLP in pythan:
![image](https://github.com/user-attachments/assets/854d3865-5021-4e33-9ca8-cb80bf2efbd4)

Implimentation using dl frameworks:
![image](https://github.com/user-attachments/assets/cf395147-5e39-488d-8d35-c444ecbdbf8f)

Next step: I'm curious to dive deeper into how neural networks update weights and biases to generate optimal output. Before exploring activation functions and backpropagation



# 🚀 Day X: Understanding Weight & Bias Optimization in Neural Networks  

## 1️⃣ Introduction  
Before diving into activation functions and backpropagation, it's crucial to understand **how neural networks update weights and biases** to minimize loss and generate optimal outputs.  

## 2️⃣ Role of Weights & Biases  
- **Weights (W):** Determine the strength of connections between neurons.  
- **Bias (b):** Allows shifting of activation functions, enhancing flexibility in decision boundaries.  

Each neuron computes:  
\[
z = W \cdot X + b
\]  
where:  
- \( W \) = weight  
- \( X \) = input  
- \( b \) = bias  

## 3️⃣ Why Bias Matters? 🤔  
Without bias, all decisions must pass through the origin (0,0), which restricts the model’s flexibility.  

Example with sigmoid:  
\[
\sigma(WX + b) = \frac{1}{1 + e^{-(WX + b)}}
\]  
- If \( b > 0 \), the function activates **earlier**.  
- If \( b < 0 \), the function activates **later**.  

For **ReLU Activation**:  
\[
f(z) = \max(0, WX + b)
\]  
- \( b > 0 \) → ReLU activates **sooner**.  
- \( b < 0 \) → ReLU activates **later**.  

(📷 Add your **bias visualization image** here)  

## 4️⃣ Gradient Descent: Updating Weights & Biases  
To minimize the loss function, we use **Gradient Descent**:  
\[
W_{new} = W - \eta \frac{\partial L}{\partial W}
\]
\[
b_{new} = b - \eta \frac{\partial L}{\partial b}
\]
where \( \eta \) is the learning rate.  

## 5️⃣ Code Implementation 💻  
(📂 Link to the Jupyter Notebook: **[weight_bias_optimization.ipynb](./weight_bias_optimization.ipynb)**)  

```python





# Neural Network Weight and Bias Updates

This repository explores how neural networks update weights and biases to optimize output. It covers forward propagation, loss calculation, gradient descent, and the role of bias in activation functions.

## Table of Contents
1. [Introduction](#introduction)
2. [Key Concepts](#key-concepts)
   - [Forward Propagation](#forward-propagation)
   - [Loss Calculation](#loss-calculation)
   - [Gradient Descent](#gradient-descent)
   - [Role of Bias](#role-of-bias)
3. [Code Implementation](#code-implementation)
4. [Visualizations](#visualizations)
5. [Notebook](#notebook)
6. [License](#license)

---

## Introduction
Neural networks learn by adjusting their weights and biases to minimize a loss function. This repository provides a step-by-step explanation and implementation of these concepts.

---

## Key Concepts

### Forward Propagation
Each neuron computes:
\[
z = W \cdot X + b
\]
where:
- \( W \) = weights
- \( X \) = input
- \( b \) = bias

The result is passed through an activation function (e.g., sigmoid, ReLU) to introduce non-linearity.

### Loss Calculation
The loss function quantifies the difference between predicted and actual values. Common loss functions include:
- **Mean Squared Error (MSE)** for regression
- **Cross-Entropy Loss** for classification

### Gradient Descent
Gradient descent is used to update weights and biases:
\[
W_{\text{new}} = W - \eta \frac{\partial L}{\partial W}
\]
\[
b_{\text{new}} = b - \eta \frac{\partial L}{\partial b}
\]
where:
- \( \eta \) = learning rate
- \( \frac{\partial L}{\partial W} \) and \( \frac{\partial L}{\partial b} \) = gradients of the loss function with respect to weights and biases

### Role of Bias
Bias allows the activation function to shift, improving the flexibility of the model. For example:
- In sigmoid: Bias shifts the curve left or right.
- In ReLU: Bias shifts the activation threshold up or down.

---

## Code Implementation
The repository includes the following Python scripts:
- **Forward Propagation**: `code/forward_propagation.py`
- **Loss Calculation**: `code/loss_calculation.py`
- **Gradient Descent**: `code/gradient_descent.py`
- **Bias Example**: `code/bias_example.py`

To run the code, clone the repository and execute the scripts:
```bash
git clone https://github.com/your-username/Neural-Network-Weight-Bias-Updates.git
cd Neural-Network-Weight-Bias-Updates/code
python forward_propagation.py
