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



##Day2
Today, I learned about Multi-Layer Perceptrons (MLP) and Forward Propagation, which are key concepts in deep learning. Below is a brief summary of my learning along with visualization, intuition, and code implementation.

🧠 What is Multi-Layer Perceptron (MLP)?
MLP is a type of artificial neural network that consists of multiple layers:
✔ Input Layer – Takes the raw input data
✔ Hidden Layer(s) – Applies transformations using weights & activation functions
✔ Output Layer – Produces the final prediction
Intution behind MLP:
!



🔁 Understanding Forward Propagation
Forward propagation is the process where input data moves through the network, passing through each layer, applying weights, biases, and activation functions, to produce an output.

💡 The steps:
1️⃣ Compute weighted sum of inputs in each neuron
2️⃣ Apply activation function (e.g., Sigmoid, ReLU)
3️⃣ Pass the result to the next layer until reaching the output

🖼 (Here, add your forward propagation visualization.)





