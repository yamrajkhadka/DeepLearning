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








