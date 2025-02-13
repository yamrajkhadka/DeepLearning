![411296442-504b4cba-d542-4546-9802-f6590e53136e](https://github.com/user-attachments/assets/f07e2161-0a0a-473e-a5c2-cc2249fc9380)





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



## 🚀 Day 3: Understanding Weight & Bias Optimization in Neural Networks  

### 1️⃣ Introduction  
Before diving into activation functions and backpropagation, it's crucial to understand **how neural networks update weights and biases** to minimize loss and generate optimal outputs.  

### 2️⃣ Role of Weights & Biases  
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

### 3️⃣ Why Bias Matters? 🤔  
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

  

### 4️⃣ Gradient Descent: Updating Weights & Biases  
To minimize the loss function, we use **Gradient Descent**:  
\[
W_{new} = W - \eta \frac{\partial L}{\partial W}
\]
\[
b_{new} = b - \eta \frac{\partial L}{\partial b}
\]
where \( \eta \) is the learning rate. 

Gradient Descent Visualization:
![image](https://github.com/user-attachments/assets/032efe7e-bb0a-4ce6-9969-f0fb5dcfb6e6)

Logistic Regretation Derivatives For Gradient Descent:
![image](https://github.com/user-attachments/assets/4e7bd5bb-6d31-42a7-87ff-17f8498f2a5c)

Implimentation Of LR:
![image](https://github.com/user-attachments/assets/0795c333-9f48-4151-835c-f559a028371a)

Vectoritation_Visualization:
![image](https://github.com/user-attachments/assets/d9d4ffa6-5ef5-49be-bf75-ec45e847de05)





### 5️⃣ Code Implementation 💻  
![first1](https://github.com/user-attachments/assets/6e138233-715a-417f-a846-a30d7155f7cb)
![second1](https://github.com/user-attachments/assets/a799f5f0-e77c-4e38-9cdc-e00655fce8a8)
![Screenshot 2025-02-09 160306](https://github.com/user-attachments/assets/7a08a370-e4ff-4de8-b7b9-cab78cbda90e)
![Screenshot 2025-02-09 160345](https://github.com/user-attachments/assets/afb3052c-2af5-4d6d-9b9f-dfe6b5291b55)



## Day 4: Activation Functions (Deep Dive)

### Step 1: Why Do We Need Activation Functions?
Activation functions are essential in neural networks because:
1. **Prevent linearity**: Without activation functions, neural networks would behave like simple linear models, regardless of the number of layers.
2. **Enable complex pattern learning**: They allow the model to learn and represent complex, non-linear relationships in data.
3. **Control gradients**: They help mitigate issues like vanishing or exploding gradients during training.

---

### Step 2: Types of Activation Functions
Below is a detailed breakdown of common activation functions:
![image](https://github.com/user-attachments/assets/a3fbb023-c843-447c-9977-bddf88efedb9)





---

### Step 3: When to Use Which Activation Function?
### Hidden Layers:
- **ReLU**: Default choice for most hidden layers due to its simplicity and effectiveness.
- **Leaky ReLU**: Preferred for deeper networks to avoid dead neurons.

### Output Layer:
- **Binary Classification**: Use **Sigmoid**.
- **Multi-class Classification**: Use **Softmax**.
- **Regression**: Use **Linear Activation**.

---

### Step 4: How Do Activation Functions Prevent Neural Networks from Behaving Like Simple Linear Models?

#### What Happens Without Activation Functions?
Without activation functions, a neural network reduces to a linear transformation, no matter how many layers it has. Here's why:

1. **Linear Transformation**:
   - Each layer computes:  
     `Z = W · X + b`
   - Stacking multiple layers without activation functions results in:  
     `Z2 = (W2 · W1) · X + (W2 · b1 + b2)`  
     This is still a linear function of the input `X`.

2. **Limitation**:  
   Without non-linearity, the network cannot learn complex patterns (e.g., XOR, image recognition).

**Conclusion**: Without activation functions, deep networks collapse into a single-layer linear model.

---

#### How Activation Functions Introduce Non-Linearity
Activation functions introduce non-linearity, enabling the network to learn complex relationships. For example:

1. **ReLU Activation**:
   - Formula: `f(x) = max(0, x)`
   - When applied to a layer:  
     `A1 = max(0, Z1)`  
     `Z2 = W2 · A1 + b2`  
   - The non-linearity introduced by ReLU ensures that `Z2` is no longer a purely linear function of `X`.

2. **Result**:  
   The network can now model complex, non-linear relationships in the data.

---



## Day5: Backpropagation

Backpropagation computes the gradients of the loss with respect to each parameter in the network. This is done by propagating the error from the output layer back to the input layer using the chain rule.

---

### 3.1 Error at the Output Layer

For the output layer \( L \), the error \( \delta^{[L]} \) is computed as:

\[
\delta^{[L]} = A^{[L]} - Y
\]

- \( A^{[L]} \): Predicted output  
- \( Y \): True labels  

---

### 3.2 Gradients for the Output Layer

The gradients of the loss with respect to the weights \( W^{[L]} \) and biases \( b^{[L]} \) in the output layer are:

\[
\frac{\partial L}{\partial W^{[L]}} = \frac{1}{m} A^{[L-1]}^T \delta^{[L]}
\]

\[
\frac{\partial L}{\partial b^{[L]}} = \frac{1}{m} \sum_{i=1}^{m} \delta^{[L](i)}
\]

- \( A^{[L-1]} \): Activations from the previous layer  
- \( \delta^{[L]} \): Error at the output layer  

---

### 3.3 Propagating Error to Hidden Layers

For each hidden layer \( l \), the error \( \delta^{[l]} \) is computed as:

\[
\delta^{[l]} = \left( W^{[l+1]}^T \delta^{[l+1]} \right) \odot g'(Z^{[l]})
\]

- \( W^{[l+1]} \): Weights of the next layer  
- \( \delta^{[l+1]} \): Error from the next layer  
- \( g'(Z^{[l]}) \): Derivative of the activation function at layer \( l \)  
- \( \odot \): Element-wise multiplication  

---

### 3.4 Gradients for Hidden Layers

The gradients for the weights \( W^{[l]} \) and biases \( b^{[l]} \) in hidden layer \( l \) are:

\[
\frac{\partial L}{\partial W^{[l]}} = \frac{1}{m} A^{[l-1]}^T \delta^{[l]}
\]

\[
\frac{\partial L}{\partial b^{[l]}} = \frac{1}{m} \sum_{i=1}^{m} \delta^{[l](i)}
\]

---

This process of computing gradients and propagating errors backward through the network is the essence of backpropagation. It enables efficient training of neural networks by updating the parameters to minimize the loss. 🚀

![image](https://github.com/user-attachments/assets/c18466d2-6197-4003-a9a6-68e17e62f95d)
![image](https://github.com/user-attachments/assets/f00bafb9-c888-4594-bc80-a207912f80da)





## Day6: Neural Network Implimentation from scratch🧠

### 
This project implements a **feedforward neural network** from scratch using NumPy.  
It supports:
- Forward propagation (ReLU/Sigmoid activation)
- Backpropagation with gradient descent
- Binary classification using log loss
- Weight initialization and parameter updates  

This was part of my **Deep Learning learning journey** where I focused on understanding the fundamental concepts behind neural networks.

---
![Day6_first](https://github.com/user-attachments/assets/0710db1b-4a59-447d-94ea-7f7e7d83fa00)
![Day6_second](https://github.com/user-attachments/assets/0f0f749a-ae75-4222-9640-f94e4baa9233)
![Day6_third](https://github.com/user-attachments/assets/955f5d26-6659-447d-be0c-bb12d168cf8b)
![Day6_forth](https://github.com/user-attachments/assets/97f7a604-c8dd-403e-be19-7f35b41c3b96)
![Day6_fifth](https://github.com/user-attachments/assets/66170f85-6dd5-4c5e-a464-9ec40c72863a)








