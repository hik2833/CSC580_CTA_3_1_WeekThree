"""
CSC580 CTA 3.1 - Linear Regression with TensorFlow
Author: Dr.Rivera Matthews

Description:
-------------
This script trains a simple linear regression model using TensorFlow
to predict Y from X given noisy linear data.

Includes:
- Random seed for reproducibility
- Data generation with noise
- Data visualization
- TensorFlow model with placeholders, trainable variables
- Cost function (MSE)
- Optimizer (Gradient Descent)
- Training loop with progress printing
- Final cost, weights, bias
- Fitted regression line plot

Instructions:
-------------
1. Install requirements:
   pip install numpy matplotlib tensorflow

2. Run:
   python linear_regression_tensorflow_assignment.py
"""

import numpy as np
import tensorflow.compat.v1 as tf
import matplotlib.pyplot as plt

tf.disable_v2_behavior()

# ======================
# 1️⃣ Set Seeds
# ======================
np.random.seed(101)
tf.set_random_seed(101)

# ======================
# 2️⃣ Generate Data
# ======================
x = np.linspace(0, 50, 50)
y = np.linspace(0, 50, 50)

# Add noise
x += np.random.uniform(-4, 4, 50)
y += np.random.uniform(-4, 4, 50)

n = len(x)

# ======================
# 3️⃣ Plot Training Data
# ======================
plt.figure(figsize=(8, 5))
plt.scatter(x, y, color='blue', label='Training Data')
plt.title('Noisy Linear Training Data')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.grid(True)
plt.savefig('training_data_plot.png')
plt.show()

# ======================
# 4️⃣ Define Placeholders
# ======================
X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

# ======================
# 5️⃣ Trainable Variables
# ======================
W = tf.Variable(np.random.randn(), name='weight', dtype=tf.float32)
b = tf.Variable(np.random.randn(), name='bias', dtype=tf.float32)

# ======================
# 6️⃣ Hyperparameters
# ======================
learning_rate = 0.01
training_epochs = 1000

# ======================
# 7️⃣ Hypothesis
# ======================
hypothesis = W * X + b

# ======================
# 8️⃣ Cost Function
# ======================
cost = tf.reduce_mean(tf.square(hypothesis - Y))

# ======================
# 9️⃣ Optimizer
# ======================
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

# ======================
# 🔟 Training Process
# ======================
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print("\nStarting Training...\n")
    
    for epoch in range(training_epochs):
        for (xi, yi) in zip(x, y):
            sess.run(optimizer, feed_dict={X: xi, Y: yi})
        
        if (epoch+1) % 100 == 0:
            c_val = sess.run(cost, feed_dict={X: x, Y: y})
            w_val = sess.run(W)
            b_val = sess.run(b)
            print(f"Epoch {epoch+1:4d}: Cost = {c_val:.4f}, Weight = {w_val:.4f}, Bias = {b_val:.4f}")

    final_cost = sess.run(cost, feed_dict={X: x, Y: y})
    final_W = sess.run(W)
    final_b = sess.run(b)

    print("\n✅ Training Complete!")
    print(f"Final Cost: {final_cost:.4f}")
    print(f"Trained Weight: {final_W:.4f}")
    print(f"Trained Bias: {final_b:.4f}")

    # ======================
    # 11️⃣ Plot Fitted Line
    # ======================
    predicted = final_W * x + final_b
    plt.figure(figsize=(8, 5))
    plt.scatter(x, y, color='blue', label='Training Data')
    plt.plot(x, predicted, color='red', label='Fitted Line')
    plt.title('Linear Regression Fit with TensorFlow')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.grid(True)
    plt.savefig('fitted_line_plot.png')
    plt.show()
