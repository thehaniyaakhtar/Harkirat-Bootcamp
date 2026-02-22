import numpy as np
import matplotlib.pyplot as plt

X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])

y = np.array([
    [0],
    [1],
    [1],
    [0]
])

# a single layer perceptron cannot learn XOR
# thus, hidden layers

# printing XOR
print("XOR Dataset")
print("-"*30)
for i in range(len(X)):
  print(f"Input: {X[i]} -> Ouput: {y[i][0]}")

# plotting XOR data:
plt.figure(figsize=(8, 6))

# plotting points
for i in range(len(X)):
  color = 'red' if y[i][0] == 0 else 'blue'
  marker = 'o' if y[i][0] == 0 else 's'
  plt.scatter(X[i][0], X[i][1], c=color, s = 200, marker=marker, edgecolors='black', linewidths=2)

plt.xlabel('Input A', fontsize=12)
plt.ylabel('Input B', fontsize=12)
plt.title('XOR Problem', fontsize=12)

plt.xlim = (-0.5, 1,5)
plt.ylim(-0.5, 1.5)
plt.grid(True, alpha=0.3)
plt.legend(['Class 0 (same)', 'Class 1 (different)'], loc = 'upper right')
plt.show()

print("A single straught line cannot seperate these classes")
print("This is why we need hidden layers, they creae non linear boundaries")

# building a neural network
# use sigmoid as output is naturally between 0 and 1

# network architecture
INPUT_SIZE = 2 # 2 inputs
HIDDEN_SIZE = 4 # 4 neurons in hidden layer
OUTPUT_SIZE = 1 # One output

# weights: input to hidden layer
weights_input_hidden = np.random.randn(INPUT_SIZE, HIDDEN_SIZE) * 0.5
bias_hidden = np.zeros((1, HIDDEN_SIZE))

# weights: hidden to output layer
weights_hidden_output = np.random.randn(HIDDEN_SIZE, OUTPUT_SIZE) * 0.5
bias_output = np.zeros((1, OUTPUT_SIZE))

print("Network initialized with random weights: ")
print(f"Input -> Hidden weights shape: {weights_input_hidden.shape}")
print(f"Hidden -> Output weights shape: {weights_hidden_output.shape}")
print(f"\nTotal parameters: {weights_input_hidden.size + weights_hidden_output.size + bias_hidden.size + bias_output.size}")

# sigmoid activation function
def sigmoid(x):
  # squash values between 0 and 1
  return 1 / (1+ np.exp(-x))

# sigmoid derivative required for backpropagation
def sigmoid_derivative(x):
  s = sigmoid(x)
  return s * (1-s)

# visualization of sigmoid
x_range = np.linspace(-6, 6, 100)
# input range
# sigmoid saturates beyond +-6
# range shows S shape fully
# 100: number of equally spaced points
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(x_range, sigmoid(x_range), 'b-', linewidth=2)
plt.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
plt.xlabel('Input')
plt.ylabel('Output')
plt.title('Sigmoid Function')
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.plot(x_range, sigmoid_derivative(x_range), 'r-', linewidth=2)
plt.axhline(y=0.25, color='gray', linestyle='--', alpha=0.5, label='max=0.25')
plt.xlabel('Input')
plt.ylabel('Derivative')
plt.title('Sigmoid Derivative')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("\nNotice: Maximum derivative is onlu 0.25")
print("This is vanishing gradient problem")
print("10 layers: 0.25^10 = ", 0.25**10)

# building forward pass
def forward(X):
  # step 1: input to hidden
  # z_hidden = X @ W + b (linear regression)
  Z_hidden = np.dot(X, weights_input_hidden) + bias_hidden

  # a_hidden = sigmoid(Z_hidden) (activation)
  a_hidden = sigmoid(Z_hidden)

  # Step 2: Hidden to output
  # Z_output = a_hidden @ W + b
  Z_output = np.dot(a_hidden, weights_hidden_output) + bias_output

  # a_output = sigmoid(Z_output)
  a_output = sigmoid(Z_output)

  return Z_hidden, a_hidden, Z_output, a_output

z_h, a_h, z_o, predictions = forward(X)

print("Forward pass with UNTRAINED network: ")
print("-" * 50)
for i in range(len(X)):
  print(f"Input: {X[i]} -> Prediction: {predictions[i][0]:.4f} (Target: {y[i][0]})")

print("\n Predictions are garbage, the network hasnt learned anything")

# loss function: MSE
# mse = mean(prediction-target^2)
# the error is squared so that:
# all errors are positive
# big errors are penalized more than small errors

def compute_loss(y_true, y_pred):
  return np.mean((y_true - y_pred) ** 2)

# calculating initial loss
initial_loss = compute_loss(y, predictions)
print(f"Initial loss (untrained): {initial_loss:.4f}")
print("\nThis number should decrease as we train")

# 0.27: It is not good — it just means “random guessing.”

# Backpropagation
# calculate the error at output
# figure out how much each output weight contributed
# propagate error to hidden layer
# find each hidden weight contribution
# adjust weights proportionally

def backward(X, y, z_hidden, Z_output, a_output, learning_rate):
  global weights_input_hidden, bias_hidden, weights_hidden_output, bias_output

  m = X.shape[0] # number of training examples

  output_error = a_output - y

  output_delta = output_error * sigmoid_derivative(Z_output)

  grad_weights_hidden_output = np.dot(a_hidden.T, output_delta)/m
  grad_bias_output = np.mean(output_delta, axis=0, keepdims=True)

  # hidden layer
  hidden_error = np.dot(output_delta, weights_hidden_output.T)
  hidden_delta = hidden_error * sigmoid_derivative(a_hidden)

  grad_weights_hidden_output = np.dot(X.T, hidden_delta / m)
  grad_bias_hidden = np.mean(hidden_delta, axis=0, keepdims=True)

  weight -= learning_rate * np.gradient
