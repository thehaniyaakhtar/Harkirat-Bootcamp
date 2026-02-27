# neural networks with PyTorch
import torch
import torch.nn as nn
import torch.optim as optim

# Our training data: XOR
import numpy as np

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

# convert data to Pytorch tenors
X_tensor = torch.FloatTensor(X)
y_tensor = torch.FloatTensor(y)

class XORNet(nn.Module):
  def __init__(self):
    super(XORNet, self).__init__()
    self.hidden = nn.Linear(2, 4)
    self.output = nn.Linear(4, 1)
    self.sigmoid = nn.Sigmoid()

  def forward(self, x):
    x = self.sigmoid(self.hidden(x))
    x = self.sigmoid(self.output(x))
    return x

torch.manual_seed(42)
model = XORNet()

criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=2.0)

pytorch_loss_history = []

print("Training Pytorch model...")
print("-" * 50)

for i in range(10000):
    predictions = model(X_tensor)
    loss = criterion(predictions, y_tensor)
    pytorch_loss_history.append(loss.item())

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if i%2000 == 0:
      print(f"Iterations {i:5d} | Loss: {loss.item():6f}")

print("-" * 50)
print(f"Iteration {10000:5d} | Loss: {pytorch_loss_history[-1]:.6f}")

print("\nPyTorch final predictions: ")
print("=" * 50)

with torch.no_grad():
  final_preds = model(X_tensor)
  for i in range(len(X)):
    pred = final_preds[i].item()
    print(f"input: {X[i]} -> Prediction: {pred:.4f} (Target: {y[i][0]})")

import matplotlib.pyplot as plt

plt.figure(figsize=(12, 5))

plt.plot(pytorch_loss_history, linewidth=0.5)
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.title('Pytorch Implementation')
plt.legend()

plt.show()

