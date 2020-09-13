# Loading necessary library
import numpy as np
import random
import torch
import matplotlib.pyplot as plt
import torch.nn as nn

# Declaring some global variable
input_size, output_size = 1, 1
num_epochs = 60



# Toy Dataset
x_train = [[round(random.uniform(1, 5), 2) for i in range(1)] for j in range(15)]
x_train = np.array(x_train, dtype=np.float32)
print(x_train)

y_train = [[round(random.uniform(1, 5), 2) for i in range(1)] for j in range(15)]
y_train = np.array(y_train, dtype=np.float32)
print(y_train)

print(x_train.shape)
print(y_train.shape)


# Linear Regression Model
model = nn.Linear(input_size, output_size)

# Loss and Optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

# Train the Model
for epoch in range(num_epochs):

    # numpy to tensor
    inputs = torch.from_numpy(x_train)
    targets = torch.from_numpy(y_train)

    # Forward pass
    outputs = model(inputs)
    loss = criterion(outputs, targets)

    # Backward and optimize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 5 == 0:
        print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch + 1, num_epochs, loss.item()))

# plot the graph
predicted = model(torch.from_numpy(np.array(x_train))).detach().numpy()
plt.plot(x_train, y_train, 'ro', label='Original data')
plt.plot(x_train, predicted, label='Fitted line')
plt.legend()
plt.show()


# Save the model checkpoint
torch.save(model.state_dict(), 'model.ckpt')

