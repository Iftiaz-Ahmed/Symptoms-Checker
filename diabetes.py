# Imports
import os
from typing import Union

import torch.nn.functional as F  # All functions that don't have any parameters
import pandas as pd
import torch
import torch.nn as nn  # All neural network modules, nn.Linear, nn.Conv2d, BatchNorm, Loss functions
import torch.optim as optim  # For all Optimization algorithms, SGD, Adam, etc.
import torchvision
import torchvision.transforms as transforms  # Transformations we can perform on our dataset
from pandas import io
import matplotlib
import matplotlib.pyplot as plt
import time

start_time = time.time()

# from skimage import io
from torch.utils.data import (
    Dataset,
    DataLoader,
)  # Gives easier dataset managment and creates mini batches
import torch.nn as nn  # All neural network modules, nn.Linear, nn.Conv2d, BatchNorm, Loss functions


# Create Fully Connected Network
class NN(nn.Module):
    def __init__(self, input_size, num_classes):
        super(NN, self).__init__()
        self.fc1 = nn.Linear(input_size, 32)
        self.fc2 = nn.Linear(32, 32)
        self.fc3 = nn.Linear(32, 16)
        self.fc4 = nn.Linear(16, 16)
        self.fc5 = nn.Linear(16, 8)
        self.fc6 = nn.Linear(8, num_classes)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        x = self.fc6(x)
        return x


class SoloDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        x_data = self.annotations.iloc[index, 0:8]
        x_data = torch.tensor(x_data)
        y_label = torch.tensor(int(self.annotations.iloc[index, 8]))

        return (x_data.float(), y_label)


# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
num_classes = 2
learning_rate = 0.001
batch_size = 5
num_epochs = 1000
input_size = 8

# Load Data
dataset = SoloDataset(
    csv_file="dataset/diabetes.csv", root_dir="test123", transform=transforms.ToTensor()
)

train_set, test_set = torch.utils.data.random_split(dataset, [1600, 400])
train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_set, batch_size=batch_size, shuffle=True)

# Model
model = NN(input_size=input_size, num_classes=num_classes).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

print(len(train_set))
print(len(test_set))
lss = []
min_loss_found = 100
min_loss_found_at_epoch = 0
# Train Network
for epoch in range(num_epochs):
    losses = []

    for batch_idx, (data, targets) in enumerate(train_loader):
        # Get data to cuda if possible
        data = data.to(device=device)
        targets = targets.to(device=device)

        # forward
        scores = model(data)
        loss = criterion(scores, targets)

        losses.append(loss.item())

        # backward
        optimizer.zero_grad()
        loss.backward()

        # gradient descent or adam step
        optimizer.step()

    lss.append(sum(losses) / len(losses))
    if sum(losses) / len(losses) < min_loss_found:
        min_loss_found = sum(losses) / len(losses)
        min_loss_found_at_epoch = epoch

    print(f"Cost at epoch {epoch} is {sum(losses) / len(losses)}")


# Check accuracy on training to see how good our model is
def check_accuracy(loader, model):
    num_correct = 0
    num_samples = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device)
            y = y.to(device=device)

            scores = model(x)
            _, predictions = scores.max(1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)

        print(
            f"Got {num_correct} / {num_samples} with accuracy {float(num_correct) / float(num_samples) * 100:.2f}"
        )

    model.train()


print("Checking accuracy on Training Set")
check_accuracy(train_loader, model)

print("Checking accuracy on Test Set")
check_accuracy(test_loader, model)

print(min_loss_found_at_epoch, min_loss_found)
print("Execution time: %s seconds" % (time.time() - start_time))
fig, ax = plt.subplots()
ax.plot(range(num_epochs), lss)
ax.set(xlabel='Epochs', ylabel='Loss')
plt.show()
