# Imports
import os
from typing import Union

import torch.nn.functional as F
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from pandas import io
import matplotlib
import matplotlib.pyplot as plt
import time
import numpy as np

start_time = time.time()


from torch.utils.data import (
    Dataset,
    DataLoader,
)  # Gives easier dataset managment and creates mini batches
import torch.nn as nn


soft = nn.Softmax(dim=1)

class NN(nn.Module):
    def __init__(self, input_size, num_classes):
        super(NN, self).__init__()
        self.fc1 = nn.Linear(input_size, 12)
        self.fc2 = nn.Linear(12, 12)
        self.fc3 = nn.Linear(12, 9)
        self.fc4 = nn.Linear(9, 9)
        self.fc5 = nn.Linear(9, 6)
        self.fc6 = nn.Linear(6, 6)
        self.fc7 = nn.Linear(6, num_classes)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        x = F.relu(self.fc6(x))
        x = self.fc7(x)
        return soft(x)


class SoloDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        x_data = self.annotations.iloc[index, 0:20]
        x_data = torch.tensor(x_data)
        y_label = torch.tensor(int(self.annotations.iloc[index, 20]))

        return (x_data.float(), y_label)


# Set device
device = torch.device("cpu" if torch.cuda.is_available() else "cpu")

# Hyperparameters
num_classes = 2
learning_rate = 0.001
batch_size = 64
num_epochs = 1000
input_size = 20

# Load Data
dataset = SoloDataset(
    csv_file="dataset/Covid_Dataset.csv", root_dir="test123", transform=transforms.ToTensor()
)

train_set, validation_set, test_set = torch.utils.data.random_split(dataset, [3260, 1087, 1087])
train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)
validation_loader = DataLoader(dataset=validation_set, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_set, batch_size=batch_size, shuffle=True)

# Model
model = NN(input_size=input_size, num_classes=num_classes).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

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
        testAcc.append(float(f"{float(num_correct) / float(num_samples) * 100:.2f}"))

    model.train()


print(len(train_set))
print(len(validation_set))
print(len(test_set))
lss = []
testAcc = []
vald_lss = []
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

    print(f"Train Cost at epoch {epoch} is {sum(losses) / len(losses)}")

    # for checkpoint...
    # torch.save({
    #     'epoch': epoch,
    #     'model_state_dict': model.state_dict(),
    #     'optimizer_state_dict': optimizer.state_dict(),
    #     'loss': sum(losses) / len(losses),
    #     'learning_rate': learning_rate,
    #     'batch_size': batch_size
    # }, os.path.join('symptoms_checker/trained_model/covid/', 'epoch_{}.pth'.format(epoch)))


second_model = NN(input_size=input_size, num_classes=num_classes).to(device)
# Loss and optimizer
criterion2 = nn.CrossEntropyLoss()
optimizer2 = optim.Adam(second_model.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
    losses2 = []

    for batch_idx, (data, targets) in enumerate(validation_loader):
        # Get data to cuda if possible
        data = data.to(device=device)
        targets = targets.to(device=device)

        # forward
        scores = second_model(data)
        loss2 = criterion2(scores, targets)

        losses2.append(loss2.item())

        # backward
        optimizer2.zero_grad()
        loss2.backward()

        # gradient descent or adam step
        optimizer2.step()

    vald_lss.append(sum(losses2) / len(losses2))
    print(f"Validation Cost at epoch {epoch} is {sum(losses2) / len(losses2)}")


print("Checking accuracy on Training Set")
check_accuracy(train_loader, model)
print("Checking Test Set accuracy on Training Set")
check_accuracy(test_loader, model)

print("Checking accuracy on Validation Set")
check_accuracy(validation_loader, second_model)
print("Checking Test Set accuracy on Validation Set")
check_accuracy(test_loader, second_model)

print(min_loss_found_at_epoch, min_loss_found)
print("Execution time: %s seconds" % (time.time() - start_time))
fig, ax = plt.subplots(1)
ax.plot(range(num_epochs), lss, color="blue", label="Train loss")
ax.plot(range(num_epochs), vald_lss, color="yellowgreen", label="Validation loss")
ax.set(xlabel='Epochs', ylabel='Loss')
plt.legend(loc="upper right", frameon=False)
plt.show()


x = torch.FloatTensor([[1,	1,	1,	1,	1,	0,	0,	0,	0,	1,	1,	1,	1,	0,	1,	0,	1,	1,	0,	0]])
with torch.no_grad():
    print(model(x))
    traced_cell = torch.jit.trace(model, (x))
torch.jit.save(traced_cell, "trained_model/covidTest.pt")

