import matplotlib.pyplot as plt
import numpy as np

import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.nn import functional as F

from learning.dataset import FontDataset
from model import Model

batch_size = 64
epochs = 2
train_size = 0.7

dataset = FontDataset("example", siamese=True)

train_size = int(train_size * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

for data in train_loader:
    img1, img2, label = data
    break

model = Model(img1.shape[3], img1.shape[2])
optimizer = Adam(model.parameters(), lr=1e-3)

def bce_loss(prediction, label):
    return F.binary_cross_entropy(prediction, label, reduction="sum")

# Train
for epoch in range(epochs):
    train_loss = 0
    for data in train_loader:
        img1, img2, label = data
        img1, img2, label = img1.type(torch.FloatTensor), img2.type(torch.FloatTensor), label.type(torch.FloatTensor)

        optimizer.zero_grad()
        
        prediction = model(img1, img2)
        loss = bce_loss(prediction, label)

        loss.backward()
        train_loss += loss.item()
        optimizer.step()

    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss / len(train_loader.dataset)))


# Test
predictions, labels = [], []
for data in test_loader:
    img1, img2, label = data
    img1, img2, label = img1.type(torch.FloatTensor), img2.type(torch.FloatTensor), label.type(torch.FloatTensor)

    predictions += list(model(img1, img2).detach().numpy())
    labels += list(label.detach().numpy())

print(labels, predictions)
print(f"Test accuracy: {1 - np.mean(np.abs(np.array(labels)-np.array(predictions)))}")
