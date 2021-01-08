import argparse
import json
import numpy as np
from collections import defaultdict
import time
import random as rd
import os

import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
import torch.nn.functional as F

from learning.dataset import get_train_test_dataset
from model import Model

def bce_loss(prediction, label):
    return F.binary_cross_entropy(prediction, label, reduction="sum")

def mse_loss(prediction, label):
    return ((prediction - label)**2).mean()

def train_and_eval(width, height, device, learning_rate, epochs, train_loader, test_loader, font_to_label, print_every_k_batches):
    model = Model(width, height)
    model.to(device)
    optimizer = Adam(model.parameters(), lr=learning_rate)
        

    losses = defaultdict(list)

    # Train
    model.train()
    for epoch in range(epochs):
        print(f"Epoch {epoch}")
        running_loss, epoch_loss = 0,0
        start_time = time.time()
        for i, data in enumerate(train_loader):
            img1, _, img2, _, label = data
            img1, img2, label = img1.type(torch.FloatTensor).to(device), img2.type(torch.FloatTensor).to(device), label.type(torch.FloatTensor).to(device)

            optimizer.zero_grad()
            
            prediction = model(img1, img2)
            loss = bce_loss(prediction, label)

            loss.backward()
            running_loss += loss.item()
            epoch_loss += loss.item()
            optimizer.step()
            losses[epoch].append(loss.item())

            if i % print_every_k_batches == print_every_k_batches - 1:
                print(f"  [{i+1:4d}] loss: {running_loss/print_every_k_batches:.3f}    ({round(time.time() - start_time, 3)} s)")
                running_loss = 0.0
                start_time = time.time()
        
        print('> Epoch: {}  -  Global average loss: {:.4f}\n'.format(
            epoch, epoch_loss / len(train_loader.dataset)))

    path_to_result_folder = os.path.join('results', str(time.time()))
    os.mkdir(path_to_result_folder)
    torch.save(model, os.path.join(path_to_result_folder, 'model'))

    with open(os.path.join(path_to_result_folder, "loss_history.json"), 'w') as f:
        json.dump(losses, f, indent=4)

    # Test
    model.eval()
    predictions, labels, font_pairs = [], [], []
    for data in test_loader:
        img1, label1, img2, label2, label = data
        img1, img2, label = img1.type(torch.FloatTensor).to(device), img2.type(torch.FloatTensor).to(device), label.type(torch.FloatTensor).to(device)

        predictions += model(img1, img2).detach().tolist()
        labels += label.detach().tolist()
        font_pairs += [(l1,l2) for l1,l2 in zip(label1.detach().tolist(), label2.detach().tolist())]

    with open(os.path.join(path_to_result_folder, "predictions.json"), 'w') as f:
        json.dump({"predictions": predictions, "labels": labels, "font_pairs": font_pairs, "font_to_label": font_to_label}, f, indent=4)
    print(f"\n> Test accuracy: {1 - np.mean(np.abs(np.array(labels)-np.round(predictions)))}")

def main(batch_size, epochs, train_size, dataset, print_every_k_batches, n_transformations, learning_rate):
    np.random.seed(0)
    rd.seed(0)
    torch.manual_seed(0)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device:",device)

    train_dataset, test_dataset = get_train_test_dataset(dataset, train_size, n_transformations, siamese=True)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    img1, _, _, _, _ = next(iter(train_loader))
    train_and_eval(img1.shape[3], img1.shape[2], device, learning_rate, epochs, train_loader, test_loader, train_dataset.font_to_label, print_every_k_batches)

    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-b", "--batch_size", type=int, default=64, 
        help="dataset batch Epochssize")
    parser.add_argument("-e", "--epochs", type=int, default=2, 
        help="number of epochs")
    parser.add_argument("-pkb", "--print_every_k_batches", type=int, default=2, 
        help="every k batch, the running loss is plot")
    parser.add_argument("-t", "--train_size", type=float, default=0.7, 
        help="dataset train size")
    parser.add_argument("-d", "--dataset", type=str, default="example", 
        help="dataset name in the data folder")
    parser.add_argument("-nt", "--n_transformations", type=int, default=2, 
        help="roughly the number of transformations per image")
    parser.add_argument("-lr", "--learning_rate", type=float, default=1e-3, 
        help="training learning rate")
    args = parser.parse_args()
    print(f"\n> args:\n{json.dumps(vars(args), sort_keys=True, indent=4)}\n")
    
    main(args.batch_size, args.epochs, args.train_size, args.dataset, args.print_every_k_batches, args.n_transformations, args.learning_rate)