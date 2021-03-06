import optuna
import argparse
import torch
import json

from torch.utils.data import DataLoader
from learning.dataset import get_train_test_dataset

from training import train_and_eval

def main(dataset, n_trials, print_every_k_batches, train_size, mode, rsr):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device:",device)

    train_dataset, test_dataset = get_train_test_dataset(dataset, train_size, 0, siamese=True, rsr=rsr)

    def objective(trial):
        batch_size = trial.suggest_categorical('batch_size', [32, 64, 128])
        n_transformations = trial.suggest_int('n_transformations', 1, 4)
        epochs = trial.suggest_int('epochs', 5, 50)
        learning_rate = trial.suggest_loguniform('learning_rate', 1e-6, 1)
        n_conv = trial.suggest_int('n_conv', 2, 6) # Number of convolutional layers
        conv_filters = [trial.suggest_int(f'conv_filters_{i}', 4, 64) for i in range(n_conv)]
        kernel = trial.suggest_int('kernel', 3, 5)
        max_pooling_x = trial.suggest_categorical('max_pooling_x', [2, 4])

        train_dataset.n_transformations = n_transformations
        test_dataset.n_transformations = n_transformations

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

        img1, _, _, _, _ = next(iter(train_loader))

        return -train_and_eval(img1.shape[3], img1.shape[2], device, learning_rate, epochs, train_loader, test_loader, train_dataset.font_to_label, 
            print_every_k_batches,mode,conv_filters=conv_filters, max_pooling=(2,max_pooling_x), kernel=kernel)

    study = optuna.create_study()
    study.optimize(objective, n_trials=n_trials)

    print(f"\n--- Finished trials ---\nBest params:\n{study.best_params}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--n_trials", type=int, default=20, 
        help="number of trials for hyperparameter optimization")
    parser.add_argument("-pkb", "--print_every_k_batches", type=int, default=2, 
        help="every k batch, the running loss is plot")
    parser.add_argument("-d", "--dataset", type=str, default="example", 
        help="dataset name in the data folder")
    parser.add_argument("-t", "--train_size", type=float, default=0.7, 
        help="dataset train size")
    parser.add_argument("-m", "--mode", type=int, default=0, 
        help="type of model: 0, 1, or 2")
    parser.add_argument("-rsr", "--rsr", type=bool, default=False, const=True, nargs="?",
        help="wether to use more important transformations (resize, shift, rotate)")
    
    args = parser.parse_args()
    print(f"\n> args:\n{json.dumps(vars(args), sort_keys=True, indent=4)}\n")
    main(args.dataset, args.n_trials, args.print_every_k_batches, args.train_size, args.mode, args.rsr)