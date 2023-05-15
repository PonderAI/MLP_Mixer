import gc
import logging
from pathlib import Path
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
from model import Img2ImgMixer
from model_config import model_parameters
import pandas as pd
from torch.utils.data import Dataset, DataLoader, random_split
from ray.tune.search.bayesopt import BayesOptSearch

class TestTrainData(Dataset):
    def __init__(self, file_dir, file_index) -> None:
        self.root_dir = file_dir
        self.file_index = pd.read_csv(file_index)
        self.path = Path.home()/"documents/Python Projects/MLP_Mixer/spacio_training_2/processed_with_corner_mask"

    def __len__(self):
        return len(self.file_index)
    
    def __getitem__(self, idx):
        x = np.load(self.path/self.file_index.iloc[idx].X)
        y = np.load(self.path/self.file_index.iloc[idx].Y)
        x_batch = torch.tensor(x).permute(2, 0, 1)
        y_batch = torch.tensor(y).permute(2, 0, 1)
        return x_batch, y_batch
    
def load_data(file_dir, file_matrix, test_fraction):
    dataset = TestTrainData(file_dir, file_matrix)
    test_part = int(len(dataset) * test_fraction)
    train_set, test_set = random_split(dataset, [test_part, len(dataset) - test_part])

    return train_set, test_set

def train_model(config, checkpoint_dir=None, data_dir=None):

    n_patches = 1024 // config['patch_size']

    model = Img2ImgMixer(img_channels=3,
                        embd_channels=config['embd_channels'], 
                        patch_size=config['patch_size'], 
                        n_patches=n_patches, 
                        f_hidden=config['f_hidden'], 
                        neighbourhood=int(config['neighbourhood']), 
                        n_layers=config['n_layers'])
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
    model.to(device)

    loss_function = nn.MSELoss()
    optimiser = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
    scheduler = torch.optim.lr_scheduler.StepLR(optimiser, step_size=config['step_size'], gamma=config['gamma'])

    path = Path.home()/"documents/Python Projects/MLP_Mixer/spacio_training_2"
    print(path.absolute())

    train_set, test_set = load_data(path/"processed_with_corner_mask", path/"test_train_matrix.csv", 0.8)

    train_loader = DataLoader(train_set,
                              batch_size=1, shuffle=True,
                              num_workers=0)

    test_loader = DataLoader(test_set,
                             batch_size=1, shuffle=True,
                             num_workers=0)
    
    model.train()

    for epoch in range(config['n_epochs']):
        for i, data in enumerate(train_loader, 0):
            x_batch, y_batch = data
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            optimiser.zero_grad(set_to_none=True)
            output = model(x_batch)
            target = torch.clamp(x_batch + y_batch, 0, 1) # Geometry + flowfield
            loss = loss_function(output, target)
            loss.backward()
            optimiser.step()
        scheduler.step()

        for i, data in enumerate(test_loader, 0):
            with torch.no_grad():
                x_batch, y_batch = data
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                output = model(x_batch)
                target = torch.clamp(x_batch + y_batch, 0, 1) # Geometry + flowfield
                loss = loss_function(output, target)

def main(num_samples=10, gpus_per_trial=1, max_num_epochs=1):

    # BayesOpt search space
    config = {'embd_channels': tune.choice([256,]),
              'patch_size': tune.choice([16,]),
              'f_hidden': tune.choice([8,]),
              'neighbourhood': tune.choice([7,]),
              'n_layers': tune.choice([8,]),
              'learning_rate': tune.loguniform(1e-4, 1e-1),
              'gamma': tune.choice([0.5,]),
              'step_size': tune.choice([5,]),
              'n_epochs': tune.choice([10,]),
              }

    scheduler = ASHAScheduler(max_t=max_num_epochs,
                              grace_period=1,
                              reduction_factor=2)
    
    tuner = tune.Tuner(
                tune.with_resources(
                    tune.with_parameters(train_model),
                    resources={"cpu": 2, "gpu": gpus_per_trial}),
                    tune_config=tune.TuneConfig(search_alg=BayesOptSearch(),
                                                metric="loss",
                                                mode="min",
                                                scheduler=scheduler,
                                                num_samples=num_samples,),
                param_space=config,)

    results = tuner.fit()
    print(results.get_best_result("loss", "min"))

if __name__ == "__main__":
    main()
# embd_channels, patch_size, n_patches, f_hidden, neighbourhood, n_layers, learning rate, gamma, step


