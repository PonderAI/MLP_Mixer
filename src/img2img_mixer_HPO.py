from pathlib import Path
from functools import partial
import logging
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import ray
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.optuna import OptunaSearch
from ray.air import session
from model import Img2ImgMixer

logging.basicConfig(level=logging.INFO,
                    format="%(levelname)s %(message)s",
                    filename="info.log",
                    filemode='w',)

logging.info("Test has started")
device = "cpu"
if torch.cuda.is_available():
    device = "cuda:0"
logging.info(f"Using device: {device}")
logging.info(f"{torch.cuda.device_count() > 1}")

ray.init(ignore_reinit_error=True, num_cpus=16)
class TestTrainData(Dataset):
    def __init__(self, file_dir, file_index) -> None:
        self.root_dir = file_dir
        self.file_index = pd.read_csv(file_index)
        self.path = Path.home()/"documents/Python Projects/MLP_Mixer/spacio_training_2/processed_with_features"

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

def train_model(config, data_dir=None):

    n_patches = 1024 // config['patch_size']

    model = Img2ImgMixer(in_channels=3,
                        out_channels=7,
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
    logging.info(f"Using device: {device}")

    model.to(device)

    loss_function = nn.MSELoss()
    optimiser = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
    scheduler = torch.optim.lr_scheduler.StepLR(optimiser, step_size=config['step_size'], gamma=config['gamma'])

    path = Path.home()/data_dir
    print(path.absolute())

    train_set, test_set = load_data(path/"processed_with_features", path/"test_train_matrix.csv", 0.8)

    train_loader = DataLoader(train_set,
                              batch_size=1, shuffle=True,
                              num_workers=0)

    test_loader = DataLoader(test_set,
                             batch_size=1, shuffle=True,
                             num_workers=0)
    
    model.train()

    for epoch in range(12):
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
        session.report({"loss": loss.item()})
        # session.report({"loss": loss.cpu().numpy()})

def main(num_samples=1, cpus_per_trial=16, gpus_per_trial=1, max_num_epochs=10):

    # Search space
    config = {'embd_channels': tune.choice([128, 256, 512,]),
              'patch_size': tune.choice([4, 8, 16, 32,]),
              'f_hidden': tune.choice([4, 8, 16]),
              'neighbourhood': tune.choice([5, 7, 9, 11,]),
              'n_layers': tune.choice([8, 10, 12,]),
              'learning_rate': tune.loguniform(1e-4, 1e-1),
              'gamma': tune.choice([0.2, 0.3, 0.5,]),
              'step_size': tune.choice([2, 5,]),
              }

    # config = {'embd_channels': tune.choice([256,]),
    #           'patch_size': tune.choice([16,]),
    #           'f_hidden': tune.choice([8,]),
    #           'neighbourhood': tune.choice([7,]),
    #           'n_layers': tune.choice([8,]),
    #           'learning_rate': tune.loguniform(1e-4, 1e-1),
    #           'gamma': tune.choice([0.2, 0.3, 0.5,]),
    #           'step_size': tune.choice([2, 5,]),
    #           }
    
    # Resources allocated per trial of the trainable function.
    # For a 16 core machine if cpu is set to 2 then 8 concurrent trials are possible.
    trainable_resources = tune.with_resources(partial(train_model, data_dir="documents/Python Projects/MLP_Mixer/spacio_training_2"),
                                              resources={"cpu": cpus_per_trial, 
                                                         "gpu": gpus_per_trial,},)
    
    known_model = [{'embd_channels': 256,
                    'patch_size': 16,
                    'f_hidden': 8,
                    'neighbourhood': 7,
                    'n_layers': 8,
                    'learning_rate': 0.000105923,
                    'gamma': 0.5,
                    'step_size': 2,
                    }]
    
    search_alg = OptunaSearch(metric="loss",
                              mode="min",
                              points_to_evaluate=known_model,)
    
    scheduler = ASHAScheduler(max_t=max_num_epochs,
                              grace_period=1,
                              reduction_factor=2,)
    
    tuner = tune.Tuner(trainable_resources,
                       tune_config=tune.TuneConfig(search_alg=search_alg,
                                                   metric="loss",
                                                   mode="min",
                                                   scheduler=scheduler,
                                                   num_samples=num_samples,),
                        param_space=config,)

    results = tuner.fit()
    print(results.get_best_result("loss", "min"))

    best_trial = results.get_best_result("loss", "min", "last")
    print(f"Best trial config: {best_trial.config}")

if __name__ == "__main__":
    main()

# Tensor board
# Saving best model