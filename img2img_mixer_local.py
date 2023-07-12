from pathlib import Path
import logging
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from model import Img2ImgMixer

logging.basicConfig(level=logging.INFO,
                    format="%(levelname)s %(message)s",
                    filename="info.log",
                    filemode='w',)

class TestTrainData(Dataset):
    def __init__(self, file_dir, file_index) -> None:
        self.root_dir = file_dir
        self.file_index = pd.read_csv(file_index)
        self.path = Path()/"spacio_training_2/processed"

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

    model.to(device)
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    parameters = sum([np.prod(p.size()) for p in parameters]) / 1_000_000
    logging.info(f"Model Parameters: {parameters:.3f}M")

    loss_function = nn.MSELoss()
    optimiser = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
    scheduler = torch.optim.lr_scheduler.StepLR(optimiser, step_size=config['step_size'], gamma=config['gamma'])

    path = Path()/data_dir

    train_set, test_set = load_data(path/"processed", path/"test_train_matrix_new.csv", 0.8)

    train_loader = DataLoader(train_set,
                              batch_size=1, shuffle=True,
                              num_workers=0)

    test_loader = DataLoader(test_set,
                             batch_size=1, shuffle=True,
                             num_workers=0)
    
    model.train()

    for epoch in range(20):
        running_loss = 0
        counter = 0
        for _, data in enumerate(train_loader, 0):
            x_batch, y_batch = data
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            optimiser.zero_grad(set_to_none=True)
            output = model(x_batch)
            target = torch.clamp(x_batch + y_batch, 0, 1) # Geometry + flowfield
            loss = loss_function(output, target)
            loss.backward()
            optimiser.step()

            running_loss += loss.item()
            counter += 1
        logging.info(f"Epoch: {epoch}, Training loss: {running_loss/counter}")
        scheduler.step()
        

        running_loss = 0
        counter = 0
        for _, data in enumerate(test_loader, 0):
            with torch.no_grad():
                x_batch, y_batch = data
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                output = model(x_batch)
                target = torch.clamp(x_batch + y_batch, 0, 1) # Geometry + flowfield
                loss = loss_function(output, target)
            running_loss += loss.item()
            counter += 1
        logging.info(f"Epoch: {epoch}, Validation loss: {running_loss/counter}")

    return model
        
def main():

    config = {'embd_channels': 256,
               'patch_size': 16,
               'f_hidden': 8,
               'neighbourhood': 7,
               'n_layers': 10,
               'learning_rate': 0.000105923,
               'gamma': 0.5,
               'step_size': 2,
               }
    
    # config = {'embd_channels': 256, 
    #           'patch_size': 16,
    #           'f_hidden': 8,
    #           'neighbourhood': 5, 
    #           'n_layers': 8, 
    #           'learning_rate': 0.00015431796010256462, 
    #           'gamma': 0.5, 
    #           'step_size': 2}
    
    logging.info(f"Training model 1...")
    model = train_model(config, "spacio_training_2")
    torch.save(model.state_dict(), Path()/"spacio_training_2/trained_models/local_new_data_20_epochs.pt")
    logging.info(f"Finished training model 1")

    # Generate test image.
    x = np.load(Path()/f"spacio_training_2/processed/93_0_geom.npy")
    x_batch = torch.tensor(x, device="cuda:0").permute(2, 0, 1).unsqueeze(0)
    model.eval()
    img = model(x_batch)
    img = img.squeeze().permute(1,2,0).to("cpu").detach().numpy()
    np.save(Path()/"spacio_training_2/validation_images/local_new_data_20_epochs.npy", img)
    

if __name__ == "__main__":
    main()