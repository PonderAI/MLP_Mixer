from pathlib import Path
import logging
from datetime import datetime
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from model import Img2ImgMixer

EPOCHS = 10

logging.basicConfig(level=logging.INFO,
                    format="%(levelname)s %(message)s",
                    filename="info.log",
                    filemode='w',)

class TestTrainData(Dataset):
    def __init__(self, file_dir, file_index) -> None:
        self.root_dir = file_dir
        self.file_index = pd.read_csv(file_index)
        self.path = Path()/"spacio_training_2/processed_actual_V"

    def __len__(self):
        return len(self.file_index)
    
    def __getitem__(self, idx):
        x = np.load(self.path/self.file_index.iloc[idx].X)
        y = np.load(self.path/self.file_index.iloc[idx].Y)
        x_batch = torch.tensor(x, dtype=torch.float32).permute(2, 0, 1)
        y_batch = torch.tensor(y, dtype=torch.float32).permute(2, 0, 1)
        return x_batch, y_batch
    
def load_data(file_dir, file_matrix, test_fraction):
    dataset = TestTrainData(file_dir, file_matrix)
    test_part = int(len(dataset) * test_fraction)
    train_set, test_set = random_split(dataset, [test_part, len(dataset) - test_part])

    return train_set, test_set

def train_model(config, data_dir=None):

    n_patches = 1024 // config['patch_size']

    model = Img2ImgMixer(in_channels=1,
                        out_channels=3,
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

    # loss_function = nn.MSELoss()
    loss_function = nn.L1Loss()

    optimiser = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
    scheduler = torch.optim.lr_scheduler.StepLR(optimiser, step_size=config['step_size'], gamma=config['gamma'])

    path = Path()/data_dir

    train_set, test_set = load_data(path/"processed_actual_V", path/"test_train_matrix.csv", 0.8)

    train_loader = DataLoader(train_set,
                              batch_size=1, shuffle=True,
                              num_workers=0)

    test_loader = DataLoader(test_set,
                             batch_size=1, shuffle=True,
                             num_workers=0)
    
    model.train()

    for epoch in range(EPOCHS):
        running_loss = 0
        counter = 0
        for _, data in enumerate(train_loader, 0):
            x_batch, y_batch = data
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            optimiser.zero_grad(set_to_none=True)
            output = model(x_batch[:, 0, :, :].unsqueeze(1))
            # target = torch.clamp(x_batch + y_batch, 0, 1) # Geometry + flowfield
            # geom = torch.stack([x_batch[:, 0, :, :]]*3, 1)
            # target = torch.clamp(geom + y_batch, -1, 1) # Geometry + flowfield
            target = y_batch #+ geom# Geometry + flowfield
            loss_p1 = loss_function(output[:,:,300:700, 300:700], target[:,:,300:700, 300:700])
            loss_p2 = loss_function(output+1, target+1)
            loss = 0.5*(loss_p2 + loss_p2)
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
                output = model(x_batch[:, 0, :, :].unsqueeze(1))
                # target = torch.clamp(x_batch + y_batch, 0, 1) # Geometry + flowfield
                # target = torch.clamp(geom + y_batch, -1, 1) # Geometry + flowfieldld
                target = y_batch #+ geom# Geometry + flowfield
                loss_p1 = loss_function(output[300:700, 300:700,:], target[300:700, 300:700,:])
                loss_p2 = loss_function(output+1, target+1)
                loss = 0.5*(loss_p2 + loss_p2)
            running_loss += loss.item()
            counter += 1
        logging.info(f"Epoch: {epoch}, Validation loss: {running_loss/counter}")

    return model
        
def main():

    config = {'embd_channels': 128,
               'patch_size': 8,
               'f_hidden': 4,
               'neighbourhood': 3,
               'n_layers': 12,
               'learning_rate': 0.00037435545642487277,
               'gamma': 0.2,
               'step_size': 5,
               }
    
    logging.info(f"Training model 1 for {EPOCHS} epochs")
    logging.info(f"Start {datetime.now()}")
    model = train_model(config, "spacio_training_2")
    logging.info(f"Finished training model 1")
    logging.info(f"Saving model")
    torch.save(model.state_dict(), Path()/"spacio_training_2/trained_models/new_model.pt")
    logging.info(f"End {datetime.now()}")

    # Generate test image.
    x = np.load(Path()/f"spacio_training_2/processed_actual_V/162_0_geom.npy")
    x_batch = torch.tensor(x, dtype=torch.float32, device="cuda:0").permute(2, 0, 1).unsqueeze(0)
    model.eval()
    img = model(x_batch)
    img = img.squeeze().permute(1,2,0).to("cpu").detach().numpy()
    np.save(Path()/"spacio_training_2/validation_images/new_model.npy", img)
    

if __name__ == "__main__":
    main()