from pathlib import Path
import logging
import tomli
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
        self.path = file_dir

    def __len__(self):
        return len(self.file_index)
    
    def __getitem__(self, idx):
        x = np.load(self.path/self.file_index.iloc[idx].X)
        y = np.load(self.path/self.file_index.iloc[idx].Y)
        x_batch = torch.tensor(x, dtype=torch.float32).permute(2, 0, 1) # [channels, height, width]
        y_batch = torch.tensor(y, dtype=torch.float32).permute(2, 0, 1)
        return x_batch, y_batch
    
def load_data(file_dir, file_matrix, train_fraction):
    dataset = TestTrainData(file_dir, file_matrix)
    train_part = int(len(dataset) * train_fraction)
    train_set, test_set = random_split(dataset, [train_part, len(dataset) - train_part])

    return train_set, test_set

def train_model(img_size, in_channels, out_channels, embd_channels, patch_size,
                f_hidden, neighbourhood, n_layers, learning_rate, step_size,
                gamma, train_epochs, data_dir, data_set, device, **kwargs):

    n_patches = img_size // patch_size

    model = Img2ImgMixer(in_channels=in_channels,
                        out_channels = out_channels,
                        embd_channels=embd_channels, 
                        patch_size=patch_size, 
                        n_patches=n_patches, 
                        f_hidden=f_hidden, 
                        neighbourhood=neighbourhood, 
                        n_layers=n_layers)

    model.to(device)
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    parameters = sum([np.prod(p.size()) for p in parameters]) / 1_000_000
    logging.info(f"Model Parameters: {parameters:.3f}M")

    #loss_function = nn.MSELoss()
    loss_function = nn.L1Loss()
 
    optimiser = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimiser, step_size=step_size, gamma=gamma)

    path = Path(data_dir)

    train_set, test_set = load_data(path/data_set, path/"test_train_matrix.csv", 0.8)

    logging.info(f"Train items {len(train_set)}, Test items {len(test_set)}")


    train_loader = DataLoader(train_set,
                              batch_size=1, shuffle=True,
                              num_workers=0)

    test_loader = DataLoader(test_set,
                            batch_size=1, shuffle=True,
                            num_workers=0)
    
    model.train()

    for epoch in range(train_epochs):
        running_loss = 0
        for _, data in enumerate(train_loader, 0):
            x_batch, y_batch = data
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            optimiser.zero_grad(set_to_none=True)
            output = model(x_batch[:,0,:,:].unsqueeze(1))
#            output = model(x_batch)
            target = y_batch
            loss_p1 = loss_function(output[:, : ,300:700, 300:700], target[:, :, 300:700, 300:700])
            loss_p2 = loss_function(output, target)
            loss = 0.5*(loss_p1 + loss_p2)
            loss.backward()
            optimiser.step()

            running_loss += loss.item()
        logging.info(f"Epoch: {epoch}, Training loss: {running_loss/len(train_loader)}")
        scheduler.step()
        

        running_loss = 0
        for _, data in enumerate(test_loader, 0):
            with torch.no_grad():
                x_batch, y_batch = data
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                output = model(x_batch[:,0,:,:].unsqueeze(1))
#                output = model(x_batch)
                target = y_batch
                loss_p1 = loss_function(output[:, : ,300:700, 300:700], target[:, :, 300:700, 300:700])
                loss_p2 = loss_function(output, target)
                loss = 0.5*(loss_p1 + loss_p2)
            running_loss += loss.item()
        logging.info(f"Epoch: {epoch}, Validation loss: {running_loss/len(test_loader)}") 

    return model
        
def main():

    with open(Path("src/config.toml"), "rb") as c:
        config = tomli.load(c)

    path = Path(config["data_dir"])

    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"

    logging.info(f"Training model '{config['model_name']}'...")
    model = train_model(**config, device=device)
    logging.info(f"Finished training model '{config['model_name']}'")
    logging.info(f"Saving model '{config['model_name']}'")
    torch.save(model.state_dict(),  path / f"trained_models/{config['model_name']}.pt")
    logging.info(f"Model '{config['model_name']}' saved")

    # Generate test image.
    x = np.load(path / config["data_set"] / "162_0_geom.npy")
    x_batch = torch.tensor(x, dtype=torch.float32, device=device).permute(2, 0, 1).unsqueeze(0)
    model.eval()
    img = model(x_batch[:,0,:,:].unsqueeze(1))
#    img = model(x_batch)
    img = img.squeeze().permute(1,2,0).to("cpu").detach().numpy()
    np.save(path / f"validation_images/{config['model_name']}", img)
    
if __name__ == "__main__":
    main()
