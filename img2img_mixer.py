import torch
import numpy as np
import logging
from pathlib import Path
from model import Img2ImgMixer
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

#Hyperparameters
device = 'cuda' if torch.cuda.is_available() else 'cpu'
img_size = 1024
img_channels = 3 # RGB
batch_size = 1

logging.info(f"""DATASET
-------------------------------
Device: {device}
Batch size: {batch_size}
Image size: {img_size}
-------------------------------
-------------------------------
""")

dropout = 0.2
embd_channels = 256
patch_size = 16
n_layers = 10
f_hidden = 8
n_patches = img_size//patch_size
learning_rate = 1e-3
n_epochs = 10


path = Path("PyTorch_mixer/spacio_training_2")

#Initialise model and count parameters
model = Img2ImgMixer(img_channels, embd_channels, patch_size, n_patches, f_hidden, dropout, n_layers)
model.to(device)
parameters = filter(lambda p: p.requires_grad, model.parameters())
parameters = sum([np.prod(p.size()) for p in parameters]) / 1_000_000
logging.info(f"""MODEL PARAMETERS
-------------------------------
Trainable parameters: {parameters:.3f}
Patch size: {patch_size}
Embedding dimension: {embd_channels}
Dropout: {dropout}
Learning Rate: {learning_rate}
--------------------------------
--------------------------------""")

# optimiser = torch.optim.AdamW(model.parameters(), lr=learning_rate)
optimiser = torch.optim.SGD(model.parameters(), lr=learning_rate)

angles = np.arange(0, 360, 45)
samples = np.arange(93)
model.train()
for epoch in range(n_epochs):
    np.random.shuffle(samples)
    for sample in samples:
        np.random.shuffle(angles)
        for angle in angles:
            x = np.load(path/f"processed/{sample}_{angle}_geom.npy")
            y = np.load(path/f"processed/{sample}_U_{angle}_4.npy")

            x_batch = torch.tensor(x).permute(2, 0, 1).unsqueeze(0).to(device)
            y_batch = torch.tensor(y).permute(2, 0, 1).unsqueeze(0).to(device)

            loss, _ = model(x_batch, y_batch)
            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            optimiser.step()
        
    logging.info(f"Epoch {epoch+1}: loss = {loss.item():.3f}")


x = np.load(path/f"processed/93_0_geom.npy")
y = np.load(path/f"processed/93_U_0_4.npy")
x_batch = torch.tensor(x).permute(2, 0, 1).unsqueeze(0).to(device)
y_batch = torch.tensor(y).permute(2, 0, 1).unsqueeze(0).to(device)
model.eval()
_, img = model(x_batch, y_batch)
img = img.squeeze().permute(1,2,0).to("cpu").detach().numpy()
plt.imshow(img)
plt.show()
