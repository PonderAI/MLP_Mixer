import gc
import logging
from pathlib import Path
import torch
import numpy as np
import matplotlib.pyplot as plt
from model import Img2ImgMixer
from model_config import model_parameters

logging.basicConfig(level=logging.INFO,
                    format="%(levelname)s %(message)s",
                    filename="info.log",
                    filemode='w',)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
logging.info(f"""Device: {device}""")

# Training Parameters
n_epochs = 10
learning_rate = 1e-3

path = Path("spacio_training_2")
Path.mkdir(path / 'trained_models', exist_ok=True)
Path.mkdir(path / 'validation_images', exist_ok=True)


for i, parameter_set in enumerate(model_parameters):

    #Initialise model and count parameters
    model = Img2ImgMixer(**parameter_set)
    model.to(device)
    optimiser = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    parameters = sum([np.prod(p.size()) for p in parameters]) / 1_000_000

    logging.info(f"Running model {i+1}")
    logging.info(f"""MODEL PARAMETERS
    -------------------------------
    Trainable parameters: {parameters:.3f}M
    Optimiser: {optimiser.__class__.__name__}
    Dropout: {parameter_set["dropout"]}
    Embedding dimension: {parameter_set["embd_channels"]}
    Patch size: {parameter_set["patch_size"]}
    Layers: {parameter_set["n_layers"]}
    Linear hidden expansion: {parameter_set["f_hidden"]}
    Number of patches: {parameter_set["n_patches"]}
    Learning Rate: {learning_rate}
    --------------------------------
    --------------------------------""")

    # Training loop
    angles = np.arange(0, 360, 45)
    samples = np.arange(93)
    model.train()

    try:
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

    except RuntimeError as e:
        logging.error(e)
        logging.error(f"""Not enough memory to run model_{i+1}
        
        """)
        del model
        gc.collect()
        torch.cuda.empty_cache()
        continue

    torch.save(model.state_dict(), path/f"trained_models/model_{i+1}.pt")

    # Generate validation image
    x = np.load(path/f"processed/93_0_geom.npy")
    y = np.load(path/f"processed/93_U_0_4.npy")
    x_batch = torch.tensor(x).permute(2, 0, 1).unsqueeze(0).to(device)
    y_batch = torch.tensor(y).permute(2, 0, 1).unsqueeze(0).to(device)
    model.eval()
    _, img = model(x_batch, y_batch)
    img = img.squeeze().permute(1,2,0).to("cpu").detach().numpy()
    plt.imshow(img)
    plt.savefig(path/f'validation_images/model_{i+1}.png')

    # Clean up
    del model, _
    gc.collect()
    torch.cuda.empty_cache()

    logging.info(f"""Finished running model {i+1}
    
    """)

