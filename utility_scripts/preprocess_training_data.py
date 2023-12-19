import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from scipy.ndimage import rotate

path = Path("spacio_training_2")
new_path = path/"processed_actual_V"
Path.mkdir(new_path, exist_ok=True)
raw_data = os.listdir(path/"spacio_training")
NUM_SAMPLES = max([int(file_name.split("_")[0]) for file_name in raw_data]) +1

def rotate_image(img, angle, cnr_mask):

    theta = np.deg2rad(angle)
    mask = np.all(img <= [0.003921569,0.003921569,0.003921569], axis=-1)
    img -= 0.5
    img[:,:,0], img[:,:,1] = img[:,:,0] * np.cos(theta) + img[:,:,1] * -np.sin(theta), img[:,:,0] * np.sin(theta) + img[:,:,1] * np.cos(theta)
    img += 0.5
    img[mask] = [0.5, 0.5, 0.5]
    img[cnr_mask] = [0.5, 0.5, 0.5]
    img = rotate(img, angle=angle, reshape=False)

    return img

def mirror_image(img, cnr_mask):

    img = np.fliplr(img)
    mask = np.all(img <= [0.003921569,0.003921569,0.003921569], axis=-1)
    img[:,:,0] = ((img[:,:,0] - 0.5) * -1)+0.5
    img[mask] = [0.5, 0.5 ,0.5]
    img[cnr_mask] = [0.5, 0.5, 0.5]

    return img

angles = np.arange(0, 360, 45)
cnr_mask = np.load(path / 'corner_mask.npy')

for sample in range(NUM_SAMPLES):
    for angle in angles:
        img = mpimg.imread(path / f"spacio_training/{sample}_U_{angle}_2.png")
        img = rotate_image(img, angle, cnr_mask)
        np.save(new_path/ f"{sample}_U_{angle}_2", img)

        geo = mpimg.imread(path/f"spacio_training/{sample}_geom.png")
        geo = rotate(geo, angle=angle, reshape=False)
        np.save(new_path/ f"{sample}_{angle}_geom", geo)

for sample in range(NUM_SAMPLES):
    for angle in angles:
        img = np.load(new_path/f"{sample}_U_{angle}_2.npy")
        img = mirror_image(img, cnr_mask)
        np.save(new_path/ f"{sample+NUM_SAMPLES}_U_{angle}_2.npy", img)

        geo = np.load(new_path/f"{sample}_{angle}_geom.npy")
        geo = np.fliplr(geo)
        np.save(new_path/ f"{sample+NUM_SAMPLES}_{angle}_geom.npy", geo)  