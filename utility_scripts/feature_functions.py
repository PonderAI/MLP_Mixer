import numpy as np
from pathlib import Path
path = Path("spacio_training_2")

def pos_features(img):
    img_size = img.shape[0]

    ones = np.ones([img_size,img_size])
    v_vec = np.arange(0,img_size).reshape(img_size,1)
    h_vec = np.arange(0,img_size).reshape(1,img_size)
    v_pos = ones*v_vec
    h_pos = ones*h_vec

    return np.dstack([v_pos, h_pos])/img_size

def nearest_building_distance(img):
    # Find the distance to the nearest building from the left
    img_size = img.shape[0]
    result = img * 0

    for row in range(img.shape[0]):
        e = -1
        for col, val in enumerate(img[row]):

            # Check if building in current position
            if val >= 1e-3:
                e = 0
                
            # Check if building has already been seen and no building in current position
            if e != -1 and val < 1e-3:
                e += 1

            result[row, col] = e

    return result/img_size

def generate_features(geometry):

    img = np.load(path/f"processed/{geometry}")
    img = img[:,:,0]

    vh_pos = pos_features(img)

    nearest_building_left = nearest_building_distance(img)

    nearest_building_right = nearest_building_distance(np.flip(img, 1))
    nearest_building_right = np.flip(nearest_building_right, 1)

    nearest_building_up = nearest_building_distance(np.rot90(img))
    nearest_building_up = np.rot90(nearest_building_up, -1)

    nearest_building_down = nearest_building_distance(np.rot90(img, -1))
    nearest_building_down = np.rot90(nearest_building_down)

    feature_array = np.dstack([img, vh_pos, nearest_building_up, nearest_building_down, nearest_building_left, nearest_building_right])

    np.save(path/f"processed_with_features/{geometry}", feature_array)