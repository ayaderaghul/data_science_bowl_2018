
import numpy as np
from scipy import ndimage

from scipy.ndimage import label


def calculate_weight(mask, height, width):
    w0 = 10
    q = 5
    weight = np.zeros((height, width, 1), dtype=np.uint8)
    
    # calculate weight for important pixels
    distances = np.array(
        [ndimage.distance_transform_edt(m == 0) for m in masks])
    shortest_dist = np.sort(distances, axis=0)
    # distance to the border of the nearest cell
    d1 = shortest_dist[0]
    # distance to the border of the second nearest cell
    d2 = shortest_dist[1] if len(shortest_dist) > 1 else np.zeros(d1.shape)

    weight = w0 * np.exp(-(d1 + d2)**2 / (2 * q**2)).astype(np.float32)
    # weight = (mask == 0) * weight
    return weight