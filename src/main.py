import cv2
from matplotlib import pyplot as plt
import os

images_directory = '../train'

images = []

# Load all images in folder
for entry in os.scandir(images_directory):
    if entry.is_file():
        # Load directly in B&W
        img = cv2.imread(entry.path, 0)
        images.append(img)
