import copy

import cv2
import os

images_directory = '../train'
resize_factor = 0.3

# Load all images in folder
for entry in os.scandir(images_directory):
    if entry.is_file():
        # Load directly in B&W
        img = cv2.imread(entry.path, 0)

        # Downscale image to reduce computation requirements
        img = cv2.resize(
            img,
            None,
            fx=resize_factor,
            fy=resize_factor,
            interpolation=cv2.INTER_LINEAR
        )

        # Reduce noise to avoid false circle detection
        img = cv2.medianBlur(img, 5)

        circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 1, 100)

        if circles is None:
            print(f"{entry.name}: Error - No circle found")
            continue

        x, y, radius = max(circles[0], key=lambda circle: circle[2])
        center = (int(round(x)), int(round(y)))
        radius = int(round(radius))

        # For now save image
        cv2.circle(img, center, radius, 255, thickness=4)
        cv2.imwrite(f"../output/{entry.name}", img)

        print(f"{entry.name}: Processed")
