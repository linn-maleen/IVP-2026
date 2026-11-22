import copy

import cv2
import os

images_directory = '../train'
max_width = 960

# Load all images in folder
for entry in os.scandir(images_directory):
    if entry.is_file():
        # Load directly in B&W
        img = cv2.imread(entry.path, cv2.IMREAD_GRAYSCALE)

        # Downscale image to reduce computation requirements
        height,width = img.shape
        scale = max_width/width
        if scale < 1:
            img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)

        # Reduce noise to avoid false circle detection
        img = cv2.medianBlur(img, 5)

        # Circle detection
        circles = cv2.HoughCircles(
            img,
            cv2.HOUGH_GRADIENT,
            1,
            minDist=max_width / 16,
            minRadius=int(max_width / 16),
            maxRadius=int(max_width / 2)
        )

        if circles is None:
            print(f"{entry.name}: Error - No circle found")
            continue

        x, y, radius = max(circles[0], key=lambda circle: circle[2])
        center = (int(round(x)), int(round(y)))
        radius = int(round(radius))

        # Save image with circle highlighted
        # cv2.circle(img, center, radius, 255, thickness=4)
        # cv2.imwrite(f"../output/{entry.name}", img)

        # Polar warp
        warped = cv2.warpPolar(img, dsize=(0,0), center=center, maxRadius=radius, flags=cv2.WARP_POLAR_LINEAR)

        # Save warped image
        cv2.imwrite(f"../output/{entry.name}", warped)

        # Detect hands ... (dilation, edge detect, Hough line detect, etc.)

        print(f"{entry.name}: Processed")
