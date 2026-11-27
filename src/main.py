import copy

import cv2
import os
import csv
import numpy as np

images_directory = '../train'
max_width = 960
submission_filename = "../submission.csv"

with open('../train.csv') as f:
    truth = dict(csv.reader(f))
del truth['ID'] # drop the header row
predictions = {}
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
            # When we don't detect, we can at least try by luck
            predictions[entry.name] = "12:00"
            print(f"{entry.name}: Error - No circle found")
            continue

        x, y, radius = max(circles[0], key=lambda circle: circle[2])
        center = (int(round(x)), int(round(y)))
        radius = int(round(radius))

        # Save image with circle highlighted
        # cv2.circle(img, center, radius, 255, thickness=4)
        # cv2.imwrite(f"../output/{entry.name}", img)

        # Polar warp to 360 rows (1 row = 1 degree, clockwise from 3)
        warped = cv2.warpPolar(img, dsize=(radius,360), center=center, maxRadius=radius, flags=cv2.WARP_POLAR_LINEAR)

        # Save warped image
        cv2.imwrite(f"../output/{entry.name}", warped)

        # Crop to band region, apply binary threshold, invert if needed
        _, binary = cv2.threshold(warped, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        inner = int(0.15 * radius)
        outer = int(0.60 * radius)
        if binary[:, inner:outer].mean() > 127:  # dark face: flip
            binary = 255 - binary
        band = binary[:, inner:outer]

        # Compute profile (average brightness per degree)
        profile = band.mean(axis=1) / 255

        # Convolute profile
        profile = np.convolve(np.tile(profile, 3), np.ones(7) / 7, mode='same')[360:720]

        # Find brightest degrees (hands)
        first = np.argmax(profile)
        rows = np.arange(360)
        distance = np.abs(rows - first)
        distance = np.minimum(distance, 360 - distance)
        profile_without_first = np.where(distance > 10, profile, 0) # everything within 5° of first peak is 0
        second  = np.argmax(profile_without_first)

        # Minute hand is longer so more white beyond the band
        if binary[first, outer:].mean() > binary[second, outer:].mean():
            first, second = second, first


        minute_deg = (second + 90) % 360
        minutes = round(minute_deg / 6) % 60

        hour_deg = (first + 90) % 360
        hour_offshift = minutes / 60 # how far the hour handle is off based on the minutes
        hours = round(hour_deg / 30 - hour_offshift - 1) % 12 + 1 # shifts down by 1, wraps into 0-11 and shifts back up

        time = f"{hours:02d}:{minutes:02d}"
        predictions[entry.name] = time

        print(f"{entry.name}: predicted {time}, truth {truth[entry.name]}")

total = 0
for name, predicted in predictions.items():
    t = int(truth[name][:2]) * 60 + int(truth[name][3:])
    p = int(predicted[:2]) * 60 + int(predicted[3:])
    diff = abs(t - p) % 720
    diff = min(diff, 720 - diff)
    total += 1 if diff <= 2 else 0.8 if diff <= 10 else 0.5 if diff <= 30 else 0

with open(submission_filename, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["ID", "time"])
    writer.writerows(predictions.items())

print("score:", total / len(truth))