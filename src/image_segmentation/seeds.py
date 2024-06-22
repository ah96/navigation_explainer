import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load image in color (SEEDS works with color images to optimize color homogeneity in superpixels)
image = cv2.imread('slika6.png')
if image is None:
    print("Error: Image not found")
else:
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Initialize SEEDS with the desired number of superpixels
    num_superpixels = 8  # Example number of superpixels
    num_levels = 4         # Number of block levels
    prior = 2              # 2 is typically enough for most images
    num_histogram_bins = 5 # Number of histogram bins
    seeds = cv2.ximgproc.createSuperpixelSEEDS(image.shape[1], image.shape[0], image.shape[2],
                                               num_superpixels, num_levels, prior, num_histogram_bins)

    # Apply SEEDS to the image
    seeds.iterate(image)

    # Retrieve the labels and the number of labels
    labels = seeds.getLabels()

    # Mask used to draw the contours of superpixels
    mask = seeds.getLabelContourMask(False)

    # Color background black where contours are
    image_rgb[mask == 255] = [0, 0, 0]

    # Display results
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(image_rgb)
    plt.title('Segmented Image by SEEDS')
    plt.axis('off')

    plt.show()
