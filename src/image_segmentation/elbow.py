import matplotlib.pyplot as plt
from skimage import io
from skimage.segmentation import slic
from skimage.color import label2rgb
from skimage import img_as_float
import numpy as np

def rgba2rgb( rgba, background=(255,255,255) ):
    row, col, ch = rgba.shape

    if ch == 3:
        return rgba

    assert ch == 4, 'RGBA image has 4 channels.'

    rgb = np.zeros( (row, col, 3), dtype='float32' )
    r, g, b, a = rgba[:,:,0], rgba[:,:,1], rgba[:,:,2], rgba[:,:,3]

    a = np.asarray( a, dtype='float32' ) / 255.0

    R, G, B = background

    rgb[:,:,0] = r * a + (1.0 - a) * R
    rgb[:,:,1] = g * a + (1.0 - a) * G
    rgb[:,:,2] = b * a + (1.0 - a) * B

    return np.asarray( rgb, dtype='uint8' )

# Load image (Ensure the image is in grayscale if that's the focus)
image = img_as_float(io.imread('slika.png', as_gray=True))
#print(image.shape)
#image = rgba2rgb(image)
#print(image.shape)

# Apply SLIC and obtain the segmented output, specifying the number of segments
segments = slic(image, n_segments=8, compactness=0.1, sigma=0, start_label=1, channel_axis=None, min_size_factor=0.01, max_size_factor=50, enforce_connectivity=True)

# Create a segmented image where each segment is colored based on the average color within it
segmented_image = label2rgb(segments, image, kind='avg')

# Display the results
fig, ax = plt.subplots(1, 2, figsize=(12, 6), sharex=True, sharey=True)
ax[0].imshow(image, cmap='gray')
ax[0].set_title('Original Image')
ax[0].axis('off')

ax[1].imshow(segmented_image, cmap='gray')
ax[1].set_title('Segmented Image by SLIC')
ax[1].axis('off')

plt.tight_layout()
plt.show()
