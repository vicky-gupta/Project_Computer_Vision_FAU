from skimage import io, segmentation, color
import matplotlib.pyplot as plt
import numpy as np
import cv2
# Load the image
image = io.imread('C:/Users/vicky/Downloads/Project_Computer_Vision/Project_5/exercise5_material/code/data/arthist/annunciation1.jpg')

# Apply the Felzenszwalb segmentation algorithm
segments = segmentation.felzenszwalb(image, scale=100, sigma=0.5, min_size=100)

# Convert segments to RGB for visualization
segmented_image = color.label2rgb(segments, image, kind='avg')

    # Normalize the segments mask to be in the range [0, 255] (for alpha channel)
alpha_channel = np.uint8(255 * (segments - segments.min()) / (segments.max() - segments.min()))

    # Stack the alpha channel to the original image to create a 4-channel RGBA image
im_orig = np.dstack((image, alpha_channel))
#rgba_image = np.clip(im_orig, 0, 1)
def calc_colour_hist(image):
    """
    Task 2.5.1
    calculate colour histogram for each region
    the size of output histogram will be BINS * COLOUR_CHANNELS(3)
    number of bins is 25 as same as [uijlings_ijcv2013_draft.pdf]
    extract HSV
    """
    BINS = 25
    hist = np.array([])
    ### YOUR CODE HERE ###

    hsv_img = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

    for i in range(3):  # Loop over the three channels in HSV
        channel_hist, _ = np.histogram(hsv_img[:, :, i], bins=BINS, range=(0, 256), density=True)
        hist = np.concatenate((hist, channel_hist)) if hist.size else channel_hist


    return hist
hist = calc_colour_hist(image)

# Display the histogram
plt.plot(hist)
plt.title('Color Histogram')
plt.xlabel('Bin')
plt.ylabel('Frequency')

# Plot the original image and the segmented image
fig, ax = plt.subplots(1, 2, figsize=(12, 6))
ax[0].imshow(image)
ax[0].set_title('Original Image')
ax[0].axis('off')

ax[1].imshow(segmented_image)
ax[1].set_title('Segmented Image')
ax[1].axis('off')

plt.show()
