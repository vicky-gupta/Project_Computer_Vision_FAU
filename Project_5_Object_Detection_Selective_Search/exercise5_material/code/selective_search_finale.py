'''
@author: Prathmesh R Madhu.
For educational purposes only
'''
# -*- coding: utf-8 -*-
from __future__ import division

import skimage.io
import skimage.feature
import skimage.color
import skimage.transform
import skimage.util
import skimage.segmentation
import numpy as np
import cv2


def generate_segments(im_orig, scale, sigma, min_size):
    """
    Task 1: Segment smallest regions by the algorithm of Felzenswalb.
    1.1. Generate the initial image mask using felzenszwalb algorithm
    1.2. Merge the image mask to the image as a 4th channel
    """
    ### YOUR CODE HERE ###
    segments = skimage.segmentation.felzenszwalb(im_orig, scale, sigma, min_size)

    # Normalize the segments mask to be in the range [0, 255] (for alpha channel)
    alpha_channel = np.uint8(255 * (segments - segments.min()) / (segments.max() - segments.min()))

    # Stack the alpha channel to the original image to create a 4-channel RGBA image
    im_orig = np.dstack((im_orig, alpha_channel))

    #better results with color.label2rgb(segments, image, kind='avg')

    return im_orig


'''A histogram is a graphical representation of the distribution of numerical data. In the context of images,
a histogram represents the distribution of pixel intensities (color levels) in an image.'''


def sim_colour(r1, r2):
    #regions : r1, r2
    """
    2.1. calculate the sum of histogram intersection of colour
    """
    ### YOUR CODE HERE ##
    # do we need this again ? Archit. should it not just be
    # total intersection = np.sum(np.minimum(r1['color_hist', r2['color_hist']))
    hist_r1 = calc_colour_hist(r1)
    hist_r2 = calc_colour_hist(r2)

    # Calculate the histogram intersection 
    # Sum of the histogram intersection
    total_intersection = np.sum(np.minimum(hist_r1, hist_r2))

    return total_intersection


def sim_texture(r1, r2):
    """
    2.2. calculate the sum of histogram intersection of texture
    """
    # do we need this again ? Archit
    # total intersection = np.sum(np.minimum(r1['texture_hist', r2['texture_hist']))
    hist_r1 = calc_texture_hist(r1)
    hist_r2 = calc_texture_hist(r2)

    intersection = np.sum(np.minimum(hist_r1, hist_r2))

    return intersection


def sim_size(r1, r2, imsize):
    size_r1 = r1.shape[0] * r1.shape[1]
    size_r2 = r2.shape[0] * r2.shape[1]
    # Calculate size similarity using the given formula
    size_similarity = 1 - ((size_r1 + size_r2) / imsize)

    return size_similarity


def sim_fill(r1, r2, R_r1, R_r2,imsize):
    """
    2.4. calculate the fill similarity over the image
    """
    ### YOUR CODE HERE ###

    #get merged region bbox
    merged_bbox = calculate_bounding_box_merged(R_r1, R_r2)

    # Calculate the fill similarity using the given formula
    size_r1 = r1.shape[0] * r1.shape[1]
    size_r2 = r2.shape[0] * r2.shape[1]
    fill_similarity = 1 - ((merged_bbox - size_r1 - size_r2)/ imsize)

    return fill_similarity


def calc_sim(r1, r2, imsize, image, R):
    region_ar = image[R[r1]["min_y"]:R[r1]["max_y"], R[r1]["min_x"]:R[r1]["max_x"]]
    region_br = image[R[r2]["min_y"]:R[r2]["max_y"], R[r2]["min_x"]:R[r2]["max_x"]]

    similarity_colour = sim_colour(region_ar, region_br)
    similarity_texture = sim_texture(region_ar, region_br)
    similarity_size = sim_size(region_ar, region_br, imsize)

    similarity_fill = sim_fill(region_ar, region_br, R[r1], R[r2], imsize)

    return similarity_colour + similarity_texture + similarity_size + similarity_fill


def calc_colour_hist(img):
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
    hsv_img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    for i in range(3):  # Loop over the three channels in HSV
        channel_hist, _ = np.histogram(hsv_img[:, :, i], bins=BINS, range=(0, 256), density=True)
        hist = np.concatenate((hist, channel_hist)) if hist.size else channel_hist
        hist = hist / np.sum(hist)  #normalizing the histogram
    return hist


def calc_texture_gradient(img):
    """
    Task 2.5.2
    calculate texture gradient for entire image
    The original SelectiveSearch algorithm proposed Gaussian derivative
    for 8 orientations, but we will use LBP instead.
    output will be [height(*)][width(*)]
    Useful function: Refer to skimage.feature.local_binary_pattern documentation
    """
    ret = np.zeros((img.shape[0], img.shape[1], img.shape[2]))

    # Define LBP parameters
    radius = 1  # Radius of LBP
    n_points = 8 * radius  # Number of points to consider in LBP

    # Calculate LBP
    for i in range(3):  #for all three channels
        lbp = skimage.feature.local_binary_pattern(img[:, :, i], n_points, radius, method='uniform')
        ret[:, :, i] = lbp
    return ret


def calc_texture_hist(img):
    """
    Task 2.5.3
    calculate texture histogram for each region
    calculate the histogram of gradient for each colours
    the size of output histogram will be
        BINS * ORIENTATIONS * COLOUR_CHANNELS(3)
    Do not forget to L1 Normalize the histogram
    """
    BINS = 10
    hist = np.array([])
    ### YOUR CODE HERE ###

    ORIENTATIONS = 8
    COLOUR_CHANNELS = 3

    # Initialize histogram
    hist = np.zeros((COLOUR_CHANNELS, BINS * ORIENTATIONS))

    radius = 1
    n_points = ORIENTATIONS * radius  #as e\per the original documentation

    for channel in range(3):  #for all three channels

        # Extract the single color channel
        img_channel = img[:, :, channel]

        # Calculate LBP for the color channel
        lbp = skimage.feature.local_binary_pattern(img_channel, n_points, radius, method='uniform')

        # Calculate the histogram for the LBP
        max_bins = int(lbp.max() + 1)
        channel_hist, _ = np.histogram(lbp, bins=max_bins, range=(0, max_bins), density=True)

        # Normalize the histogram
        channel_hist = channel_hist / np.linalg.norm(channel_hist, ord=1)

        # Append the channel histogram to the overall histogram
        hist[channel, :len(channel_hist)] = channel_hist

    # Flatten the histogram array and L1 normalize the final histogram
    hist = hist.flatten()
    hist = hist / np.linalg.norm(hist, ord=1)

    return hist


#Calculate the bounding box of the merged region
def calculate_bounding_box_merged(r1, r2):
    min_x = min(r1["min_x"], r2["min_x"])
    min_y = min(r1["min_y"], r2["min_y"])
    max_x = max(r1["max_x"], r2["max_x"])
    max_y = max(r1["max_y"], r2["max_y"])

    # Calculate the size of the bounding box
    bounding_box_size = (max_x - min_x) * (max_y - min_y)
    return bounding_box_size

def calculate_bounding_boxes(img, regions):
    for y, row in enumerate(img):
        for x, pixel in enumerate(row):
            label = pixel[3]
            if label not in regions:
                regions[label] = {
                    "min_x": np.inf, "min_y": np.inf,
                    "max_x": 0, "max_y": 0, "labels": label
                }
            regions[label]["min_x"] = min(regions[label]["min_x"], x)
            regions[label]["min_y"] = min(regions[label]["min_y"], y)
            regions[label]["max_x"] = max(regions[label]["max_x"], x)
            regions[label]["max_y"] = max(regions[label]["max_y"], y)

            r1 = img[regions[label]["min_y"]:regions[label]["max_y"], regions[label]["min_x"]:regions[label]["max_x"]]
            regions[label]["size"] = r1.shape[0] * r1.shape[1]

    return regions


def calculate_histogram(img, regions):

    for label in regions:

        # Extract the region from the image
        region = img[regions[label]["min_y"]:regions[label]["max_y"],
                     regions[label]["min_x"]:regions[label]["max_x"]]

        # Calculate the color histogram
        color_hist = calc_colour_hist(region)

        # Calculate the texture histogram
        texture_hist = calc_texture_hist(region)

        regions[label]["color_hist"] = color_hist
        regions[label]["texture_hist"] = texture_hist

    return regions


def extract_regions(img):
    '''
    Task 2.5: Generate regions denoted as datastructure R
    - Convert image to hsv color map
    - Count pixel positions
    - Calculate the texture gradient
    - calculate color and texture histograms
    - Store all the necessary values in R.
    '''
    R = {}

    # Convert image to HSV color map
    hsv_img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

    # Count pixel positions
    pixel_positions = np.indices((img.shape[0], img.shape[1]))

    # Calculate the texture gradient
    texture_gradient = calc_texture_gradient(img)

    # Calculate bounding boxes
    R = calculate_bounding_boxes(img, R)

    # Calculate color and texture histograms
    R = calculate_histogram(img, R)

    # Store all the necessary values in R
    #R['hsv_img'] = hsv_img
    #R['pixel_positions'] = pixel_positions
    #R['texture_gradient'] = texture_gradient

    return R, hsv_img, pixel_positions, texture_gradient


def extract_neighbours(regions):
    def intersect(a, b):
        if (a["min_x"] < b["min_x"] < a["max_x"]
            and a["min_y"] < b["min_y"] < a["max_y"]) or (
                a["min_x"] < b["max_x"] < a["max_x"]
                and a["min_y"] < b["max_y"] < a["max_y"]) or (
                a["min_x"] < b["min_x"] < a["max_x"]
                and a["min_y"] < b["max_y"] < a["max_y"]) or (
                a["min_x"] < b["max_x"] < a["max_x"]
                and a["min_y"] < b["min_y"] < a["max_y"]):
            return True
        return False

    # Hint 1: List of neighbouring regions
    # Hint 2: The function intersect has been written for you and is required to check neighbours
    neighbours = []
    ### YOUR CODE HERE ###
    region_keys = list(regions.keys())

    # Iterate through all pairs of regions
    for i in range(len(region_keys)):
        for j in range(i + 1, len(region_keys)):
            r1 = regions[region_keys[i]]
            r2 = regions[region_keys[j]]
            try:
                if intersect(r1, r2):
                    neighbours.append((region_keys[i], region_keys[j]))
            except:
                print("Error in intersect function")
                pass

    return neighbours


def merge_regions(r1, r2):

    new_size = r1["size"] + r2["size"]
    rt = {}

    # Calculate new size
    rt["size"] = new_size

    # Calculate new bounding box
    rt["min_x"] = min(r1["min_x"], r2["min_x"])
    rt["min_y"] = min(r1["min_y"], r2["min_y"])
    rt["max_x"] = max(r1["max_x"], r2["max_x"])
    rt["max_y"] = max(r1["max_y"], r2["max_y"])
# Merge color histograms (weighted average)
    rt["color_hist"] = (r1["color_hist"] * r1["size"] + r2["color_hist"] * r2["size"]) / new_size

    # Merge texture histograms (weighted average)
    rt["texture_hist"] = (r1["texture_hist"] * r1["size"] + r2["texture_hist"] * r2["size"]) / new_size

    rt["labels"] = r1["labels"] + r2["labels"]

    return rt

def selective_search(image_orig, scale=1.0, sigma=0.8, min_size=50):
    '''
    Selective Search for Object Recognition" by J.R.R. Uijlings et al.
    :arg:
        image_orig: np.ndarray, Input image
        scale: int, determines the cluster size in felzenszwalb segmentation
        sigma: float, width of Gaussian kernel for felzenszwalb segmentation
        min_size: int, minimum component size for felzenszwalb segmentation

    :return:
        image: np.ndarray,
            image with region label
            region label is stored in the 4th value of each pixel [r,g,b,(region)]
        regions: array of dict
            [
                {
                    'rect': (left, top, width, height),
                    'labels': [...],
                    'size': component_size
                },
                ...
            ]
    '''

    # Checking the 3 channel of input image
    assert image_orig.shape[2] == 3, "Please use image with three channels."
    imsize = image_orig.shape[0] * image_orig.shape[1]

    # Task 1: Load image and get smallest regions. Refer to `generate_segments` function.
    image = generate_segments(image_orig, scale, sigma, min_size)

    if image is None:
        return None, {}

    # Task 2: Extracting regions from image
    # Task 2.1-2.4: Refer to functions "sim_colour", "sim_texture", "sim_size", "sim_fill"
    # Task 2.5: Refer to function "extract_regions". You would also need to fill "calc_colour_hist",
    # "calc_texture_hist" and "calc_texture_gradient" in order to finish task 2.5.
    R, hsv_img, pixel_positions, texture_gradient  = extract_regions(image)

    # Task 3: Extracting neighbouring information
    # Refer to function "extract_neighbours"
    neighbours = extract_neighbours(R)

    # Calculating initial similarities
    S = {}
    for (first, second) in neighbours:

        S[(first, second)] = calc_sim(first, second, imsize, image, R)

    # Hierarchical search for merging similar regions
    regions_to_remove = []
    while S != {}:
        # Get highest similarity
        i, j = sorted(S.items(), key=lambda i: i[1])[-1][0]

        # Task 4: Merge corresponding regions. Refer to function "merge_regions"
        t = max(R.keys()) + 1.0

        R[t] = merge_regions(R[i], R[j])

        # Task 5: Mark similarities for regions to be removed
        ### YOUR CODE HERE ###

        # Iterate over each item in the dictionary S
        for region_pair, similarity_value in S.items():
            # Check for i or j is in the current region pair
            if i in region_pair or j in region_pair:
                # If the condition is true, append the region pair to the list
                regions_to_remove.append(region_pair)



        # Task 6: Remove old similarities of related regions
        ### YOUR CODE HERE ###
        for k in regions_to_remove:
            del S[k]

        # Task 7: Calculate similarities with the new region
        ### YOUR CODE HERE ###
        for k in regions_to_remove:
            if k != (i, j):
                n = k[1] if k[0] in (i, j) else k[0]
                S[(t, n)] = calc_sim(t, n, imsize, image, R)
                            #calc_sim(first, second, imsize, image, R)

    # Task 8: Generating the final regions from R
    regions = []
    ### YOUR CODE HERE ###
    print("4.Generating final regions")
    regions = [{
        'rect': (r['min_x'], r['min_y'], r['max_x'] - r['min_x'], r['max_y'] - r['min_y']),
        'size': r['size'],
        'labels': r['labels']
    } for k, r in R.items()]

    return image, regions
