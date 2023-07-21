import numpy as np
import scipy.io
import glob
from scipy import misc
import matplotlib.pyplot as plt
from displaySIFTPatches import displaySIFTPatches
from selectRegion import roipoly
from getPatchFromSIFTParameters import getPatchFromSIFTParameters
from skimage.color import rgb2gray
import matplotlib.cm as cm
from skimage import io
import pylab as pl
# from rough import find_closest_centroids

idx = np.load('dataFiles/idx.npy')
centroids = np.load('dataFiles/centroids.npy')

framesdir = 'frames/'
siftdir = 'sift/'
fnames = glob.glob(siftdir + '*.mat')
fnames = [i[-27:] for i in fnames]


def build_histogram_of_image(image_name, visual_words):
    """
        image name (string): represents the name of the image
        visual_words (ndarray (K, )): represents the centers of visual_words
    """

    image_path = siftdir + image_name
    mat = scipy.io.loadmat(image_path)
    X = mat['descriptors']

    K = visual_words.shape[0]
    histogram = np.zeros(K, dtype=int)

    temp = np.zeros((X.shape[0], K), dtype=float)
    for i in range(X.shape[0]):
        distance = []
        for j in range(visual_words.shape[0]):
            temp = X[i] - visual_words[j]
            temp = temp**2
            distance.append(np.sum(temp))
        a = np.argmin(distance)
        histogram[a] += 1

    normalised_histogram = histogram / np.sum(histogram)

    return histogram, normalised_histogram


# histogram_combined = np.empty((1, centroids.shape[0]), dtype=float)
# normalised_histogram_combined = np.empty((1, centroids.shape[0]), dtype=float)
#
# for i in range(len(fnames)):
#
#     if i % 10 == 0:
#         print(f'{i} images completed of {len(fnames)}')
#
#     histogram, normalised_histogram = build_histogram_of_image(fnames[i], centroids)
#
#     histogram = np.expand_dims(histogram, axis=0)
#     normalised_histogram = np.expand_dims(normalised_histogram, axis=0)
#
#     histogram_combined = np.concatenate((histogram_combined, histogram), axis=0)
#     normalised_histogram_combined = np.concatenate((normalised_histogram_combined, normalised_histogram), axis=0)
#
# histogram_combined = np.delete(histogram_combined, 0, 0)
# normalised_histogram_combined = np.delete(normalised_histogram_combined, 0, 0)

# np.save('histogram_images_combined', histogram_combined)
# np.save('normalised_histogram_images_combined', normalised_histogram_combined)

histogram_images_combined = np.load('histogram_images_combined.npy')
normalised_histogram_images_combined = np.load('normalised_histogram_images_combined.npy')

print(normalised_histogram_images_combined.shape)


