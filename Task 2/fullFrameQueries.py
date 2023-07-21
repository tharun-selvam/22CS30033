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
from rough import find_closest_centroids

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

    mat = scipy.io.loadmat(image_name)
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

    return histogram














