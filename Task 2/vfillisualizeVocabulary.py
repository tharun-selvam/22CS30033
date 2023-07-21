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

# running the k-means clustering algorithm
framesdir = 'frames/'
siftdir = 'sift/'

# Get a list of all the .mat file in that directory. There is one .mat file per image.
# Preparation of data
fnames = glob.glob(siftdir + '*.mat')
fnames = [i[-27:] for i in fnames]

fname1 = siftdir + fnames[0]
mat1 = scipy.io.loadmat(fname1)

X_data = mat1['descriptors']

count = mat1['descriptors'].shape[0]
image_decriptors_count_list = [count]
for i in range(1, 5):

    if (i%10 == 0):
        print('reading frame %d of %d' %(i, len(fnames)))

    # load that file
    fname = siftdir + fnames[i]
    mat = scipy.io.loadmat(fname)

    numfeats = mat['descriptors'].shape[0]
    count += numfeats
    image_decriptors_count_list.append(count)

    X = mat['descriptors']

    X_data = np.concatenate((X_data, X), axis=0)

print(X_data.shape, count)

# Building the required functions
# UNQ_C1
# GRADED FUNCTION: find_closest_centroids


def find_closest_centroids(X, centroids):
    """
    Computes the centroid memberships for every example

    Args:
        X (ndarray): (m, n) Input values
        centroids (ndarray): (K, n) centroids

    Returns:
        idx (array_like): (m,) closest centroids

    """

    # Set K
    K = centroids.shape[0]

    # You need to return the following variables correctly
    idx = np.zeros(X.shape[0], dtype=int)

    temp = np.zeros((X.shape[0], K), dtype=float)
    for i in range(X.shape[0]):
        for j in range(K):
            a = X[i] - centroids[j]
            a = a**2
            sum = np.sum(a)
            temp[i][j] = sum
        idx[i] = np.argmin(temp[i])

    return idx


# UNQ_C2
# GRADED FUNCTION: compute_centroids


def compute_centroids(X, idx, K):
    """
    Returns the new centroids by computing the means of the
    data points assigned to each centroid.

    Args:
        X (ndarray):   (m, n) Data points
        idx (ndarray): (m,) Array containing index of closest centroid for each
                       example in X. Concretely, idx[i] contains the index of
                       the centroid closest to example i
        K (int):       number of centroids

    Returns:
        centroids (ndarray): (K, n) New centroids computed
    """

    # Useful variables
    m, n = X.shape

    # You need to return the following variables correctly
    centroids = np.zeros((K, n), dtype=float)

    for i in range(K):
        count = 0
        for j in range(m):
            if (idx[j] == i):
                centroids[i] += X[j]
                count+=1
        centroids[i]/=count

    return centroids


def kMeans_init_centroids(X, K):
    """
    This function initializes K centroids that are to be
    used in K-Means on the dataset X

    Args:
        X (ndarray): Data points
        K (int):     number of centroids/clusters

    Returns:
        centroids (ndarray): Initialized centroids
    """

    # Randomly reorder the indices of examples
    randidx = np.random.permutation(X.shape[0])

    # Take the first K examples as centroids
    centroids = X[randidx[:K]]

    return centroids


def run_kMeans(X, initial_centroids, max_iters=10, plot_progress=False):
    """
    Runs the K-Means algorithm on data matrix X, where each row of X
    is a single example
    """

    # Initialize values
    m, n = X.shape
    K = initial_centroids.shape[0]
    centroids = initial_centroids
    previous_centroids = centroids
    idx = np.zeros(m)
    plt.figure(figsize=(8, 6))

    # Run K-Means
    for i in range(max_iters):

        #Output progress
        print("K-Means iteration %d/%d" % (i, max_iters-1))

        # For each example in X, assign it to the closest centroid
        idx = find_closest_centroids(X, centroids)

        # Given the memberships, compute new centroids
        centroids = compute_centroids(X, idx, K)
    return centroids, idx


# K = 50
# max_iters = 200
#
# initial_centroids = kMeans_init_centroids(X_data, K)
#
# centroids, idx = run_kMeans(X_data, initial_centroids, max_iters)
#
# print(image_decriptors_count_list, X_data.shape, idx)
#
# count = 0
# for i in range(len(idx)):
#     if idx[i] == 0 and i < 1042:
#         print(i)
#         count += 1
#
#         # for j in range(len(image_decriptors_count_list)):
#         #     if i < image_decriptors_count_list[j]:
#         #         img_num = j
#         #         if j >= 1:
#         #             patch_num = i - image_decriptors_count_list[j-1]
#         #         else:
#         #             patch_num = i
#
#         fname = siftdir + fnames[0]
#         mat = scipy.io.loadmat(fname)
#
#         patch_num = i
#
#         imname = framesdir + fnames[0][:-4]
#         im = io.imread(imname)
#         img_patch = getPatchFromSIFTParameters(mat['positions'][patch_num,:], mat['scales'][patch_num], mat['orients'][patch_num], rgb2gray(im))
#         print(mat['positions'][patch_num, :])
#         plt.imshow(img_patch, cmap=cm.Greys_r)
#         plt.show()
#
#         print('imname = %s contains %d total features, each of dimension %d' %(imname, numfeats, mat['descriptors'].shape[1]))
#         fig=plt.figure()
#         ax=fig.add_subplot(111)
#         ax.imshow(im)
#         coners = displaySIFTPatches(mat['positions'][patch_num:patch_num+1,:], mat['scales'][patch_num:patch_num+1,:], mat['orients'][patch_num:patch_num+1,:])
#
#         for j in range(len(coners)):
#             ax.plot([coners[j][0][1], coners[j][1][1]], [coners[j][0][0], coners[j][1][0]], color='g', linestyle='-', linewidth=1)
#             ax.plot([coners[j][1][1], coners[j][2][1]], [coners[j][1][0], coners[j][2][0]], color='g', linestyle='-', linewidth=1)
#             ax.plot([coners[j][2][1], coners[j][3][1]], [coners[j][2][0], coners[j][3][0]], color='g', linestyle='-', linewidth=1)
#             ax.plot([coners[j][3][1], coners[j][0][1]], [coners[j][3][0], coners[j][0][0]], color='g', linestyle='-', linewidth=1)
#         ax.set_xlim(0, im.shape[1])
#         ax.set_ylim(0, im.shape[0])
#         plt.gca().invert_yaxis()
#
#         plt.show()
#

# print(count)

fname = siftdir + fnames[0]
mat = scipy.io.loadmat(fname)

print(mat['scales'].shape, '\n', mat['scales'], np.min(mat['scales']), np.max(mat['scales']))

patch_num = []


for i in range(len(mat['scales'])):
    a = mat['scales'][i]
    if a < 3 and a > 1:
        print(f'Scale: {a} | Index: {i}')
        patch_num = i
        imname = framesdir + fnames[0][:-4]
        im = io.imread(imname)
        print('imname = %s contains %d total features, each of dimension %d' %(imname, numfeats, mat['descriptors'].shape[1]))
        fig=plt.figure()
        ax=fig.add_subplot(111)
        ax.imshow(im)
        coners = displaySIFTPatches(mat['positions'][patch_num:patch_num+1,:], mat['scales'][patch_num:patch_num+1,:], mat['orients'][patch_num:patch_num+1,:])

        for j in range(len(coners)):
            ax.plot([coners[j][0][1], coners[j][1][1]], [coners[j][0][0], coners[j][1][0]], color='g', linestyle='-', linewidth=1)
            ax.plot([coners[j][1][1], coners[j][2][1]], [coners[j][1][0], coners[j][2][0]], color='g', linestyle='-', linewidth=1)
            ax.plot([coners[j][2][1], coners[j][3][1]], [coners[j][2][0], coners[j][3][0]], color='g', linestyle='-', linewidth=1)
            ax.plot([coners[j][3][1], coners[j][0][1]], [coners[j][3][0], coners[j][0][0]], color='g', linestyle='-', linewidth=1)
        ax.set_xlim(0, im.shape[1])
        ax.set_ylim(0, im.shape[0])
        plt.gca().invert_yaxis()

        plt.show()




