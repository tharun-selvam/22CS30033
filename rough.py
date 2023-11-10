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

    ### START CODE HERE ###

    temp = np.zeros((X.shape[0], K), dtype=float)
    for i in range(X.shape[0]):
        # distance = []
        # for j in range(centroids.shape[0]):
        #     temp = X[i] - centroids[j]
        #     temp = temp**2
        #     distance.append(np.sum(temp))
        temp = np.expand_dims(X[i], axis=0)
        temp = np.tile(temp, (K, 1))
        temp = (temp - centroids) ** 2
        temp = np.sum(temp, axis=1)
        temp.squeeze()
        idx[i] = np.argmin(temp)

    ### END CODE HERE ###

    return idx


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

    ### START CODE HERE ###
    for i in range(K):
        # count = 0.0
        # for j in range(m):
        #     if (idx[j] == i):
        #         # print(centroids[i])
        #         centroids[i] = centroids[i] + X[j]
        #         # print(centroids[i])
        #         count += 1.0
        # # print(centroids[i])
        # centroids[i] = centroids[i] / count
        points = X[idx == i]
        centroids[i] = np.mean(points, axis=0)

    ### END CODE HERE ##

    return centroids


def run_kMeans(X, initial_centroids, max_iters=10):
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
        # Output progress
        print("K-Means iteration %d/%d" % (i, max_iters - 1))

        # For each example in X, assign it to the closest centroid
        # print(idx)
        idx = find_closest_centroids(X, centroids)

        # Given the memberships, compute new centroids
        centroids = compute_centroids(X, idx, K)
    return centroids, idx


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


# k-means clustering
K = 1500
max_iters = 100
initial_centroids = kMeans_init_centroids(updated_descriptors, K)
centroids, idx = run_kMeans(updated_descriptors, initial_centroids, max_iters)

np.save('centroids_updated.npy', centroids)
np.save('idx_updated.npy', idx)
