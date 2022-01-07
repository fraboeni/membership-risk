import numpy as np
from foolbox.distances import LpDistance
from scipy.stats import wasserstein_distance


def get_data_by_class(c, X, y):
    """
    :param c: class index
    :param X: features of the data
    :param y: lables of the data
    :return: returns all data that belongs to class c and the corresponding labels
    """
    indices = y == c
    X_ret = X[indices]
    y_ret = y[indices]
    return X_ret, y_ret


def evaluate_per_class(X, y, model):
    """ Evaluates prediction accuracy on all data in X per class"""
    for cl in np.unique(y):
        X_class, y_class = get_data_by_class(cl, X, y)
        performance = model.evaluate(X_class, y_class, verbose=0)
        print("Cl: ", cl, " Acc: ", performance[1])


def find_correctly_classified(X, y, model):
    """Given some data and labels, use the model to identify the correctly classified ones
  return them"""
    y_pred = np.argmax(model.predict(X), axis=1)
    indices = y == y_pred
    X_ret = X[indices]
    y_ret = y[indices]
    return X_ret, y_ret


def l2_dist(a, b):
    """Returns l2 distance of two points a and b"""
    return np.linalg.norm(a - b)


def generate_neighboring_points(data_sample, amount, epsilon=None, noise_vectors=None, scale=0.1):
    """
    This function is given a data point data_sample and it generates amount many data points in an epsilon neighborhood
    scale is given to the noise function to sample the random noise
    :param data_sample: the data point for which we want to generate neighbors
    :param amount: how many neighbors to generate
    :param epsilon: how far the neighbors should be away in an L2 norm, if not specified, no further clipping is performed
    :param noise_vectors: if we want deterministic noise vectors for neighbor creation, we can pass them here
    :param scale: the sigma for the normal noise distribution
    :return: an array of the generated neighbors all in range [0,1]
    """
    l2_dist = LpDistance(2)
    shape = (amount,) + data_sample.shape  # make a shape of [amount, ax1, ax2, ax3]

    data_sample = np.expand_dims(data_sample, 0)  # reshape for broadcast operations

    # generate noisy samples
    if noise_vectors is not None:  # in case we do need deterministic neighbors they have to be passed
        noises = noise_vectors
    else:
        noises = np.random.normal(loc=0, scale=scale, size=shape)

    # now make neighbors
    perturbed_samples = noises + data_sample  # add data to the noise
    perturbed_samples_clip = np.clip(perturbed_samples, 0, 1)  # bring them to range 0,1

    # and make sure that perturbation does not exceed epsilon
    if epsilon is not None:
        repeated_data = np.repeat(data_sample, amount, axis=0)
        perturbed_samples_clip = l2_dist.clip_perturbation(repeated_data, perturbed_samples_clip, epsilon)

        # make sure that we meet the required L2 distance and are not smaller
        l2_distances = l2_dist(repeated_data, perturbed_samples_clip)
        is_far_enough = np.isclose(l2_distances, epsilon)
        # if one is not true, the neighbors might be at the wrong distance
        assert np.all(is_far_enough), \
            "Your perturbed samples do not seem to have the distance to the original ones that you specified with " \
            "epsilon. " \
            "This might be the case if scale is small and epsilon large"
    return perturbed_samples_clip


def evaluate_model_acc(model, data, labels):
    """Given a compiled model, data and the corresponding labels, get accuracy"""
    loss, acc = model.evaluate(data, labels)
    return acc


def calculate_ws_distance(neighbors, original, num_classes=10):
    """
    Calculates the average pairwise wasserstein-distances between the logits of an original point and its neighbors
    :param neighbors: logits of model predictions on neighbors, np.array in shape (NUM_DATA_POINTS, NUM_NEIGHBORS, num_classes)
    :param original: logits of model predictions on original data, np.array in shape (NUM_DATA_POINTS, num_classes)
    :return: mean pairwise wasserstein distances per point's logits
    """
    NUM_DATA_POINTS = neighbors.shape[0]
    NUM_NEIGHBORS = neighbors.shape[1]
    means_train_ws = []
    for i in range(NUM_DATA_POINTS):
        ws_dists = []
        for j in range(NUM_NEIGHBORS):
            ws_dist = wasserstein_distance(neighbors[i][j], original[i])
            ws_dists.append(ws_dist)
        means_train_ws.append(np.mean(np.asarray(ws_dists)))
    return means_train_ws


def calculate_l2_distances(neighbors, original):
    """
    Calculates the average pairwise l2-distances between the logits of an original point and its neighbors
    :param neighbors: logits of model predictions on neighbors, np.array in shape (NUM_DATA_POINTS, NUM_NEIGHBORS, num_classes)
    where num-classes determines how many logits we have per sample
    :param original: logits of model predictions on original data, np.array in shape (NUM_DATA_POINTS, num_classes)
    :return:means_train_l2: array of shape num-classes: per class output neuron mean pairwise distance of logits,
            stds_train_l2: standard deviations belonging to means_train_l2
            mses_train_l2: average mean squared error over all pairwise means between orig and its neighbors
    """

    NUM_DATA_POINTS = neighbors.shape[0]
    NUM_NEIGHBORS = neighbors.shape[1]

    means_train_l2 = []
    stds_train_l2 = []
    mses_train_l2 = []
    for i in range(NUM_DATA_POINTS):
        dists = []
        mses = []
        # this could be vectorized
        for j in range(NUM_NEIGHBORS):
            dist = neighbors[i][j] - original[i]
            mse = (np.square(dist)).mean()  # mean squared error
            dists.append(dist)
            mses.append(mse)
        stds_train_l2.append(np.std(np.asarray(dists), axis=0))
        means_train_l2.append(np.mean(np.asarray(dists), axis=0))
        mses_train_l2.append(np.mean(np.asarray(mses)))
    return means_train_l2, stds_train_l2, mses_train_l2


def calculate_var_mean_std_at_correct_class(neighbors, correct_cls, num_classes=10):
    """
    Otherwise the mean/var/std is calculated at every class. But that might not be so expressive. Hence, this function
    extracts the respective scores per data point only for the correct class
    :param neighbors: logits of model predictions on neighbors, np.array in shape (NUM_DATA_POINTS, NUM_NEIGHBORS, num_classes)
    :param correct_cls: the correct classes for the neighbors, np.array in shape (NUM_DATA_POINTS,)
    :return: arrays of shape (NUM_DATA_POINTS,1) for 1) the variance, 2) mean, and 3) std among the neighbors, only for the correct class
    """

    # calculate these values over all classes
    varis, means, stds = calculate_var_mean_std(neighbors)

    # correct class indices
    var_indices = correct_cls.astype(int)

    # per row (data point) choose the correct class' metrics
    varis_arr = np.choose(var_indices, varis.T)
    means_arr = np.choose(var_indices, means.T)
    stds_arr = np.choose(var_indices, stds.T)

    return varis_arr, means_arr, stds_arr


def calculate_var_mean_std(neighbors):
    """
    Calculates the variance, mean, and std within all neighbors per specific data point
    :param neighbors: logits of model predictions on neighbors, np.array in shape (NUM_DATA_POINTS, NUM_NEIGHBORS, num_classes)
    :return: arrays of shape (NUM_DATA_POINTS, NUM_CLASSES) for 1) variances, 2) mean, and 3) std in the neighbors per
    original data point per class.
    """
    NUM_DATA_POINTS = neighbors.shape[0]
    varis = []
    means = []
    stds = []
    for i in range(NUM_DATA_POINTS):
        varis.append(np.var(neighbors[i], axis=0))
        means.append(np.mean(neighbors[i], axis=0))
        stds.append(np.std(neighbors[i], axis=0))

    return np.asarray(varis), np.asarray(means), np.asarray(stds)


def flatten_neighbors_predictions(neighbors):
    """
    Flattens the neighbors' predictions in order to use them as input to the membership inference attack
    :param neighbors: logits of model predictions on neighbors, np.array in shape (NUM_DATA_POINTS, NUM_NEIGHBORS, num_classes)
    :return: flattened array of all the neighbor's predictions per original data point
    """
    NUM_DATA_POINTS = neighbors.shape[0]
    vars_train_l2 = []
    for i in range(NUM_DATA_POINTS):
        var = neighbors[i].flatten()
        vars_train_l2.append(var)
    return vars_train_l2


def class_consistency(logits_datapoints, logits_neighbors):
    """
    Given the logits for some data points and their respective neighbors: first calculates the prediction order of
    classes for the data points (from highest confidence to lowest confidence).
    Then calculates per data point and per class what percentage of neighbors have the same predicted class at the
    respective index in the order of confidences
    :param logits_datapoints: np.array of shape (num_datapoints, classes) Original data points.
    :param logits_neighbors: np.array of shape (num_datapoints, num_neighbors, num_classes) Respective neighbors.
    :return: np.array of shape (num_datapoints, num_classes): percentage of neighbors whose prediction at the given
    confidence index matches the original prediction
    """

    # calculate the order of the predicted class for the original data points and the neighbors
    ground_truth_order = np.argsort(-1 * logits_datapoints, axis=1)  # we need to do *-1 to have descending order
    neighbor_order = np.argsort(-1 * logits_neighbors, axis=2)

    num_neighbors = neighbor_order.shape[1]
    # repeat the elements in the ground truth for more efficient np-based comparison
    ground_truth_order_aug = np.repeat(ground_truth_order, num_neighbors, axis=0).reshape(neighbor_order.shape)
    # calculate consistency based on for how many neighbors the class predictions match
    equal = np.equal(ground_truth_order_aug, neighbor_order)
    consistency = np.mean(equal, axis=1)

    return consistency


def distance_until_class_change(logits_orig_train_arr, list_logits_neighbors):
    """
    Given the logits for some data points and a list of arrays containing their respective neighbors, for example
    a list with different neighbors generated with different noise scales: calculates the first index in list where
    the model prediction on the original data point does not match the prediction of one of the neighbors.
    This might allow to estimate a data point's distance to the decision boundary.
    For the function to yield meaningful results, the list_logits_neighbors needs to contain the neighbors in an order
    (e.g. neighbors with noise scales [0.01, 0.05, 0.1]).
    This is because we use the indices(and return the first one) where a change in the neighbors appears.
    If no change in any neighbors occurs for a given data point, the result for it will be -1.
    :param logits_orig_train_arr: np.array of shape (num_datapoints, classes) Original data points.
    :param list_logits_neighbors: list of np.arrays of shape (num_datapoints, num_neighbors, num_classes)
        Respective neighbors generated with different functions
    :return: np.array of shape (num_datapoints, len(list_logits_neighbors)):
        array containing per data point the first index where a different class than the original one was predicted
        for at least one of its neighbors. If no mismatch occurs, the default value is -1.
    """

    # make a numpy array out of the list for more efficient computation
    stuck = np.stack(list_logits_neighbors)
    neighbor_labels = np.argmax(stuck, axis=3)

    # get the original classes
    ground_truth_clas = np.argmax(logits_orig_train_arr, axis=1)
    # for easier broadcasting in comparison repeat labels
    num_datapoints = logits_orig_train_arr.shape[0]
    num_neighbors = list_logits_neighbors[0].shape[1]

    ground_truth_clas_repeat = np.repeat(ground_truth_clas, num_neighbors, axis=0).reshape(neighbor_labels.shape[1:])

    # compare where the neighbors labes deviate from the original labels
    non_equal = ~np.equal(neighbor_labels, ground_truth_clas_repeat)

    # creates array of shape (len(list_logits_neighbors), num_datapoints)
    # True values indicate that among the neighbor array at the given index there was one or more mismatch
    mismatch_indices = np.any(non_equal, axis=2)

    # obtain a mask specifying for each data point if there is a label mismatch
    data_points_that_have_missmatch = np.any(mismatch_indices, axis=0)

    # get the earliest index where this mismatch happens, when there is no mismatch, we have default value -1
    default = np.ones(num_datapoints) * (-1)
    default[data_points_that_have_missmatch] = np.argmax(mismatch_indices == 1, axis=0)[data_points_that_have_missmatch]
    return default
