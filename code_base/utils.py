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


def generate_neighboring_points(data_sample, epsilon, amount, noise_vectors=None, scale=0.1):
    """
    This function is given a data point data_sample and it generates amount many data points in an epsilon neighborhood
    scale is given to the noise function to sample the random noise
    :param data_sample: the data point for which we want to generate neighbors
    :param epsilon: how far the neighbors should be away in an L2 norm
    :param amount: how many neighbors to generate
    :param noise_vectors: if we want deterministic noise vectors for neighbor creation, we can pass them here
    :param scale: the sigma for the normal noise distribution
    :return: an array of the generated neighbors
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
    # the clip function cannot deal with np broadcasts
    repeated_data = np.repeat(data_sample, amount, axis=0)  # therefore, repeat sample

    # the following line is for debugging purpose only:
    a = l2_dist(repeated_data, perturbed_samples_clip)

    perturbed_samples_clip_eps = l2_dist.clip_perturbation(repeated_data, perturbed_samples_clip, epsilon)

    b = l2_dist(repeated_data, perturbed_samples_clip_eps)
    c = np.isclose(b, epsilon)  # it's floats so check for every element if roughly the same as epsilon
    # if one is not true, the neighbors might be at the wrong distance
    assert np.all(c), \
        "Your perturbed samples do not seem to have the distance to the original ones that you specified with epsilon. " \
        "This might be the case if scale is small and epsilon large"
    return perturbed_samples_clip_eps


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
        stds = []
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

    :param neighbors: logits of model predictions on neighbors, np.array in shape (NUM_DATA_POINTS, NUM_NEIGHBORS, num_classes)
    :param correct_cls: the correct classes for the neighbors, np.array in shape (NUM_DATA_POINTS,)
    :return: an array of shape (NUM_DATA_POINTS,3) with the variance, mean, and std among the neighbors, only for the correct class
    """

    NUM_DATA_POINTS = neighbors.shape[0]
    NUM_NEIGHBORS = neighbors.shape[1]

    # calculate these values over all classes -> have them flattened
    all_classes_metrics = calculate_var_mean_std(neighbors)

    var_indices = correct_cls.astype(int)  # repeated labels -> yields var at correct class
    mean_indices = (var_indices + num_classes).astype(int)  # the mean of correct class is found num_classes later
    std_indices = (var_indices + 2 * num_classes).astype(int)  # the variance is found two later

    take_indices = np.stack((var_indices, mean_indices, std_indices), axis=-1)

    # for now slow way; later fancy indexing
    metrics = []
    for i in range(NUM_DATA_POINTS):
        metric = all_classes_metrics[i][take_indices[i, :]]
        metrics.append(metric)

    # metrics = np.take(all_classes_metrics, np.transpose(take_indices))
    # metrics = np.take(all_classes_metrics, take_indices)
    # metrics = all_classes_metrics[take_indices] # now select values at these classes
    # rows= np.arange(NUM_DATA_POINTS).astype(int)
    # metrics = all_classes_metrics[rows, take_indices]
    metrics = np.asarray(metrics)
    return metrics


def calculate_var_mean_std(neighbors):
    """
    Calculates the variance within all neighbors per specific data point
    :param neighbors: logits of model predictions on neighbors, np.array in shape (NUM_DATA_POINTS, NUM_NEIGHBORS, num_classes)
    :return: array of variances, mean, and std in the neighbor per original data point
    """
    NUM_DATA_POINTS = neighbors.shape[0]
    vars_train_l2 = []
    for i in range(NUM_DATA_POINTS):
        var = np.array([
            # neighbors[i].flatten(),
            np.var(neighbors[i], axis=0),
            np.mean(neighbors[i], axis=0),
            np.std(neighbors[i], axis=0),
        ]).flatten()
        vars_train_l2.append(var)
    return vars_train_l2


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
