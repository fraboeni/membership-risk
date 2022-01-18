from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import pickle

from code_base.dknn import *
from code_base.get_data import get_data
from code_base.utils import generate_neighboring_points

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

K_NEIGHBORS = 50  # for DkNN: how many nearest neighbors to return
NUM_NEIGHBORS = 200  # how many Gaussian noise neighbors to generate
WIDTH, HEIGHT, DEPTH = 32, 32, 3
SCALES = [0.005, 0.01]  # for generate_neighboring_points
EPS = [0.2, 0.3] # for generate_neighboring_points according to the noise SCALES
NUM_DATA_POINTS = 1000 # how many of member/non member elements are used
NUM_CALIBRATION = 1  # calibration is not necessary for our usecase anyways, so it would be enough to calibrate on one sample
NUM_TOTAL = NUM_NEIGHBORS * NUM_DATA_POINTS # how many data points to operate with


backend = (
    NearestNeighbor.BACKEND.FAISS
)

# get data
members, test_data = get_data('cifar10', augmentation=False, batch_size=NUM_DATA_POINTS, indices_to_use=range(0, 25000))
non_members, _ = get_data('cifar10', augmentation=False, batch_size=NUM_DATA_POINTS, indices_to_use=range(25000, 50000))

# get NUM_DATA_POINTS many neighbors
a = members.next()
mem = a[0]
mem_lab = a[1]

# get NUM_DATA_POINTS many non-neighbors
b = non_members.next()
non_mem = b[0]
non_mem_lab = b[1]

# get one data point for calibration:
# Actually, our implementation does not need the calibration, but since DkNN won't run otherwise, we still do it.
c = test_data.next()
cali_data = b[0][0]  # keep only one data point
cali_label = b[1][0]


# Get the model and obtain all the layers that return some feature maps
model_name = '1011_cifar10_custom'
model = tf.keras.models.load_model('../log_dir/models/cifar10/' + model_name)

layer_indices = []
nb_layers = []
for i in range(len(model.layers)):
    layer = model.layers[i]

    # check for convolutional layer
    if ("conv" not in layer.name) and ("dense" not in layer.name):
        # print(layer.name)
        continue
    # get filter weights
    filters, biases = layer.get_weights()
    # print(layer.name, filters.shape)
    nb_layers.append(layer.name)
    layer_indices.append(i)


# Define callable that returns a dictionary of all activations for a dataset
def get_activations(data) -> dict:
    """
    A callable that takes a np array and a layer name and returns its activations on the data.
    :param data: dataset
    :return: data_activations (dictionary of all activations for given dataset)
    """
    data_activations = {}

    # obtain all the predictions on the data by making a multi-output model
    outputs = [model.get_layer(name=layer).output for layer in nb_layers]
    model_mult = Model(inputs=model.inputs, outputs=outputs)

    # add dimension if only one data point
    if len(data.shape) == 3:
        data = np.expand_dims(data, axis=0)
        predictions = model_mult.predict(data)
    else:
        # use the model for predictions (returns a list)
        predictions = model_mult.predict(data)

    for i in range(len(predictions)):
        pred = predictions[i]
        layer = nb_layers[i]

        # number of samples
        num_samples = pred.shape[0]

        # given the first dimension, numpy reshape has to deduce the other shape
        reshaped_pred = pred.reshape(num_samples, -1)

        data_activations[layer] = reshaped_pred
    return data_activations


result_dict = {}
result_dict['member_data'] = mem
result_dict['member_labels'] = mem_lab
result_dict['non_member_data'] = non_mem
result_dict['non_member_labels'] = non_mem_lab
result_dict['cali_data'] = cali_data
result_dict['cali_labels'] = cali_label
result_dict["k_neighbors"] = K_NEIGHBORS
result_dict["num_data_points"] = NUM_DATA_POINTS

# experiment with different scales and their respective epsilon
for index in range(len(SCALES)):

    scl = SCALES[index]
    eps = EPS[index]

    # dummy arrays for neighbors to be filled
    all_neighbors_train_exp = np.zeros((NUM_TOTAL, WIDTH, HEIGHT, DEPTH))
    all_neighbors_test_exp = np.zeros((NUM_TOTAL, WIDTH, HEIGHT, DEPTH))

    # Generate all the neighbors
    for i in range(NUM_DATA_POINTS):
        all_neighbors_train_exp[i * NUM_NEIGHBORS:(i + 1) * NUM_NEIGHBORS] = generate_neighboring_points(mem[i],
                                                                                                         NUM_NEIGHBORS,
                                                                                                         epsilon=eps,
                                                                                                         scale=scl,
                                                                                                         )
    for i in range(NUM_DATA_POINTS):
        all_neighbors_test_exp[i * NUM_NEIGHBORS:(i + 1) * NUM_NEIGHBORS] = generate_neighboring_points(non_mem[i],
                                                                                                        NUM_NEIGHBORS,
                                                                                                        epsilon=eps,
                                                                                                        scale=scl,
                                                                                                        )
    # get clibration data for DkNN
    calibration_data = generate_neighboring_points(non_mem[i],
                                                   NUM_NEIGHBORS,
                                                   epsilon=eps,
                                                   scale=scl,
                                                   )
    calibration_labels = np.repeat(cali_label, NUM_NEIGHBORS)

    # reshape to batches according to original data point
    all_neighbors_train_resh = all_neighbors_train_exp.reshape(NUM_DATA_POINTS, NUM_NEIGHBORS, WIDTH, HEIGHT, DEPTH)
    all_neighbors_test_resh = all_neighbors_test_exp.reshape(NUM_DATA_POINTS, NUM_NEIGHBORS, WIDTH, HEIGHT, DEPTH)
    # todo: comment in if you want to save neighbors. Might use much storage.
    #result_dict['all_neighbors_train_resh'] = all_neighbors_train_resh
    #result_dict['all_neighbors_test_resh'] = all_neighbors_test_resh

    knns_ind_member_list = []
    knns_ind_non_member_list = []
    knns_distances_member_list = []
    knns_distances_non_member_list = []

    for i in range(NUM_DATA_POINTS):
        # get the original data point
        member_data = mem[i]
        non_member_data = non_mem[i]

        # get the corresponding neighbors
        all_neighbors_train = all_neighbors_train_resh[i]
        all_neighbors_test = all_neighbors_test_resh[i]

        # generate the labels accordingly
        lables_repeted_train = np.repeat(mem_lab[i], NUM_NEIGHBORS)
        lables_repeted_test = np.repeat(non_mem_lab[i], NUM_NEIGHBORS)

        # feed data through DkNNs
        dknn_member = DkNNModel(
            neighbors=K_NEIGHBORS,
            layers=nb_layers,
            get_activations=get_activations,
            train_data=all_neighbors_train,
            train_labels=lables_repeted_train,
            back_end=backend,
        )

        dknn_non_member = DkNNModel(
            neighbors=K_NEIGHBORS,
            layers=nb_layers,
            get_activations=get_activations,
            train_data=all_neighbors_test,
            train_labels=lables_repeted_test,
            back_end=backend,
        )

        # calibrate dknn
        dknn_member.calibrate(
            calibration_data,
            calibration_labels,
        )
        dknn_non_member.calibrate(
            calibration_data,
            calibration_labels,
        )

        # forward propagation through DkNNs and get the metrics for the respective points
        _, knns_ind_member, _, knns_distances_member = dknn_member.fprop_np(
            member_data, get_knns=True
        )
        _, knns_ind_non_member, _, knns_distances_non_member = dknn_non_member.fprop_np(
            non_member_data, get_knns=True
        )
        knns_ind_member_list.append(knns_ind_member)
        knns_ind_non_member_list.append(knns_ind_non_member)
        knns_distances_member_list.append(knns_distances_member)
        knns_distances_non_member_list.append(knns_distances_non_member)

    result_dict[str((scl, eps)), 'knns_ind_member'] = knns_ind_member_list
    result_dict[str((scl, eps)), "knns_ind_non_member"] = knns_ind_non_member_list
    result_dict[str((scl, eps)), "knns_distances_member"] = knns_distances_member_list
    result_dict[str((scl, eps)), "knns_distances_non_member"] = knns_distances_non_member_list

# save:
result_dir = os.getcwd() + '/../log_dir/result_data/exp03/'
pickle.dump(result_dict, open(result_dir + "dknn_metrics.pkl", "wb"))
