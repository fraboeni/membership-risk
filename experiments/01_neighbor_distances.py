import os
import tensorflow as tf

import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

from code_base.get_data import get_data
from code_base.utils import *
from code_base.models import MODELS

os.environ["CUDA_VISIBLE_DEVICES"] = ""

MODEL_NAME = os.getcwd() + '../log_dir/models/cifar10/1004_cifar10_cnn'
model = tf.keras.models.load_model('../log_dir/models/cifar10/1004_cifar10_cnn')
NUM_DATA_POINTS = 1000
WIDTH = 32
HEIGHT = 32
DEPTH = 3

SCALES = [0.01, 0.05, 0.1, 0.2]
NUM_NEIGHBORS = 100


members, _ = get_data('cifar10', augmentation=False, batch_size=NUM_DATA_POINTS, indices_to_use=range(0,25000))
non_members, _ = get_data('cifar10', augmentation=False, batch_size=NUM_DATA_POINTS, indices_to_use=range(25000, 50000))

# get NUM_DATA_POINTS many neighbors
a = members.next()
mem = a[0]
mem_lab = a[1]

# get NUM_DATA_POINTS many non-neighbors
b = non_members.next()
non_mem = b[0]
non_mem_lab = b[1]


# train and test acc
train_acc = model.evaluate(members)[1]
test_acc = model.evaluate(non_members)[1]

result_dict = {}
result_dict['train_acc'] = (train_acc)
result_dict['test_acc'] = (test_acc)
result_dict['num_samples'] = NUM_DATA_POINTS
#result_dict['noise_scale'] = SCALE

for scale in SCALES:
    print(scale)

    # how many samples in this run?
    NUM_TOTAL = NUM_NEIGHBORS * NUM_DATA_POINTS
    # dummy arrays for neighbors
    all_neighbors_train = np.zeros((NUM_TOTAL, WIDTH, HEIGHT, DEPTH))
    all_neighbors_test = np.zeros((NUM_TOTAL, WIDTH, HEIGHT, DEPTH))

    # Generate all the neighbors
    # now make it a really ugly way to generate all the neighbors for 1000 data points
    for i in range(NUM_DATA_POINTS):
        all_neighbors_train[i * NUM_NEIGHBORS:(i + 1) * NUM_NEIGHBORS] = generate_neighboring_points(mem[i],
                                                                                                     NUM_NEIGHBORS,
                                                                                                     scale=scale)
    # now make it a really ugly way to generate all the neighbors for the data points,
    for i in range(NUM_DATA_POINTS):
        all_neighbors_test[i * NUM_NEIGHBORS:(i + 1) * NUM_NEIGHBORS] = generate_neighboring_points(non_mem[i],
                                                                                                    NUM_NEIGHBORS,
                                                                                                    scale=scale)
    # predict all points to obtain the logits
    all_neighbors_train_pred = model.predict(all_neighbors_train)
    all_neighbors_test_pred = model.predict(all_neighbors_test)

    # also for the original points
    logits_orig_train_arr = model.predict(mem[:NUM_DATA_POINTS])
    logits_orig_test_arr = model.predict(non_mem[:NUM_DATA_POINTS])

    # reshape to batches per train data point
    all_neighbors_train_pred_resh = all_neighbors_train_pred.reshape(NUM_DATA_POINTS, NUM_NEIGHBORS,
                                                                     10)  # 10 because 10 classes
    all_neighbors_test_pred_resh = all_neighbors_test_pred.reshape(NUM_DATA_POINTS, NUM_NEIGHBORS, 10)

    # WASSERSTEIN calculation
    train_ws = calculate_ws_distance(all_neighbors_train_pred_resh, logits_orig_train_arr)
    test_ws = calculate_ws_distance(all_neighbors_test_pred_resh, logits_orig_test_arr)

    # L2 calculation
    means_train_l2, stds_train_l2, mses_train_l2 = calculate_l2_distances(all_neighbors_train_pred_resh,
                                                                          logits_orig_train_arr)
    means_test_l2, stds_test_l2, mses_test_l2 = calculate_l2_distances(all_neighbors_test_pred_resh,
                                                                       logits_orig_test_arr)

    # now neighbors only: their per-data point metrics at the correct class
    varis_train, means_train, stds_train = calculate_var_mean_std_at_correct_class(all_neighbors_train_pred_resh,
                                                                                   mem_lab)
    varis_test, means_test, stds_test = calculate_var_mean_std_at_correct_class(all_neighbors_test_pred_resh,
                                                                                non_mem_lab)

    # prediction accuracy
    lables_repeted_train = np.repeat(mem_lab[:NUM_DATA_POINTS], NUM_NEIGHBORS)
    lables_repeted_test = np.repeat(non_mem_lab[:NUM_DATA_POINTS], NUM_NEIGHBORS)
    preds_train = np.argmax(all_neighbors_train_pred, axis=1)
    preds_test = np.argmax(all_neighbors_test_pred, axis=1)
    train_acc_neighbors = accuracy_score(lables_repeted_train, preds_train)
    test_acc_neighbors = accuracy_score(lables_repeted_test, preds_test)

    sub_dict = {}
    sub_dict['wasserstein'] = (train_ws, test_ws)
    sub_dict['msel2'] = (mses_train_l2, mses_test_l2)
    sub_dict['variances'] = (varis_train, varis_test)
    sub_dict['means'] = (means_train, means_test)
    sub_dict['stds'] = (stds_train, stds_test)
    sub_dict['train_acc_neigh'] = train_acc_neighbors
    sub_dict['test_acc_neigh'] = test_acc_neighbors
    result_dict[scale] = sub_dict

result_dir = os.getcwd() + '/../log_dir/result_data/exp01/'
import pickle
pickle.dump( result_dict, open( result_dir+"metrics.pkl", "wb" ) )