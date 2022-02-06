"""
Deep k-Nearest Neighbors.
The idea is based on https://arxiv.org/pdf/1803.04765.pdf.
The code is based on https://github.com/cleverhans-lab/cleverhans/blob/master/cleverhans_v3.1.0/cleverhans/model_zoo/deep_k_nearest_neighbors/dknn.py
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import copy
from bisect import bisect_left
import numpy as np
import enum
import tensorflow as tf
from sklearn.metrics import euclidean_distances

from tensorflow.keras.models import Model


class NearestNeighbor:
    """Unsupervised learner for finding k nearest neighbors.
    For this, the libraries FALCONN and FAISS can be used.
    FALCONN: https://github.com/FALCONN-LIB/FALCONN
    Faiss: https://github.com/facebookresearch/faiss
    :param backend: the library which should be used, either FALCONN or FAISS.
    :param dimension: dimension.
    :param neighbors: number (k) of neighbors to find.
    :param number_bits: number of hash bits used by locality-sensitive hashing (LSH).
    :param nb_tables: number of tables used by FALCONN to perform LSH.
    :param num_rotations: number of rotations used by FALCONN to perform LSH, for dense set it to 1; for sparse data set it to 2.
    """

    class BACKEND(enum.Enum):
        """Libraries that can be used to find kNN."""

        FALCONN = 1
        FAISS = 2

    def __init__(
        self,
        backend,
        dimension,
        neighbors,
        number_bits,
        nb_tables=None,
        num_rotations=2,
    ):
        assert backend in NearestNeighbor.BACKEND

        self._NEIGHBORS = neighbors
        self._BACKEND = backend

        if self._BACKEND is NearestNeighbor.BACKEND.FALCONN:
            self._init_falconn(dimension, number_bits, nb_tables, num_rotations)
        elif self._BACKEND is NearestNeighbor.BACKEND.FAISS:
            self._init_faiss(
                dimension,
            )
        else:
            raise NotImplementedError

    def _init_falconn(self, dimension, number_bits, nb_tables, num_rotations):
        import falconn

        assert nb_tables >= self._NEIGHBORS

        # LSH parameters
        params_cp = falconn.LSHConstructionParameters()
        params_cp.dimension = dimension
        params_cp.lsh_family = falconn.LSHFamily.CrossPolytope
        params_cp.distance_function = falconn.DistanceFunction.EuclideanSquared
        params_cp.l = nb_tables
        params_cp.num_rotations = num_rotations
        params_cp.seed = 5721840
        # we want to use all the available threads to set up
        params_cp.num_setup_threads = 0
        params_cp.storage_hash_table = falconn.StorageHashTable.BitPackedFlatHashTable

        # we build number_bits-bit hashes so that each table has
        # 2^number_bits bins; a rule of thumb is to have the number
        # of bins be the same order of magnitude as the number of data points
        falconn.compute_number_of_hash_functions(number_bits, params_cp)
        self._falconn_table = falconn.LSHIndex(params_cp)
        self._falconn_query_object = None
        self._FALCONN_NB_TABLES = nb_tables

    def _init_faiss(
        self,
        dimension,
    ):
        import faiss

        # following code should be used if GPU is used
        # res = faiss.StandardGpuResources()
        # self._faiss_index = faiss.GpuIndexFlatL2(
        #    res,
        #    dimension,
        # )

        self._faiss_index = faiss.IndexFlatL2(dimension)

    def _find_knns_falconn(
        self, data_activations_layer, output, train_activations_lsh_layer
    ):
        """
        Finds the k nearest neighbors with the falconn library.
        """

        if self._falconn_query_object is None:
            self._falconn_query_object = self._falconn_table.construct_query_object()
            self._falconn_query_object.set_num_probes(self._FALCONN_NB_TABLES)

        missing_indices = np.zeros(output.shape, dtype=np.bool_)
        neighbor_distance = []
        for i in range(data_activations_layer.shape[0]):
            # find k nearest neighbors for data point data_activations_layer[i]
            query_res = self._falconn_query_object.find_k_nearest_neighbors(
                data_activations_layer[i], self._NEIGHBORS
            )
            try:
                output[i, :] = query_res
            except:  # pylint: disable-msg=W0702
                # mark missing indices
                missing_indices[i, len(query_res) :] = True

                output[i, : len(query_res)] = query_res
            # get euclidean distances of activations of nn to activation of point
            distances = euclidean_distances(
                data_activations_layer[i].reshape(1, -1),
                train_activations_lsh_layer[query_res],
            )
            neighbor_distance.append(distances[0])

        return missing_indices, neighbor_distance

    def _find_knns_faiss(self, x, output, train_activations_lsh_layer):
        """
        Finds the k nearest neighbors with the faiss library.
        :param x: tensor, feature vector of input of layer.
        :param output: output of layer.
        :return: missing_indices, neighbor_distance, knns_datapoints
        """
        neighbor_distance, neighbor_index = self._faiss_index.search(x, self._NEIGHBORS)
        missing_indices = neighbor_distance == -1

        d1 = neighbor_index.reshape(-1)

        output.reshape(-1)[np.logical_not(missing_indices.flatten())] = d1[
            np.logical_not(missing_indices.flatten())
        ]

        # get the actual data points of the knns in case there are needed later
        knns_datapoints = []
        for point in range(len(neighbor_index)):
            for index in range(self._NEIGHBORS):
                knns_datapoints.append(
                    self._faiss_index.reconstruct(int(neighbor_index[point, index]))
                )

        return missing_indices, neighbor_distance, knns_datapoints

    def add(self, activations_lsh_for_one_layer):
        """
        Depending on used library, add x to index/table for finding knns.
        """
        if self._BACKEND is NearestNeighbor.BACKEND.FALCONN:
            self._falconn_table.setup(activations_lsh_for_one_layer)
        elif self._BACKEND is NearestNeighbor.BACKEND.FAISS:
            self._faiss_index.add(activations_lsh_for_one_layer)
        else:
            raise NotImplementedError

    def find_knns(self, data_activations_layer, output, train_activations_lsh_layer):
        """
        Finds the k nearest neighbors.
        """
        if self._BACKEND is NearestNeighbor.BACKEND.FALCONN:
            return self._find_knns_falconn(
                data_activations_layer, output, train_activations_lsh_layer
            )
        elif self._BACKEND is NearestNeighbor.BACKEND.FAISS:
            return self._find_knns_faiss(
                data_activations_layer, output, train_activations_lsh_layer
            )
        else:
            raise NotImplementedError


class DkNNModel(Model):
    def __init__(
        self,
        neighbors,
        layers,
        get_activations,
        train_data,
        train_labels,
        nb_classes=10,
        nb_tables=200,
        number_bits=17,
        back_end=NearestNeighbor.BACKEND.FALCONN,
        nb_cali=-1,
    ):
        """
        Implements the DkNN algorithm. See https://arxiv.org/abs/1803.04765 for more details.
        :param neighbors: number of neighbors to find per layer.
        :param layers: a list of layer names to include in the DkNN. (those are the layers at which we get the neighbors)
        :param get_activations: a callable that takes a np array and a layer name and returns its activations on the data.
        :param train_data: a np array of training data.
        :param train_labels: a np vector of training labels.
        :param nb_classes: the number of classes in the task.
        :param nb_tables: number of tables used by FALCONN to perform locality-sensitive hashing.
        :param number_bits: number of hash bits used by LSH.
        :param back_end: NearestNeighbor backend.
        :param nb_cali: number of calibration points for the DkNN.
        """
        super(DkNNModel, self).__init__()
        self.neighbors = neighbors
        self.nb_tables = nb_tables
        self.nb_layers = layers
        self.get_activations = get_activations
        self.nb_cali = nb_cali
        self.calibrated = False
        self.number_bits = number_bits
        self.nb_classes = nb_classes
        self.back_end = back_end

        # Compute training data activations
        self.nb_train = train_labels.shape[0]
        assert self.nb_train == train_data.shape[0]
        #print("Getting activations for training data")
        self.train_activations = get_activations(train_data)
        #print("Received activations for training data")
        self.train_labels = train_labels

        # Build locality-sensitive hashing tables for training representations
        self.train_activations_lsh = copy.copy(self.train_activations)
        #print("Initializing locality-sensitive hashing")
        self.init_lsh()

    def init_lsh(self):
        """
        Initializes locality-sensitive hashing with FALCONN to find nearest neighbors in training data.
        """
        self.query_objects = (
            {}
        )  # contains the object that can be queried to find nearest neighbors at each layer.
        # mean of training data representation per layer (that needs to be substracted before
        # NearestNeighbor).
        self.centers = {}
        for layer in self.nb_layers:
            # Normalize all the lengths, since we care about the cosine similarity.
            self.train_activations_lsh[layer] /= np.linalg.norm(
                self.train_activations_lsh[layer], axis=1
            ).reshape(-1, 1)

            # Center the dataset and the queries: this improves the performance of LSH quite a bit.
            center = np.mean(self.train_activations_lsh[layer], axis=0)
            self.train_activations_lsh[layer] -= center
            self.centers[layer] = center

            # print("Constructing the NearestNeighbor table")
            # print("dimension for tables is ", self.train_activations_lsh[layer].shape[1])
            self.query_objects[layer] = NearestNeighbor(
                backend=self.back_end,
                dimension=self.train_activations_lsh[layer].shape[1],
                number_bits=self.number_bits,
                neighbors=self.neighbors,
                nb_tables=self.nb_tables,
            )

            self.query_objects[layer].add(self.train_activations_lsh[layer])

    def find_train_knns(self, data_activations):
        """
        Given a data_activation dictionary that contains a np array with activations for each layer,
        find the knns in the training data.
        :param data_activations: dictionary that contains a np array with activations for each layer.
        :return: knns_ind (indices of k nearest neighbors), knns_labels (labels of k nearest neighbors), for FAISS: knns_distances (distances of k nearest neighbors)
        """
        knns_ind = {}
        knns_labels = {}
        knns_distances = {}

        for layer in self.nb_layers:
            # Pre-process representations of data to normalize and remove training data mean.
            data_activations_layer = copy.copy(data_activations[layer])
            nb_data = data_activations_layer.shape[0]
            data_activations_layer /= np.linalg.norm(
                data_activations_layer, axis=1
            ).reshape(-1, 1)
            data_activations_layer -= self.centers[layer]

            # find indices of nearest neighbors in training data.
            knns_ind[layer] = np.zeros(
                (data_activations_layer.shape[0], self.neighbors), dtype=np.int32
            )

            knn_errors = 0
            knns_distances[layer] = np.zeros(
                (data_activations_layer.shape[0], self.neighbors), dtype=np.int32
            )

            if self.back_end is NearestNeighbor.BACKEND.FALCONN:
                # FALCONN does not return distance
                knn_missing_indices, knn_distance = self.query_objects[layer].find_knns(
                    data_activations_layer,
                    knns_ind[layer],
                    self.train_activations_lsh[layer],
                )
                knns_distances[layer] = knn_distance

            elif self.back_end is NearestNeighbor.BACKEND.FAISS:
                # FAISS returns distance

                knn_missing_indices, knn_distance, _ = self.query_objects[
                    layer
                ].find_knns(
                    data_activations_layer,
                    knns_ind[layer],
                    self.train_activations_lsh[layer],
                )
                knns_distances[layer] = knn_distance

            knn_errors += knn_missing_indices.flatten().sum()

            # Find labels of neighbors found in the training data.
            knns_labels[layer] = np.zeros((nb_data, self.neighbors), dtype=np.int32)

            knns_labels[layer].reshape(-1)[
                np.logical_not(knn_missing_indices.flatten())
            ] = self.train_labels[
                knns_ind[layer].reshape(-1)[
                    np.logical_not(knn_missing_indices.flatten())
                ]
            ]

        return knns_ind, knns_labels, knns_distances

    def nonconformity(self, knns_labels):
        """
        Given an dictionary of nb_data x nb_classes dimension, compute the nonconformity of
        each candidate label for each data point: i.e. the number of knns whose label is
        different from the candidate label.
        :param knns_labels: labels of the k nearest neighbors.
        :return: k nearest neighbors whose labels differ from the candidate label.
        """
        nb_data = knns_labels[self.nb_layers[0]].shape[0]
        knns_not_in_class = np.zeros((nb_data, self.nb_classes), dtype=np.int32)
        for i in range(nb_data):
            # Compute number of nearest neighbors per class
            knns_in_class = np.zeros(
                (len(self.nb_layers), self.nb_classes), dtype=np.int32
            )
            for layer_id, layer in enumerate(self.nb_layers):
                knns_in_class[layer_id, :] = np.bincount(
                    knns_labels[layer][i], minlength=self.nb_classes
                )

            # Compute number of knns in other class than class_id
            for class_id in range(self.nb_classes):
                knns_not_in_class[i, class_id] = np.sum(knns_in_class) - np.sum(
                    knns_in_class[:, class_id]
                )
        return knns_not_in_class

    def preds_conf_cred(self, knns_not_in_class):
        """
        Given an array of nb_data x nb_classes dimensions, use conformal prediction to compute
        the DkNN's prediction, confidence and credibility.
        :param knns_not_in_class: k nearest neighbors whose labels differ from the candidate label.
        :return: preds_knn (prediction), confs (confidence), creds (credibility)
        """
        nb_data = knns_not_in_class.shape[0]
        preds_knn = np.zeros(nb_data, dtype=np.int32)
        confs = np.zeros((nb_data, self.nb_classes), dtype=np.float32)
        creds = np.zeros((nb_data, self.nb_classes), dtype=np.float32)

        for i in range(nb_data):
            # p-value of test input for each class
            p_value = np.zeros(self.nb_classes, dtype=np.float32)
            for class_id in range(self.nb_classes):
                # p-value of (test point, candidate label)
                p_value[class_id] = (
                    float(self.nb_cali)
                    - bisect_left(
                        self.cali_nonconformity, knns_not_in_class[i, class_id]
                    )
                ) / float(self.nb_cali)
            preds_knn[i] = np.argmax(p_value)
            confs[i, preds_knn[i]] = 1.0 - np.sort(p_value)[-2]
            creds[i, preds_knn[i]] = p_value[preds_knn[i]]
        return preds_knn, confs, creds

    def fprop_np(self, data_np, get_knns=False):
        """
        Performs a forward pass through the DkNN on an numpy array of data.
        :param data_np: numpy array of data
        :param get_knns: if True, get details about the knns as output (index, label, and distance for FAISS)
        :return: creds (credibility)
        """
        if not self.calibrated:
            raise ValueError(
                "DkNN needs to be calibrated by calling DkNNModel.calibrate method once before inferring."
            )
        if get_knns:
            data_activations = self.get_activations(data_np)

            knns_ind, knns_labels, knns_distances = self.find_train_knns(
                data_activations
            )

            knns_not_in_class = self.nonconformity(knns_labels)
            _, _, creds = self.preds_conf_cred(knns_not_in_class)

            return creds, knns_ind, knns_labels, knns_distances
        else:
            data_activations = self.get_activations(data_np)
            _, knns_labels = self.find_train_knns(data_activations)
            knns_not_in_class = self.nonconformity(knns_labels)
            _, _, creds = self.preds_conf_cred(knns_not_in_class, data_np)
            return creds

    def fprop(self, x):
        """
        Performs a forward pass through the DkNN on a TF tensor by wrapping
        the fprop_np method.
        :param x: tensor.
        :return: {self.O_LOGITS: logits}
        """
        logits = tf.compat.v1.py_func(self.fprop_np, [x], tf.float32)
        return {self.O_LOGITS: logits}

    def calibrate(self, cali_data, cali_labels):
        """
        Runs the DkNN on holdout data to calibrate the credibility metric.
        :param cali_data: np array of calibration data.
        :param cali_labels: np vector of calibration labels.
        """
        self.nb_cali = cali_labels.shape[0]
        self.cali_activations = self.get_activations(cali_data)
        self.cali_labels = cali_labels

        #print("Starting calibration of DkNN.")
        cali_knns_ind, cali_knns_labels, _ = self.find_train_knns(self.cali_activations)
        assert all(
            [v.shape == (self.nb_cali, self.neighbors) for v in cali_knns_ind.values()]
        )
        assert all(
            [
                v.shape == (self.nb_cali, self.neighbors)
                for v in cali_knns_labels.values()
            ]
        )

        cali_knns_not_in_class = self.nonconformity(cali_knns_labels)
        cali_knns_not_in_l = np.zeros(self.nb_cali, dtype=np.int32)
        for i in range(self.nb_cali):
            cali_knns_not_in_l[i] = cali_knns_not_in_class[i, cali_labels[i]]
        cali_knns_not_in_l_sorted = np.sort(cali_knns_not_in_l)
        self.cali_nonconformity = np.trim_zeros(cali_knns_not_in_l_sorted, trim="f")
        self.calibrated = True
        #print("DkNN calibration complete.")