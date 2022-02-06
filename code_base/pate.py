"""Core functions for RDP analysis in PATE framework.
This library comprises the core functions for doing differentially private
analysis of the PATE architecture and its various Noisy Max and other
mechanisms.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

from absl import app
from tensorflow.keras.utils import to_categorical
import scipy.stats


def _logaddexp(x):
    """Addition in the log space. Analogue of numpy.logaddexp for a list."""
    m = max(x)
    return m + math.log(sum(np.exp(x - m)))


def _log1mexp(x):
    """Numerically stable computation of log(1-exp(x))."""
    if x < -1:
        return math.log1p(-math.exp(x))
    elif x < 0:
        return math.log(-math.expm1(x))
    elif x == 0:
        return -np.inf
    else:
        raise ValueError("Argument must be non-positive.")


def compute_eps_from_delta(orders, rdp, delta):
    """Translates between RDP and (eps, delta)-DP.
    Args:
      orders: A list (or a scalar) of orders.
      rdp: A list of RDP guarantees (of the same length as orders).
      delta: Target delta.
    Returns:
      Pair of (eps, optimal_order).
    Raises:
      ValueError: If input is malformed.
    """
    if len(orders) != len(rdp):
        raise ValueError("Input lists must have the same length.")
    eps = np.array(rdp) - math.log(delta) / (np.array(orders) - 1)
    idx_opt = np.argmin(eps)
    return eps[idx_opt], orders[idx_opt]


#####################
# RDP FOR THE GNMAX #
#####################


def compute_logq_gaussian(counts, sigma):
    """Returns an upper bound on ln Pr[outcome != argmax] for GNMax.
    Implementation of Proposition 7.
    Args:
      counts: A numpy array of scores.
      sigma: The standard deviation of the Gaussian noise in the GNMax mechanism.
    Returns:
      logq: Natural log of the probability that outcome is different from argmax.
    """
    n = len(counts)
    variance = sigma ** 2
    idx_max = np.argmax(counts)
    counts_normalized = counts[idx_max] - counts
    counts_rest = counts_normalized[np.arange(n) != idx_max]  # exclude one index
    # Upper bound q via a union bound rather than a more precise calculation.
    logq = _logaddexp(
        scipy.stats.norm.logsf(counts_rest, scale=math.sqrt(2 * variance)))

    # A sketch of a more accurate estimate, which is currently disabled for two
    # reasons:
    # 1. Numerical instability;
    # 2. Not covered by smooth sensitivity analysis.
    # covariance = variance * (np.ones((n - 1, n - 1)) + np.identity(n - 1))
    # logq = np.log1p(-statsmodels.sandbox.distributions.extras.mvnormcdf(
    #     counts_rest, np.zeros(n - 1), covariance, maxpts=1e4))

    return min(logq, math.log(1 - (1 / n)))


def rdp_data_independent_gaussian(sigma, orders):
    """Computes a data-independent RDP curve for GNMax.
    Implementation of Proposition 8.
    Args:
      sigma: Standard deviation of Gaussian noise.
      orders: An array_like list of Renyi orders.
    Returns:
      Upper bound on RPD for all orders. A scalar if orders is a scalar.
    Raises:
      ValueError: If the input is malformed.
    """
    if sigma < 0 or np.any(orders <= 1):  # not defined for alpha=1
        raise ValueError("Inputs are malformed.")

    variance = sigma ** 2
    if np.isscalar(orders):
        return orders / variance
    else:
        return np.atleast_1d(orders) / variance


def rdp_gaussian(logq, sigma, orders):
    """Bounds RDP from above of GNMax given an upper bound on q (Theorem 6).
    Args:
      logq: Natural logarithm of the probability of a non-argmax outcome.
      sigma: Standard deviation of Gaussian noise.
      orders: An array_like list of Renyi orders.
    Returns:
      Upper bound on RPD for all orders. A scalar if orders is a scalar.
    Raises:
      ValueError: If the input is malformed.
    """
    if logq > 0 or sigma < 0 or np.any(orders <= 1):  # not defined for alpha=1
        raise ValueError("Inputs are malformed.")

    if np.isneginf(logq):  # If the mechanism's output is fixed, it has 0-DP.
        if np.isscalar(orders):
            return 0.
        else:
            return np.full_like(orders, 0., dtype=np.float)

    variance = sigma ** 2

    # Use two different higher orders: mu_hi1 and mu_hi2 computed according to
    # Proposition 10.
    mu_hi2 = math.sqrt(variance * -logq)
    mu_hi1 = mu_hi2 + 1

    orders_vec = np.atleast_1d(orders)

    ret = orders_vec / variance  # baseline: data-independent bound

    # Filter out entries where data-dependent bound does not apply.
    mask = np.logical_and(mu_hi1 > orders_vec, mu_hi2 > 1)

    rdp_hi1 = mu_hi1 / variance
    rdp_hi2 = mu_hi2 / variance

    log_a2 = (mu_hi2 - 1) * rdp_hi2

    # Make sure q is in the increasing wrt q range and A is positive.
    if (np.any(mask) and logq <= log_a2 - mu_hi2 *
            (math.log(1 + 1 / (mu_hi1 - 1)) + math.log(1 + 1 / (mu_hi2 - 1))) and
            -logq > rdp_hi2):
        # Use log1p(x) = log(1 + x) to avoid catastrophic cancellations when x ~ 0.
        log1q = _log1mexp(logq)  # log1q = log(1-q)
        log_a = (orders - 1) * (
                log1q - _log1mexp((logq + rdp_hi2) * (1 - 1 / mu_hi2)))
        log_b = (orders - 1) * (rdp_hi1 - logq / (mu_hi1 - 1))

        # Use logaddexp(x, y) = log(e^x + e^y) to avoid overflow for large x, y.
        log_s = np.logaddexp(log1q + log_a, logq + log_b)
        ret[mask] = np.minimum(ret, log_s / (orders - 1))[mask]

    assert np.all(ret >= 0)

    if np.isscalar(orders):
        return np.asscalar(ret)
    else:
        return ret


def is_data_independent_always_opt_gaussian(num_teachers, num_classes, sigma,
                                            orders):
    """Tests whether data-ind bound is always optimal for GNMax.
    Args:
      num_teachers: Number of teachers.
      num_classes: Number of classes.
      sigma: Standard deviation of the Gaussian noise.
      orders: An array_like list of Renyi orders.
    Returns:
      Boolean array of length |orders| (a scalar if orders is a scalar). True if
      the data-independent bound is always the same as the data-dependent bound.
    """
    unanimous = np.array([num_teachers] + [0] * (num_classes - 1))
    logq = compute_logq_gaussian(unanimous, sigma)

    rdp_dep = rdp_gaussian(logq, sigma, orders)
    rdp_ind = rdp_data_independent_gaussian(sigma, orders)
    return np.isclose(rdp_dep, rdp_ind)


###################################
# RDP FOR THE THRESHOLD MECHANISM #
###################################


def compute_logpr_answered(t, sigma, counts):
    """Computes log of the probability that a noisy threshold is crossed.
    Args:
      t: The threshold.
      sigma: The stdev of the Gaussian noise added to the threshold.
      counts: An array of votes.
    Returns:
      Natural log of the probability that max is larger than a noisy threshold.
    """
    # Compared to the paper, max(counts) is rounded to the nearest integer. This
    # is done to facilitate computation of smooth sensitivity for the case of
    # the interactive mechanism, where votes are not necessarily integer.
    return scipy.stats.norm.logsf(t - round(max(counts)), scale=sigma)


def compute_rdp_data_independent_threshold(sigma, orders):
    # The input to the threshold mechanism has stability 1, compared to
    # GNMax, which has stability = 2. Hence the sqrt(2) factor below.
    return rdp_data_independent_gaussian(2 ** .5 * sigma, orders)


def compute_rdp_threshold(log_pr_answered, sigma, orders):
    logq = min(log_pr_answered, _log1mexp(log_pr_answered))
    # The input to the threshold mechanism has stability 1, compared to
    # GNMax, which has stability = 2. Hence the sqrt(2) factor below.
    return rdp_gaussian(logq, 2 ** .5 * sigma, orders)


def is_data_independent_always_opt_threshold(num_teachers, threshold, sigma,
                                             orders):
    """Tests whether data-ind bound is always optimal for the threshold mechanism.
    Args:
      num_teachers: Number of teachers.
      threshold: The cut-off threshold.
      sigma: Standard deviation of the Gaussian noise.
      orders: An array_like list of Renyi orders.
    Returns:
      Boolean array of length |orders| (a scalar if orders is a scalar). True if
      the data-independent bound is always the same as the data-dependent bound.
    """

    # Since the data-dependent bound depends only on max(votes), it suffices to
    # check whether the data-dependent bounds are better than data-independent
    # bounds in the extreme cases when max(votes) is minimal or maximal.
    # For both Confident GNMax and Interactive GNMax it holds that
    #   0 <= max(votes) <= num_teachers.
    # The upper bound is trivial in both cases.
    # The lower bound is trivial for Confident GNMax (and a stronger one, based on
    # the pigeonhole principle, is possible).
    # For Interactive GNMax (Algorithm 2), the lower bound follows from the
    # following argument. Since the votes vector is the difference between the
    # actual teachers' votes and the student's baseline, we need to argue that
    # max(n_j - M * p_j) >= 0.
    # The bound holds because sum_j n_j = sum M * p_j = M. Thus,
    # sum_j (n_j - M * p_j) = 0, and max_j (n_j - M * p_j) >= 0 as needed.
    logq1 = compute_logpr_answered(threshold, sigma, [0])
    logq2 = compute_logpr_answered(threshold, sigma, [num_teachers])

    rdp_dep1 = compute_rdp_threshold(logq1, sigma, orders)
    rdp_dep2 = compute_rdp_threshold(logq2, sigma, orders)

    rdp_ind = compute_rdp_data_independent_threshold(sigma, orders)
    return np.isclose(rdp_dep1, rdp_ind) and np.isclose(rdp_dep2, rdp_ind)


#############################
# RDP FOR THE LAPLACE NOISE #
#############################


def compute_logq_laplace(counts, lmbd):
    """Computes an upper bound on log Pr[outcome != argmax] for LNMax.
    Args:
      counts: A list of scores.
      lmbd: The lambda parameter of the Laplace distribution ~exp(-|x| / lambda).
    Returns:
      logq: Natural log of the probability that outcome is different from argmax.
    """
    # For noisy max, we only get an upper bound via the union bound. See Lemma 4
    # in https://arxiv.org/abs/1610.05755.
    #
    # Pr[ j beats i*] = (2+gap(j,i*))/ 4 exp(gap(j,i*)
    # proof at http://mathoverflow.net/questions/66763/

    idx_max = np.argmax(counts)
    counts_normalized = (counts - counts[idx_max]) / lmbd
    counts_rest = np.array(
        [counts_normalized[i] for i in range(len(counts)) if i != idx_max])

    logq = _logaddexp(np.log(2 - counts_rest) + math.log(.25) + counts_rest)

    return min(logq, math.log(1 - (1 / len(counts))))


def rdp_pure_eps(logq, pure_eps, orders):
    """Computes the RDP value given logq and pure privacy eps.
    Implementation of https://arxiv.org/abs/1610.05755, Theorem 3.
    The bound used is the min of three terms. The first term is from
    https://arxiv.org/pdf/1605.02065.pdf.
    The second term is based on the fact that when event has probability (1-q) for
    q close to zero, q can only change by exp(eps), which corresponds to a
    much smaller multiplicative change in (1-q)
    The third term comes directly from the privacy guarantee.
    Args:
      logq: Natural logarithm of the probability of a non-optimal outcome.
      pure_eps: eps parameter for DP
      orders: array_like list of moments to compute.
    Returns:
      Array of upper bounds on rdp (a scalar if orders is a scalar).
    """
    orders_vec = np.atleast_1d(orders)
    q = math.exp(logq)
    log_t = np.full_like(orders_vec, np.inf)
    if q <= 1 / (math.exp(pure_eps) + 1):
        logt_one = math.log1p(-q) + (
                math.log1p(-q) - _log1mexp(pure_eps + logq)) * (
                           orders_vec - 1)
        logt_two = logq + pure_eps * (orders_vec - 1)
        log_t = np.logaddexp(logt_one, logt_two)

    ret = np.minimum(
        np.minimum(0.5 * pure_eps * pure_eps * orders_vec,
                   log_t / (orders_vec - 1)), pure_eps)
    if np.isscalar(orders):
        return np.asscalar(ret)
    else:
        return ret


def main(argv):
    del argv  # Unused.

import math
import numpy as np
import os
import pickle

#from personalized_pate.models.classifiers import Classifier, encode_one_hot, ConvNetMNIST


def average_dp_budgets(epsilons, deltas, weights=None):
    n = len(epsilons)
    assert n == len(deltas)
    if weights:
        assert n == len(weights)
        assert sum(weights) == 1
    else:
        weights = [1 / n] * n
    epsilons_avg = math.log(sum(np.exp(np.array(epsilons)) * np.array(weights)))
    deltas_avg = sum(np.array(deltas) * np.array(weights))
    return epsilons_avg, deltas_avg


def set_duplications(budgets, precision):
    """
    This method is used for upsampling PATE only. It determines the number of duplications each sensitive data point
    gets corresponding to the given budgets.
    :param budgets:     list or array that contains one privacy budget as epsilon-value for each sensitive data point
    :param precision:   float to determine how precisely the intended ratios between different privacy budgets have to
                        be approximated
    :return:            array of integers to determine the number of duplications for each point
    """
    p = int(1 / precision)
    budgets = np.around(np.array(np.sqrt(budgets / np.min(budgets))) * p) / p
    tmp = np.copy(budgets)
    while np.any(np.abs(np.around(tmp, 2) - np.around(tmp)) > precision):
        tmp += budgets
    return np.around(tmp).astype(int)


def pack_overlapping_subsets(df, col_label, n_classes, n_teachers, budgets, seed, precision):
    """
    This method is used for upsampling PATE only. It determines the allocation of all sensitive data points
    (including duplications) to all teachers.
    :param df:          DataFrame that contains all sensitive data
    :param col_label:   string specifies the column name of the labels within df
    :param n_classes:   int for the number of different classes within the distribution of data as in df
    :param n_teachers:  int for the number of teachers to be trained
    :param budgets:     list of budgets corresponding to the data in df
    :param seed:        int for the random seeding
    :param precision:   float to determine how precisely the intended ratios between different privacy budgets have to
                        be approximated
    :return:            tuple that contains the subsets of sensitive data (including duplications) for all teachers, and
                        a list that contains each a list of all data points corresponding to one teacher for each
                        teacher
    """
    n_data = len(budgets)
    assert len(df) == n_data
    duplications = set_duplications(budgets=budgets, precision=precision)
    mapping_t2p = [[] for _ in range(n_teachers)]
    teacher = 0
    np.random.seed(seed)
    permutation = np.random.permutation(n_teachers)
    for point in range(n_data):
        for _ in range(min(n_teachers, duplications[point])):
            mapping_t2p[permutation[teacher]] += [point]
            teacher += 1
            if teacher == n_teachers:
                teacher = 0
                permutation = np.random.permutation(n_teachers)
    subsets = []
    labels = df.pop(col_label).to_numpy()
    features = df.to_numpy()
    for points in mapping_t2p:
        subsets.append((features[points], to_categorical(labels=labels[points], n_classes=n_classes)))
    return subsets, mapping_t2p


def pack_subsets(df, col_label, n_classes, n_teachers, epsilons, seed, collector, precision=0.1):
    """
    This method determines the allocation of all sensitive data to all teachers.
    :param df:          DataFrame that contains all sensitive data
    :param col_label:   string specifies the column name of the labels within df
    :param n_classes:   int for the number of different classes within the distribution of data as in df
    :param n_teachers:  int for the number of teachers to be trained
    :param epsilons:    list of privacy specifications corresponding to the data in df
    :param seed:        int for the random seeding
    :param collector:   Collector that is used for personalized privacy accounting
    :param precision:   float to determine how precisely the intended ratios between different privacy budgets have to
                        be approximated
    :return:            tuple that contains the subsets of sensitive data (including duplications) for all teachers, and
                        a list that contains each a list of all data points corresponding to one teacher for each
                        teacher
    """
    assert collector in ['GNMax', 'uGNMax', 'vGNMax', 'wGNMax']
    if collector == 'uGNMax':
        return pack_overlapping_subsets(df, col_label, n_classes, n_teachers, epsilons, seed, precision)
    n_data = len(epsilons)
    assert len(df) == n_data
    if collector == 'GNMax':
        epsilons = np.array([min(epsilons)] * n_data)
    levels = np.unique(epsilons)
    ratios = np.array([sum(epsilons == level) / n_data for level in levels])
    assert all(ratios >= 1 / n_teachers - 0.01)
    nums_partitions = np.array([int(round(ratio * n_teachers, 0)) for ratio in ratios])
    nums_partitions[np.argmax(nums_partitions)] += n_teachers - sum(nums_partitions)
    labels = df.pop(col_label).to_numpy()
    features = df.to_numpy()
    subsets = []
    mapping_t2p = []
    np.random.seed(seed)
    for i, n_partitions in enumerate(nums_partitions):
        idx_level = np.array(epsilons == levels[i])
        idx_used = np.array([False] * sum(idx_level))
        n_data_teacher = sum(idx_level) // n_partitions
        for j in range(n_partitions):
            features_tmp = features[idx_level][np.logical_not(idx_used)]
            labels_tmp = labels[idx_level][np.logical_not(idx_used)]
            int_idx = np.random.choice(a=range(len(labels_tmp)), size=n_data_teacher, replace=False) \
                if j < n_partitions - 1 else np.arange(len(labels_tmp), dtype=int)
            idx_teacher = np.array([False] * len(labels_tmp))
            idx_teacher[int_idx] = True
            mapping_t2p.append(list(np.arange(len(epsilons))[idx_level][np.logical_not(idx_used)][idx_teacher]))
            x_train = features_tmp[idx_teacher] if j < n_partitions - 1 else features_tmp
            y_train = to_categorical(labels=labels_tmp[idx_teacher], n_classes=n_classes)
            idx_used[np.logical_not(idx_used)] = np.logical_or(idx_used[np.logical_not(idx_used)], idx_teacher)
            subsets.append((x_train, y_train))
    return subsets, mapping_t2p


def set_epsilons(data, epsilons, distribution, n_private, prng):
    """
    This method determines the privacy specification of each sensitive data point according to the given distribution of
    privacy specifications.
    :param data:            array of all sensitive data
    :param epsilons:        list of different privacy specifications
    :param distribution:    dict that contains one tuple for each label which contains one ratio for each epsilon
    :param n_private:       int for the number of sensitive data
    :param prng:            RandomState that determines the pseudo-random number generation
    :return:                array that contains the privacy budget of each sensitive data point
    """
    n_eps = len(epsilons)
    assert all([len(distribution[label]) == n_eps for label in distribution.keys()])
    assert all([sum(distribution[label]) == 1 for label in distribution.keys()])
    eps_array = np.ones(n_private)
    for i, label in enumerate(distribution.keys()):
        idx = data.index if label not in data['label'] else data[data['label'] == label].index
        n = len(idx)
        for j, ratio in enumerate(distribution[label]):
            if ratio == 0:
                continue
            if j == n_eps - 1:
                eps_array[idx] *= epsilons[j]
            else:
                idx_tmp = data.iloc[idx].sample(n=int(round(n * ratio)), random_state=prng).index
                eps_array[idx_tmp.array] *= epsilons[j]
                idx = idx.difference(idx_tmp)
    return eps_array


def save_mappings(seed, n_teachers, epsilons, mapping, collector, budgets, distribution):
    """
    This method stores the allocation of teachers to sensitive data and the budgets of the latter.
    :param seed:            int for the random seeding
    :param n_teachers:      int for the number of teachers
    :param epsilons:        list of privacy specifications corresponding to all sensitive data points
    :param mapping:         list that contains each a list of all data points corresponding to one teacher for each
                            teacher
    :param collector:       Collector that is used for personalized privacy accounting
    :param budgets:         list of all different actual privacy budgets corresponding to the sensitive data
    :param distribution:    list of relative ratios corresponding to the different privacy budgets
    :return:                None
    """
    teachers_dir = 'teachers'
    if not os.path.isdir(teachers_dir):
        os.mkdir(teachers_dir)
    teachers_dir = teachers_dir + f'/{n_teachers}_teachers_{budgets}_{distribution}'
    if not os.path.isdir(teachers_dir):
        os.mkdir(teachers_dir)
    seed_dir = teachers_dir + f'/seed_{seed}'
    if not os.path.isdir(seed_dir):
        os.mkdir(seed_dir)
    aggregator_dir = seed_dir + f'/{collector}'
    if not os.path.isdir(aggregator_dir):
        os.mkdir(aggregator_dir)
    with open(file=f'{aggregator_dir}/mapping.pickle', mode='wb') as f:
        pickle.dump(mapping, f)
    with open(file=f'{aggregator_dir}/budgets.pickle', mode='wb') as f:
        pickle.dump(epsilons, f)


def rdp_to_dp(alphas, epsilons, delta):
    """
    This method computes the DP costs for a given delta corresponding to given RDP costs.
    :param alphas:      list of RDP orders
    :param epsilons:    list of RDP costs corresponding to alphas
    :param delta:       float that specifies the precision of DP costs
    :return:            tuple that contains the computed DP costs, and the corresponding best RDP order
    """
    assert len(alphas) == len(epsilons)
    eps = np.array(epsilons) - math.log(delta) / (np.array(alphas) - 1)
    idx_opt = np.argmin(eps)
    return eps[idx_opt], alphas[idx_opt]



class Collector:

    def __init__(self, n_classes: int, n_teachers: int, sigma: float, seed: int):
        """
        This is an abstract class to determine methods to collect teachers' votes and to account (individual) privacy
        costs.

        :param n_classes:   number of classes corresponding to the data distribution
        :param n_teachers:  number of classifiers that were trained on sensitive data
        :param sigma:       standard deviation of noise added to vote aggregates to enforce privacy
        :param seed:        seed for pseudo random number generator to create reproducible randomness
        """
        self.n_classes = n_classes
        self.sigma = sigma
        self.seed = seed
        self.prng = np.random.RandomState(seed)
        self.n_teachers = n_teachers
        self.mapping_c2p = None
        self.groups = None
        self.budgets_g = None
        self.mapping_c2g = None

    def prepare(self, budgets, mapping):
        """
        :param budgets:     list of one (epsilon, delta)-DP epsilon per sensitive data point
        :param mapping:     list of lists: each inner list contains indices of all data points learned by one teacher
        :return:            None
        """
        raise NotImplementedError

    def collect_votes(self, x, teachers, mapping, predictions=None):
        raise NotImplementedError

    def tight_bound(self, votes, alphas):
        raise NotImplementedError

    @staticmethod
    def loose_bound_personalized(sigma, alphas, sensitivities):
        raise NotImplementedError

    @staticmethod
    def loose_bound(sigma, alphas):
        raise NotImplementedError


class GNMax(Collector):

    def __init__(self, n_classes: int, n_teachers: int, sigma: float, seed: int):
        """
        This class implements the Gaussian NoisyMax (GNMax) mechanism from Papernot et al.: 'Scalable Private Learning
        with PATE' 2018.

        :param n_classes:   number of classes corresponding to the data distribution
        :param n_teachers:  number of classifiers that were trained on sensitive data
        :param sigma:       standard deviation of noise added to vote aggregates to enforce privacy
        :param seed:        seed for pseudo random number generator to create reproducible randomness
        """
        super().__init__(n_classes=n_classes, n_teachers=n_teachers, sigma=sigma, seed=seed)
        self.sensitivities = None

    def prepare(self, budgets, mapping):
        self.budgets_g = np.array([min(budgets)])
        self.sensitivities = np.array([1])
        self.groups = np.array([0])
        self.mapping_c2g = self.groups

    def collect_votes(self, x, teachers, mapping, predictions=None):
        votes = np.zeros(self.n_classes)
        if predictions is not None:
            for i in range(self.n_classes):
                votes[i] = sum(predictions == i)
        else:
            for teacher in teachers:
                votes[teacher.predict(x)] += 1
        return votes

    def tight_bound(self, votes, alphas):
        epsilons = np.zeros((len(self.sensitivities), len(alphas)))
        for i, sensitivity in enumerate(self.sensitivities):
            sigma = self.sigma / sensitivity
            vote_counts = votes / sensitivity
            log_q = compute_logq_gaussian(counts=vote_counts, sigma=sigma)
            assert log_q <= 0 < sigma and np.all(alphas > 1)
            if np.isneginf(log_q):
                return np.full_like(alphas, 0., dtype=np.float)
            alpha2 = math.sqrt(sigma ** 2 * -log_q)
            alpha1 = alpha2 + 1
            epsilons[i] = self.loose_bound(alphas=alphas, sigma=sigma)
            mask = np.logical_and(alpha1 > np.atleast_1d(alphas), alpha2 > 1)
            epsilons1 = self.loose_bound(alphas=[alpha1], sigma=sigma)
            epsilons2 = self.loose_bound(alphas=[alpha2], sigma=sigma)
            log_a2 = (alpha2 - 1) * epsilons2
            if (np.any(mask)
                    and log_q <= log_a2 - alpha2 * (math.log(1 + 1 / (alpha1 - 1)) + math.log(1 + 1 / (alpha2 - 1)))
                    and -log_q > epsilons2):
                log1q = _log1mexp(log_q)
                log_a = (alphas - 1) * (log1q - _log1mexp((log_q + epsilons2) * (1 - 1 / alpha2)))
                log_b = (alphas - 1) * (epsilons1 - log_q / (alpha1 - 1))
                log_s = np.logaddexp(log1q + log_a, log_q + log_b)
                epsilons[i][mask] = np.minimum(epsilons[i], log_s / (alphas - 1))[mask]
            assert np.all(np.array(epsilons[i]) >= 0)
        return epsilons

    @staticmethod
    def loose_bound_personalized(sigma, alphas, sensitivities):
        return np.array([[s ** 2 * alpha / sigma ** 2 for alpha in alphas] for s in sensitivities])

    @staticmethod
    def loose_bound(sigma, alphas):
        return np.array([alpha / sigma ** 2 for alpha in alphas])



class Aggregator:

    def __init__(self, collector: Collector, seed: int):
        """
        This is an abstract class to determine methods to perturb the aggregate of teachers' votes and to select which
        labels are taken.

        :param collector:   method to collect teachers' votes and to account the (individual) privacy cost
        :param seed:        seed for pseudo random number generator to create reproducible randomness
        """
        self.collector = collector
        self.n_classes = collector.n_classes
        self.seed = seed
        self.prng = np.random.RandomState(seed)

    def aggregate_votes(self, votes, alphas):
        """
        :param votes:   array that contains the vote count of each class
        :param alphas:  array of floats > 1 that determine the orders of RDP bounds
        :return:        label:      int that corresponds to a specific class
                        epsilons:   RDP costs (for each order in alphas) per group (sensitive data points that have the
                                    same privacy budget)
        """
        raise NotImplementedError

    def vote(self, x, teachers, alphas, mapping, predictions=None):
        """
        :param x:           feature to create a label for
        :param teachers:    list of classifiers that were trained on sensitive data
        :param alphas:      list of orders for calculation of RDP costs
        :param mapping:     list of teachers or data points that belong to the same group (share same privacy budget)
        :param predictions: array of teachers' predictions if PATE is terminated by reaching specific number of labels
        :return:            label:      int that corresponds to a specific class
                            epsilons:   RDP costs (for each order in alphas) per group (sensitive data points that have
                                        the same privacy budget)
        """
        votes = self.collector.collect_votes(x=x, teachers=teachers, mapping=mapping, predictions=predictions)
        return self.aggregate_votes(votes=votes, alphas=alphas)


class DefaultGNMax(Aggregator):

    def __init__(self, collector: GNMax, seed: int):
        """
        This class implements the standard aggregation method of teachers' votes that only adds Gaussian noise.

        :param collector:   method to collect teachers' votes and to account the (individual) privacy cost
        :param seed:        seed for pseudo random number generator to create reproducible randomness
        """
        super().__init__(collector=collector, seed=seed)

    def aggregate_votes(self, votes, alphas):
        """
        :param votes:   array that contains the vote count of each class
        :param alphas:  array of floats > 1 that determine the orders of RDP bounds
        :return:        label:      int that corresponds to a specific class
                        epsilons:   RDP costs (for each order in alphas) per group (sensitive data points that have the
                                    same privacy budget)
        """
        noise = self.prng.normal(loc=0.0, scale=self.collector.sigma, size=self.n_classes)
        epsilons = np.array(self.collector.tight_bound(votes=votes, alphas=alphas))
        label = np.argmax(votes + noise)
        return label, epsilons


class ConfidentGNMax(Aggregator):

    def __init__(self, collector: GNMax, seed: int, sigma: float, t: float):
        """
        This class implements the confident aggregation method of teachers' votes that adds Gaussian noise and selects
        labels by privately checking if there is a consensus between the teachers.

        :param collector:   method to collect teachers' votes and to account the (individual) privacy cost
        :param seed:        seed for pseudo random number generator to create reproducible randomness
        :param sigma:       standard deviation of Gaussian noise that is used to check if the teachers consent
        :param t:           threshold to determine if the teachers consent
        """
        super().__init__(collector=collector, seed=seed)
        self.sigma = sigma
        self.t = t

    def aggregate_votes(self, votes, alphas):
        """
        :param votes:   array that contains the vote count of each class
        :param alphas:  array of floats > 1 that determine the orders of RDP bounds
        :return:        label:      int that corresponds to a specific class; -1 if there is no teachers' consensus
                        epsilons:   RDP costs (for each order in alphas) per group (sensitive data points that have the
                                    same privacy budget)
        """
        noise1 = self.prng.normal(loc=0.0, scale=self.sigma, size=1)
        noise2 = self.prng.normal(loc=0.0, scale=self.collector.sigma, size=self.n_classes)
        label = -1
        epsilons = self.collector.loose_bound_personalized(sigma=self.sigma, alphas=alphas,
                                                           sensitivities=self.collector.sensitivities) / 2
        if np.max(votes) + noise1 >= self.t:
            label = np.argmax(votes + noise2)
            epsilons += self.collector.tight_bound(votes=votes, alphas=alphas)
        return label, epsilons



if __name__ == "__main__":
    app.run(main)