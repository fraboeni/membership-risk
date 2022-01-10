import matplotlib.pyplot as plt
import os
import pickle
import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = ""

USE_SOFTMAX = False
USE_EPS = False

MODEL_ID = 1004
dataset = 'cifar10'

if USE_EPS:
    result_dir = os.getcwd() + '/../log_dir/result_data/exp01/eps_'
    SCALES = [0.1, 0.2, 0.5, 0.9, 1., 1.5]
else:
    result_dir = os.getcwd() + '/../log_dir/result_data/exp01/scale_'
    SCALES = [0.005, 0.01, 0.02, 0.05, 0.1, 0.2]

if USE_SOFTMAX:
    result_dict = pickle.load(open(result_dir + '{id}_softmax_metrics.pkl'.format(id=MODEL_ID), 'rb'))
else:
    result_dict = pickle.load(open(result_dir + '{id}_metrics.pkl'.format(id=MODEL_ID), 'rb'))

# extract general information
num_samples = result_dict['num_samples']
train_acc = result_dict['train_acc']
test_acc = result_dict['test_acc']



# generate
fig, axs = plt.subplots(len(SCALES), 5, figsize=(18, 18), sharey='row')
strings = """   Dataset: {ds},
                Model-acc-on-members: {train_acc},
                Model-acc-on-non-members: {test_acc},
                Number of Datapoints: {num} (member and non-member respectively), 
                For 100 generated neighbors per data point.
                """.format(ds=dataset,
                           num=num_samples,
                           train_acc=np.round(train_acc, 3),
                           test_acc=np.round(test_acc, 3),
                           )
fig.suptitle(strings)

PLOTS = ['wasserstein', 'msel2', 'variances', 'means', 'stds']
for i, scale in enumerate(SCALES):
    for j, plot_name in enumerate(PLOTS):  # the 5 plots

        ax = axs[i, j]

        sub_dict = result_dict[scale]
        (train_res, test_res) = sub_dict[plot_name]

        train_acc_neighbors = sub_dict['train_acc_neigh']
        test_acc_neighbors = sub_dict['test_acc_neigh']

        if USE_EPS:
            strings = """ L2-eps:{scale}, metric: {metric}
                                    """.format(scale=scale,
                                               metric=plot_name,
                                               )
        else:
            
            strings = """ Noise Scale:{scale}, metric:{metric}
                            """.format(scale=scale,
                                       metric=plot_name,
                                       )
        if USE_SOFTMAX:
            ax.set_yscale('log')
        ax.set_title(strings)

        legend = ['member', 'non-member']
        plotting_variables = [train_res, test_res]
        ax.hist(plotting_variables, bins=50, label=legend)
        ax.legend()

fig.text(0.5, 0.001, 'metric over data points and their respective neighbors', ha='center')
fig.text(0.001, 0.5, 'number of data points', va='center', rotation='vertical')
fig.tight_layout()

if USE_EPS:
    output_dir = os.getcwd() + "/01_eps_"
else:
    output_dir = os.getcwd() + "/01_scale_"

if USE_SOFTMAX:
    plt.savefig(output_dir + "{id}_softmax.png".format(id=MODEL_ID))
else:
    plt.savefig(output_dir + "{id}.png".format(id=MODEL_ID))
plt.show()
