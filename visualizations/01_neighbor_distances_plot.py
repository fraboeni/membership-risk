import matplotlib.pyplot as plt
import os
import pickle
import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = ""

dataset = 'cifar10'
result_dir = os.getcwd() + '/../log_dir/result_data/exp01/'
result_dict = pickle.load(open( result_dir+"metrics.pkl", "rb" ))

# extract general information
num_samples = result_dict['num_samples']
train_acc = result_dict['train_acc']
test_acc = result_dict['test_acc']

SCALES = [0.01, 0.05, 0.1, 0.2]

# generate
fig, axs = plt.subplots(len(SCALES), 5, figsize=(18, 18), sharey='row')
strings = """   Dataset: {ds},
                Model-acc-on-members: {train_acc},
                Model-acc-on-non-members: {test_acc},
                Number of Datapoints: {num} (member and non-member respectively), 
                For 100 generated neighbors per data point.
                """.format(ds=dataset,
                           num=num_samples,
                            train_acc=np.round(train_acc,3),
                            test_acc=np.round(test_acc,3),
                            )
fig.suptitle(strings)

PLOTS = ['wasserstein', 'msel2', 'variances', 'means', 'stds']
for i, scale in enumerate(SCALES):
    for j, plot_name in enumerate(PLOTS): # the 5 plots


        ax = axs[i, j]

        sub_dict = result_dict[scale]
        (train_res, test_res) = sub_dict[plot_name]


        train_acc_neighbors = sub_dict['train_acc_neigh']
        test_acc_neighbors = sub_dict['test_acc_neigh']

        strings = """ Noise Scale : {scale},metric: {metric}
                        """.format(scale=scale,
                                metric=plot_name,
                         )
        ax.set_title(strings)

        legend = ['member', 'non-member']
        plotting_variables = [train_res, test_res]
        ax.hist(plotting_variables, label=legend)
        ax.legend()

fig.text(0.5, 0.001, 'metric over data points and their respective neighbors', ha='center')
fig.text(0.001, 0.5, 'number of data points', va='center', rotation='vertical')
fig.tight_layout()
plt.savefig(os.getcwd() + "/01.png")
plt.show()