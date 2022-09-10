"""
Plot the saved error data
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import pickle

tf.random.set_seed(1234)

font = {'size': 14, 'family': 'DeJavu Serif', 'serif': ['Palatino']}
plt.rc('font', **font)
params = {'legend.fontsize': 14, 'lines.markersize': 9, 'backend': 'Qt5Agg'}
plt.rcParams.update(params)
plt.rc('text', usetex=True)


def sqrerrn(yTrue, yPred):
    errn = np.sum(np.sqrt(np.sum((yTrue - yPred) ** 2, axis=1)))  # / np.sum(yTrue) # Latest
    errn = errn / (np.shape(yTrue)[0] * np.shape(yTrue)[1])
    # errn = np.sum(np.sqrt((yTrue - yPred) ** 2)) / np.sum(yTrue)  # Previous
    return errn


# Work folder
work_folder = os.getcwd()
if not os.path.exists(work_folder + '/plots/'):
    os.makedirs(work_folder + '/plots/')

# Load the predicted and observed weights (VALIDATION)
file = open(work_folder + "/results/y_val_seq.pkl", "rb")
yval = pickle.load(file)
file2 = open(work_folder + "/results/y_val_pred_seq.pkl", "rb")
yvalpred = pickle.load(file2)

# (TESTING)
file3 = open(work_folder + "/results/y_test_seq.pkl", "rb")
ytest = pickle.load(file3)
file4 = open(work_folder + "/results/y_test_pred_seq.pkl", "rb")
ytestpred = pickle.load(file4)
print()

with open(work_folder + '/results/lenTrains.pkl', 'rb') as file_pi:
    lenTrains = pickle.load(file_pi)

vals_val = list()
for i in range(len(yval)):
    vals = sqrerrn(yval[i], yvalpred[i])
    vals_val.append((np.mean(vals)))

plt.plot(np.array(lenTrains), vals_val, "-o", ms=4, color='k', label="Validation")

inv_counter = list()
vals_test = list()
for i in range(len(ytest)):
    if np.mean(ytest[i]) != -99:
        vals = sqrerrn(ytest[i], ytestpred[i])
        vals_test.append((np.mean(vals)))
        inv_counter.append(lenTrains[i])

plt.plot(inv_counter, vals_test, "-x", ms=8, color='b', label="Test")
plt.loglog(base=10)
plt.xlabel("training sets")
plt.ylabel(r"$\epsilon$")
plt.grid(visible=True, which='major', color='k', linestyle='-')
plt.grid(visible=True, which='minor', color='k', linestyle='--')
lgnd = plt.legend(loc='upper right')
for j in range(len(lgnd.legendHandles)):
    lgnd.legendHandles[j].set_markersize(6)
plt.tight_layout()
plt.savefig(work_folder + "/plots/err_vs_sim.png", dpi=300)
plt.close()

print("plotted !!")
