"""
(5)
To be used after 'rnn_step14.py'
Program to use the trained ROM for generating
artificial simulations.
"""

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import load_model
import os
import pyvista as pv

tf.random.set_seed(1234)
font = {'size': 18, 'family': 'DeJavu Serif', 'serif': ['Palatino']}
plt.rc('font', **font)
params = {'legend.fontsize': 14, 'lines.markersize': 4}
plt.rcParams.update(params)
plt.rc('text', usetex=True)


def useCPUforTraining(switch):
    if switch:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    elif not switch:
        print("Using GPU : Nvidia M1200\n")
    else:
        print("Invalid option!!")


useCPUforTraining(True)


# Multistep predictions function
def multistep_predict(latest_model, tstart, tstop, starting_weights, BCs):
    n_pred_wts = list()
    times_ = list()
    if tstart < 0:  # or tstop > BCs.shape[0] - tstart:
        print("Check tstart and tstop\n")
    elif tstart <= tstop:
        x_i_minus_1 = starting_weights
        for i in range(tstart, tstop, ):  # int((tstop - tstart) / 70)):
            print(i)
            times_.append(i)
            x_i_minus_1 = x_i_minus_1.reshape([1, -1])
            x_i_minus_1 = np.expand_dims(x_i_minus_1, axis=[1])
            y_pred_i = latest_model.predict(x_i_minus_1)
            n_pred_wts.append(y_pred_i)  # y -> 0
            x_i_minus_1 = np.hstack([np.squeeze(y_pred_i), BCs[i, :]])
            if i == BCs.shape[0] - 1:  # detect last row --> predict last of the ys
                x_i_minus_1 = x_i_minus_1.reshape([1, -1])
                x_i_minus_1 = np.expand_dims(x_i_minus_1, axis=[1])
                y_pred_i = latest_model.predict(x_i_minus_1)
                n_pred_wts.append(y_pred_i)
    else:
        print('\nCannot predict when tstart >= tstop --> Correct it!')

    n_pred_wts = np.reshape(n_pred_wts, [len(n_pred_wts), -1])
    return n_pred_wts, np.array(times_)


def denormalize(y_pred_):
    rbs, mms = rbs_wts, mms_wts
    mms_y_pred = mms.inverse_transform(y_pred_)
    rbs_y_pred = rbs.inverse_transform(mms_y_pred)
    state_y_pred = pca_.inverse_transform(rbs_y_pred).T
    return state_y_pred


def saveReconMesh(vlwts, totalsteps, simname, vars_present, var_shape, currentpath, saveMesh=1):
    if saveMesh:
        smesh = pv.read(currentpath + "/analysis/sample.vtu")

        # indices for point and cell data
        pidx = [x for x in range(0, (len(vars_present[0]) + 1) * var_shape[0][1], var_shape[0][1])]
        cidx = [x for x in
                range(pidx[-1], (len(vars_present[1]) + 1 + var_shape[0][0]) * var_shape[1][1], var_shape[1][1])]
        pcidx = np.hstack([pidx, cidx[1:]])

        for tstep in range(vlwts.shape[0]):
            singleset = denormalize(vlwts[tstep, :].reshape(1, -1))
            print("Writing {}th step out of {} steps".format(tstep, totalsteps))
            changemesh = smesh.copy()
            for j, var_ in enumerate(vars_present[0] + vars_present[1]):
                if j < len(vars_present[0]):
                    changemesh.point_data[var_] = singleset[pcidx[j]:pcidx[j + 1], 0]
                elif j >= len(vars_present[0]):
                    changemesh.cell_data[var_] = singleset[pcidx[j]:pcidx[j + 1], 0]
            if tstep > 0:
                savethis = currentpath + "/recon" + "/Forecast_" + simname + "_{:04d}.vtu".format(tstep)
                changemesh.save(savethis)
    return None


# ----------------------------------------------------------------------------------------------------------------------
# Create save folders
work_folder = os.getcwd()
if not os.path.exists(work_folder + '/recon/'):
    os.makedirs(work_folder + '/recon/')

# Load BCs corresponding to the "Fast" case
# bc_ = work_folder + "/analysis/Fast.txt"
lenOfSim = "Fast"
# bcvals_ = np.loadtxt(fname=bc_, delimiter=" ")
bcvals_ = np.load(work_folder + "/analysis/fast.npy")
bcvals_shift = np.vstack([bcvals_[:, 3], bcvals_[:, 1], bcvals_[:, 2], bcvals_[:, 0]]).T
bcvals_shift = bcvals_shift[:600000, :]

# Load the initial weights
start_wts_ = np.load(work_folder + "/analysis/startwts.npy").reshape(-1, 1)
pca_ = np.load(work_folder + "/analysis/u/pca_.npy", allow_pickle=True)[0]
sim101 = pca_.transform(start_wts_.T)

# Load other prerequisites
state_vars = np.load(work_folder + '/normalized/state_.npy', allow_pickle=True)
wts = state_vars[0]
rbs_wts = state_vars[2]
mms_wts = state_vars[3]
rbs_bcs = state_vars[4]
mms_bcs = state_vars[5]
simnames = state_vars[6]

# Load trained model
rnn = load_model(filepath=work_folder + "/analysis/rnn_permute_59_.h5", compile=False)

# Preparing and structuring the BCs for prediction
bcs_normalised = mms_bcs.transform(rbs_bcs.transform(bcvals_shift))
present = bcs_normalised[:-1, :]
future = bcs_normalised[1:, :]
bcs_ori = np.hstack([present, future])
bcs = bcs_ori[np.arange(0, bcs_ori.shape[0], 3), :]

# Simulate fast case
startwts = np.hstack([sim101[0, :], bcs[0, :]])

# Call the predictor
forecast_moderate, times_moderate = multistep_predict(latest_model=rnn,
                                                      tstart=1, tstop=bcs.shape[0] - 1,
                                                      starting_weights=startwts,
                                                      BCs=bcs)
forecast_moderate = forecast_moderate[np.arange(0, forecast_moderate.shape[0], int(forecast_moderate.shape[0] / 60))]

print("Weights predicted!!")

# Load auxiliary variables
varnames = np.load(work_folder + "/analysis/varnames.npy", allow_pickle=True)
varshape = np.load(work_folder + "/analysis/varsShape.npy", allow_pickle=True)
varshape[0][1] = varshape[0][1] - 2
varshape[1][1] = varshape[1][1] - 2

"""Save Meshes"""
saveReconMesh(forecast_moderate,
              np.max(times_moderate),
              lenOfSim,
              varnames,
              varshape,
              work_folder)

print("Done!")
