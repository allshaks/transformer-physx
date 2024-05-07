import csv
import os
import numpy as np
import matplotlib.pyplot as plt
import h5py
import logging

logger = logging.getLogger(__name__)

def open_h5(file_name,dataset_name):
    file_path = './data/Somatosensory/HDF5/'
    filename = file_path + file_name
    dataset_name = dataset_name
    with h5py.File(filename, 'r') as file:
        data_set = file[dataset_name]
        #make sure we only use the 369 time points of the data
        data=data_set[:369]
    return data



s = '06' # subject
filename = "dataset_sub" + s + "_data_noise_val.h5"

# range of the data plotted
ms_range = (-5, 38)
start, end = ms_range
ms_start = (50 + start)*2
ms_end = (50 + end)*2 
len_ds = 180 # manually set to length of validation data
selected_channels = [0, 100, 255] # manually set channel to plot

num_time_points = 369 - 1  # Assuming all rows have the same number of columns
time_points = np.array(range(num_time_points))
time_points = time_points*1000/2048-50
fig, axs = plt.subplots(3,2, sharey='row')
    
# channel data contains exactly one channel of every trial
# it has 180 entries (1 per trial) with each 369 elements (time points) 

set_x_limits = False
for i, chan in enumerate(selected_channels):
    channel_data = np.empty((len_ds, 369)) # 369 time points
    for j in range(180):
        ds_name = "data_noise_val" + str(j)
        trial_data = open_h5(filename, ds_name)
        channel_data[j] = trial_data[chan]
    trial_averaged_channel_data = np.mean(channel_data, axis = 0)

    if set_x_limits:
        # plot the single channel for all the trials
        axs[i][0].plot(time_points[ms_start:ms_end], channel_data.T[ms_start:ms_end])
        axs[i][0].set_title(f"All trials data for channel {chan} of subject " + s)
        # plot the trial-average of the single channel  
        axs[i][1].plot(time_points[ms_start:ms_end], trial_averaged_channel_data[ms_start:ms_end])
        axs[i][1].set_title(f"Trial-averaged data for channel {chan} of subject " + s)

    else:
        # plot the single channel for all the trials
        axs[i][0].plot(channel_data.T)
        axs[i][0].set_title(f"All trials data for channel {chan} of subject " + s)
        # plot the trial-average of the single channel  
        axs[i][1].plot(trial_averaged_channel_data)
        axs[i][1].set_title(f"Trial-averaged data for channel {chan} of subject " + s)

fig.tight_layout()
plt.show()