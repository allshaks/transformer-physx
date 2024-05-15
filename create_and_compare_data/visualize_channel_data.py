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

# Function to read the CSV file and return the data
def read_csv(file_path):
    with open(file_path, 'r') as f:
        reader = csv.reader(f)
        data = list(reader)
    return data

def get_data_points(data, selected_channels=256, selected_time_points=369):
    #num_time_points = len(data[0]) - 1  # Assuming all rows have the same number of columns
    selected_channels = list(range(selected_channels)) 
    num_time_points = selected_time_points
    time_points = np.array(range(num_time_points))
    time_points = time_points*1000/2048-50

    labels=[]
    signals = []
    
    valid_indices = [index for index in selected_channels if index < len(data)]
    for index in valid_indices:
        time_series = np.array([float(value) for value in data[index][1:]])
        labels.append(data[index][0])
        signals.append(time_series)

    if signals:
        # each element in signals is one channel containing one entry for each of the time points
        signals = np.array(signals)      
        return (time_points, signals)
    else:
        print("No valid indices provided.")


s = '06' # subject
filename = "dataset_sub" + s + "_data_noise_val.h5"

len_ds = 180 # manually set to length of validation data
selected_channels = [0, 100, 255] # manually set channel to plot

num_time_points = 369 - 1  # Assuming all rows have the same number of columns
time_points = np.array(range(num_time_points))
time_points = time_points*1000/2048-50
 
# channel data contains exactly one channel of every trial
# it has 180 entries (1 per trial) with each 369 elements (time points) 

#######
## All trils and trial averages of individual channels in comparable plot view 
if False:
    fig, axs = plt.subplots(3,2, sharey='row')
    for i, chan in enumerate(selected_channels):
        channel_data = np.empty((len_ds, 369)) # 369 time points
        for j in range(180):
            ds_name = "data_noise_val" + str(j)
            trial_data = open_h5(filename, ds_name)
            channel_data[j] = trial_data[chan]
        trial_averaged_channel_data = np.mean(channel_data, axis = 0)

        # plot the single channel for all the trials
        axs[i][0].plot(channel_data.T)
        axs[i][0].set_title(f"All trials data for channel {chan} of subject " + s)
        # plot the trial-average of the single channel  
        axs[i][1].plot(trial_averaged_channel_data)
        axs[i][1].set_title(f"Trial-averaged data for channel {chan} of subject " + s)

    fig.tight_layout()
    plt.show()


    #######
    ## Trial averages of all channels of artificial data in butterfly plot
if False:
    trial_averaged_channels = []
    for i, chan in enumerate(range(255)): # all channels
        channel_data = np.empty((len_ds, 369)) # 369 time points
        for j in range(180):
            ds_name = "data_noise_val" + str(j)
            trial_data = open_h5(filename, ds_name)
            channel_data[j] = trial_data[chan]
        trial_averaged_channels.append(np.mean(channel_data, axis = 0))

    fig, ax = plt.subplots(1,1)

    # plot the single channel for all the trials
    ax.plot(trial_averaged_channels)
    ax.set_title(f"Artificial averaged data for all channels of subject " + s)

    fig.tight_layout()
    plt.show()

    #######
    ## Trial averages of all channels of original (average) data in butterfly plot
if True:
    s = '06' # subject
    file_path = './data/Somatosensory/PlosOne/' + s + '_SEP_prepro_-50_130ms.csv'

    org_avg_data = read_csv(file_path)
    _, org_avg_data = get_data_points(org_avg_data)

    fig, ax = plt.subplots(1,1)
    # plot the single channel for all the trials
    ax.plot(org_avg_data.T)
    ax.set_title(f"Original averaged data for all channels of subject " + s)


    fig.tight_layout()
    plt.show()

    ######
    ## Singular Value Decomposition
    # Perform SVD
    U, s, V = np.linalg.svd(org_avg_data)

    # Plot the singular values
    plt.plot(s, 'o-')
    plt.title('Singular Values')
    plt.xlabel('Singular Value Index')
    plt.ylabel('Singular Value Magnitude')
    plt.grid(True)
    plt.show()