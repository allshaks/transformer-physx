import csv
import os
import numpy as np
import matplotlib.pyplot as plt
import h5py
import logging

logger = logging.getLogger(__name__)

#######
### Goal: 
# Create training, test and validation data set
# Remain flexible in sample size and time point size 
# (org. paper used only a fraction of the time points for training), see compare_hdf5_files copy.py
# For start create a total of 6000 samples and split it accordig to proportions that org. physx-paper used:
#   - ~11% training data (= 660 samples)
#   - ~86% test data (= 5160 samples)
#   - ~3% validation data (= 180 samples)
#######

# Function to read the CSV file and return the data
def read_csv(file_path):
    with open(file_path, 'r') as f:
        reader = csv.reader(f)
        data = list(reader)
    return data

def get_data_points(data, selected_channels=256, selected_time_points=369, noise=False):
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
        if noise:
            time_series = add_gaus_noise(time_series,0,0.02)
        signals.append(time_series)

    if signals:
        # each element in signals is an electrode containing one entry for each of the time points
        signals = np.array(signals)
        #print(len(signals))
        #print(len(signals[0]))        
        return (time_points, signals)
    else:
        print("No valid indices provided.")

def add_gaus_noise(y_coord, mean=0, standard_dev = 0):
    noise = np.random.normal(mean, standard_dev, len(y_coord))
    y_noise = y_coord + noise
    return y_noise

def save_to_h5(file_name, data_array, dataset_name, train_test_val=None):
    if train_test_val:
        file_path = './data/Somatosensory/HDF5/' + file_name + "_" + train_test_val + ".h5"
    else:
        file_path = './data/Somatosensory/HDF5/' + file_name + ".h5"
    # 'w' to overwrite, 'a' to add multiple files, if error occurs, delete existing files with same name
    with h5py.File(file_path, 'a') as file:     
        dset = file.create_dataset(dataset_name, data = data_array)

def open_h5(file_name,dataset_name):
    file_path = './data/Somatosensory/HDF5/'
    filename = file_path + file_name
    dataset_name = dataset_name
    with h5py.File(filename, 'r') as file:
        data_set = file[dataset_name]
        #make sure we only use the 369 time points of the data
        data=data_set[:369]
    return data

s = '02' # subject
file_path = './data/Somatosensory/PlosOne/' + s + '_SEP_prepro_-50_130ms.csv'
data = read_csv(file_path)
selected_channels = 256
selected_time_points = 369

filename = "dataset_sub" + s
# number of trials we create artificially
total_num_datasets = 600
num_datasets_train = int(total_num_datasets*0.11)
num_datasets_test = int(total_num_datasets*0.86)
num_datasets_val = int(total_num_datasets*0.03)
checksum = total_num_datasets == (num_datasets_test + num_datasets_train + num_datasets_val)
print("Checksum is " + str(checksum))

all_ds_names = ["data_noise_train", "data_noise_test", "data_noise_val"]
all_num_datasets = {"data_noise_train": num_datasets_train, "data_noise_test": num_datasets_test, "data_noise_val": num_datasets_val}
averages_noisy_channels = []

# saving the two generated noisy data simulations in one h5 file, dataset_name is used to differentiate the files
averages_noisy_channels = np.zeros((selected_channels,selected_time_points))
for ds_name in all_ds_names:
    print('Starting to create file for ' + ds_name)
    num_ds = all_num_datasets[ds_name]
    for i in range(num_ds):
        ds = ds_name + str(i)
        _, all_channel_data_noisy = get_data_points(data, selected_channels, selected_time_points, noise=True)
        save_to_h5(filename, all_channel_data_noisy, ds, train_test_val=ds_name)
        averages_noisy_channels += all_channel_data_noisy
        if i%25 == 0:
            print('---> Currently at dataset number ' + str(i))

averages_noisy_channels /= total_num_datasets

# range of the data plotted (used to be in get_data_points function)
ms_range = (-5, 38)
start, end = ms_range
ms_start = (50 + start)*2
ms_end = (50 + end)*2 

if True: #switch on to show plots for checking the created noisy data
    # set file name and ds_name to check test, training or validation data
    check_filename = "dataset_sub" + s + "_data_noise_train.h5"
    ds_name = "data_noise_train"
    # check some of the datasets by retrieving + plotting
    for i in [0, 1, 56, 520, 999]:
        noisy_data = open_h5(check_filename, ds_name + str(i))
        num_time_points = len(noisy_data[0]) - 1  # Assuming all rows have the same number of columns
        time_points = np.array(range(num_time_points))
        time_points = time_points*1000/2048-50
        rms_noisy_data = np.sqrt(np.mean(np.square(noisy_data), axis=0))  
        fig, axs = plt.subplots(1,2)
        # plot channel-average (RSM) of noisy data generated for a simulated first trial 
        axs[0].plot(time_points[ms_start:ms_end], rms_noisy_data[ms_start:ms_end])
        axs[0].set_title(f"Trial {i}: RSM Noisy channels subject " + s)
        # plot all channels of noisy data generated for a simulated first trial 
        axs[1].plot(time_points[ms_start:ms_end], noisy_data[ms_start:ms_end])
        axs[1].set_title(f"Trial {i}: All noisy channels subject " + s)
        fig.tight_layout()
        plt.show()

# Plot trial-average of the artificially created channels and compare it to the actual trial-average of the data
# since we use gaussian noise, the artificial average should be very close to the original one (depending on the number of trials)
# Steps:
#   1. The data we need is already present in "all_channel_data_noisy" so we take the rms there and store it.
#   2. Take average of all the rms-data stored 
#   3. Plot results and compare it to original data

avg_noisy_rms = (np.sqrt(np.mean(np.square(averages_noisy_channels), axis=0)))

num_time_points = len(avg_noisy_rms) - 1 
time_points = np.array(range(num_time_points))
time_points = time_points*1000/2048-50

fig, ax = plt.subplots(2,1)
ax[0].plot(time_points[ms_start:ms_end], avg_noisy_rms[ms_start:ms_end])
ax[0].set_title(f"Noisy trial-averaged RMS for subject " + s)
ax[0].grid()

# compare to original trial-average rms:
_, all_channel_data = get_data_points(data, selected_channels, selected_time_points, noise=False)
y_coordinates = np.sqrt(np.mean(np.square(all_channel_data), axis=0))  
ax[1].plot(time_points[ms_start:ms_end], y_coordinates[ms_start:ms_end])
ax[1].set_title(f"Original trial-averaged RMS for subject " + s)
ax[1].grid()
plt.show()
