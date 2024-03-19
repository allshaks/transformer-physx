import csv
import os
import numpy as np
import matplotlib.pyplot as plt
import h5py

#######
### Goal: 
# Create 200 (later 6000) datasets applying gaussian noise to the trial-averaged data on a channel-based level for each subject. 
# Results will be stored in a single .h5-file (for each subject)
# File contains 200 (later 6000) datasets, each with 256 channels and 369 time points
#######
### Next steps:
#   - Fix scale issues
#   - Play around with the standard deviation to find tolerable limit for noise
#   - Create datasets for all 11 subjects
#######

# Function to read the CSV file and return the data
def read_csv(file_path):
    with open(file_path, 'r') as f:
        reader = csv.reader(f)
        data = list(reader)
    return data

def get_data_points(data, selected_indices, noise=False):
    num_time_points = len(data[0]) - 1  # Assuming all rows have the same number of columns
    time_points = np.array(range(num_time_points))
    time_points = time_points*1000/2048-50

    labels=[]
    signals = []
    
    valid_indices = [index for index in selected_indices if index < len(data)]
    for index in valid_indices:
        time_series = np.array([float(value) for value in data[index][1:]])
        labels.append(data[index][0])
        if noise:
            time_series = add_gaus_noise(time_series,0,0.5)
        signals.append(time_series)

    if signals:
        # each element in signals is an electrode containing one entry for each of the 369 time slots
        signals = np.array(signals)
        #print(len(signals))
        #print(len(signals[0]))        
    
        return (time_points, signals)
    else:
        print("No valid indices provided for MGFP calculation.")

def add_gaus_noise(y_coord, mean=0, standard_dev = 0.1):
    noise = np.random.normal(mean, standard_dev, len(y_coord))
    y_noise = y_coord + noise
    return y_noise

def save_to_h5(file_name, data_array, dataset_name):
    file_path = './data/Somatosensory/HDF5/' + file_name
    # 'w' to overwrite, 'a' to add multiple files, if error occurs, delete existing files with same name
    with h5py.File(file_path, 'a') as file:     
        dset = file.create_dataset(dataset_name, data = data_array)

def open_h5(file_name,dataset_name):
    file_path = './data/Somatosensory/HDF5/'
    filename = file_path + file_name
    dataset_name = dataset_name
    with h5py.File(filename, 'r') as file:
        data_set = file[dataset_name]
        data=data_set[:369]
    return data

def main():
    #print(f"Current Working Directory: {os.getcwd()}")
    s = '06' # subject
    file_path = './data/Somatosensory/PlosOne/' + s + '_SEP_prepro_-50_130ms.csv'
    data = read_csv(file_path)
    selected_indices = list(range(256)) 

    filename = "dataset_sub" + s
    # number of trials we create artificially
    num_datasets = 1000
    ds_name = "data_noise_"

    all_noisy_rms = []
    # saving the two generated noisy data simulations in one h5 file, dataset_name is used to differentiate the files
    for i in range(num_datasets):
        ds = ds_name + str(i)
        _, all_channel_data_noisy = get_data_points(data, selected_indices, noise=True)
        save_to_h5(filename, all_channel_data_noisy, ds)
        all_noisy_rms.append(np.sqrt(np.mean(np.square(all_channel_data_noisy), axis=0)))
    print(len(all_noisy_rms))
    print(len(all_noisy_rms[0]))

    # range of the data plotted (used to be in get_data_points function)
    ms_range = (-5, 38)
    start, end = ms_range
    ms_start = (50 + start)*2
    ms_end = (50 + end)*2 
    
    if False: #switch on to show plots for checking the created noisy data
        # check some of the datasets by retrieving + plotting
        for i in [0, 1, 56, 520, 999]:
            noisy_data = open_h5(filename, ds_name + str(i))
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
    # Problems:
    # - works overall but something is wrong with the scale (artificial data has higher values)
    #   -> first taking mean than RMS should solve the problem
            
    averages_noisy_rms = np.mean(all_noisy_rms, axis=0)
    print(all_noisy_rms[0][0])
    print(len(averages_noisy_rms))
    print(averages_noisy_rms[0])
    
    num_time_points = len(averages_noisy_rms) - 1 
    time_points = np.array(range(num_time_points))
    time_points = time_points*1000/2048-50
    
    fig, ax = plt.subplots(2,1)
    ax[0].plot(time_points[ms_start:ms_end], averages_noisy_rms[ms_start:ms_end])
    ax[0].set_title(f"Noisy trial-averaged RMS for subject " + s)
    ax[0].grid()
    
    # compare to original trial-average rms:
    _, all_channel_data = get_data_points(data, selected_indices, noise=False)
    y_coordinates = np.sqrt(np.mean(np.square(all_channel_data), axis=0))  
    ax[1].plot(time_points[ms_start:ms_end], y_coordinates[ms_start:ms_end])
    ax[1].set_title(f"Original trial-averaged RMS for subject " + s)
    ax[1].grid()
    plt.show()

if __name__ == "__main__":
    main()