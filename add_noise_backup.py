import csv
import os
import numpy as np
import matplotlib.pyplot as plt
import h5py

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
    s = '05' # subject
    file_path = './data/Somatosensory/PlosOne/' + s + '_SEP_prepro_-50_130ms.csv'
    
    data = read_csv(file_path)

    selected_indices = list(range(256))

    x_coordinates, all_channel_data = get_data_points(data, selected_indices, noise=False)
    x_coordinates, all_channel_data_noise1 = get_data_points(data, selected_indices, noise=True)
    _, all_channel_data_noise2 = get_data_points(data, selected_indices, noise=True)

    filename = "dataset_sub" + s
    dataset_names = ["data_noise1", "data_noise2"]

    # saving the two generated noisy data simulations in one h5 file, dataset_name is used to differentiate the files
    save_to_h5(filename, all_channel_data_noise1, "data_noise1")
    save_to_h5(filename, all_channel_data_noise2, "data_noise2")

    # open the data sets to see if saving and retrieveing the data works as anticipated
    noisy_data1 = open_h5(filename, dataset_names[0])
    noisy_data2= open_h5(filename, dataset_names[1])

    # Calculate MGFP as the root mean square across electrodes for each time point
    y_coordinates = np.sqrt(np.mean(np.square(all_channel_data), axis=0))  
    y_coordinates_noise1 = np.sqrt(np.mean(np.square(all_channel_data_noise1), axis=0))    
    y_coordinates_noise2 = np.sqrt(np.mean(np.square(all_channel_data_noise2), axis=0))    
    # Note: if all the reading and writing of the data to HDF5 worked correctly, y_coordinates_noise should be identical to rms_noisy_data
    #       same goes for all_channel_data_noise and noisy_data
    rms_noisy_data1 = np.sqrt(np.mean(np.square(noisy_data1), axis=0))   
    rms_noisy_data2 = np.sqrt(np.mean(np.square(noisy_data2), axis=0))   

    # range of the data plotted (used to be in get_data_points function)
    ms_range = (-5, 38)
    start, end = ms_range
    ms_start = (50 + start)*2
    ms_end = (50 + end)*2  
    
    fig, axs = plt.subplots(3,2)
    # plot channel-average of trial-average data based on original data to have a "benchmark"
    axs[0,0].plot(x_coordinates[ms_start:ms_end], y_coordinates[ms_start:ms_end])
    axs[0,0].set_title("RSM trial-average subject " + s)
    # plot channel-average of noisy data generated for a simulated first trial 
    axs[0,1].plot(x_coordinates[ms_start:ms_end], all_channel_data[ms_start:ms_end])
    axs[0,1].set_title("All channel trial-average subject " + s)
    # plot channel-average of noisy data generated for a simulated second trial 
    axs[1,0].plot(x_coordinates[ms_start:ms_end], rms_noisy_data1[ms_start:ms_end])
    axs[1,0].set_title("Trial 1: RSM Noisy channels subject " + s)
    # plot all channels of trial-average data based on original data to have a "benchmark"
    axs[1,1].plot(x_coordinates[ms_start:ms_end], noisy_data1[ms_start:ms_end])
    axs[1,1].set_title("Trial 1: All noisy channels subject " + s)
    # plot all channels of noisy data generated for a simulated first trial 
    axs[2,0].plot(x_coordinates[ms_start:ms_end], rms_noisy_data2[ms_start:ms_end])
    axs[2,0].set_title("Trial 2: RSM noisy channels subject " + s)
    # plot all channels of noisy data generated for a simulated second trial 
    axs[2,1].plot(x_coordinates[ms_start:ms_end], noisy_data2[ms_start:ms_end])
    axs[2,1].set_title("Trial 2: All noisy channels subject " + s)
    fig.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()