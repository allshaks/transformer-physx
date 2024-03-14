import csv
import os
import numpy as np
import matplotlib.pyplot as plt

# Function to read the CSV file and return the data
def read_csv(file_path):
    with open(file_path, 'r') as f:
        reader = csv.reader(f)
        data = list(reader)
    return data

# Function to plot the selected electrodes
def plot_selected_electrodes(data, selected_indices):
    time_points = np.array(range(len(data[0]) - 1))  # Assuming all rows have the same number of columns
    plt.figure(figsize=(10, 7))

    for index in selected_indices:
        if index < len(data):
            electrode_name = data[index][0]
            time_series = np.array([float(value) for value in data[index][1:]])
            plt.plot(time_points*1000/2048-50, time_series, label=electrode_name)
        else:
            print(f"Index {index} is out of bounds. Skipping.")

    plt.xlabel('Time Points')
    plt.ylabel('Signal')
    plt.title('EEG Electrode Signals')
    plt.legend()
    plt.show()

def plot_average_signal(data, selected_indices, ms_range=(-50, 130)):
    num_time_points = len(data[0]) - 1  # Assuming all rows have the same number of columns
    time_points = np.array((range(num_time_points)))
    time_points = time_points*1000/2048-50
    start, end = ms_range
    ms_start = (50 + start)*2
    ms_end = (50 + end)*2

    average_signal = [0] * num_time_points

    valid_indices = [index for index in selected_indices if index < len(data)]
    for index in valid_indices:
        time_series = np.array([float(value) for value in data[index][1:]])
        average_signal = [avg + ts for avg, ts in zip(average_signal, time_series)]
    
    if valid_indices:
        average_signal = [avg / len(valid_indices) for avg in average_signal]
        plt.figure(figsize=(10, 7))
        plt.plot(time_points[ms_start:ms_end], average_signal[ms_start:ms_end], label='Average Signal')
        plt.xlabel('Time Points')
        plt.ylabel('Signal')
        plt.title('Average EEG Electrode Signal')
        plt.legend()
        plt.show()
    else:
        print("No valid indices provided for averaging.")

def plot_mgfp(data, selected_indices, ms_range=(-5, 38)):
    num_time_points = len(data[0]) - 1  # Assuming all rows have the same number of columns
    time_points = np.array(range(num_time_points))
    time_points = time_points*1000/2048-50
    start, end = ms_range
    ms_start = (50 + start)*2
    ms_end = (50 + end)*2

    signals = []

    valid_indices = [index for index in selected_indices if index < len(data)]
    for index in valid_indices:
        time_series = np.array([float(value) for value in data[index][1:]])
        signals.append(time_series)
    
    if signals:
        signals = np.array(signals)
        # Calculate MGFP as the root mean square across electrodes for each time point
        mgfp = np.sqrt(np.mean(np.square(signals), axis=0))      

        return (time_points[ms_start:ms_end], mgfp[ms_start:ms_end])
        """
        plt.figure(figsize=(10, 7))
        plt.plot(time_points[ms_start:ms_end], mgfp[ms_start:ms_end], label='MGFP')
        plt.xlabel('Time Points')
        plt.ylabel('MGFP')
        plt.title('Mean Global Field Power (MGFP)')
        plt.legend()
        plt.show()
        """
    else:
        print("No valid indices provided for MGFP calculation.")
    
# Main function to run the script
def main():
    #print(f"Current Working Directory: {os.getcwd()}")
    subjects = ['01','02', '03', '04', '05', '06', '07', '08', '09', '10']
    file_path = np.empty(len(subjects), dtype = object)
    data = np.empty(len(subjects), dtype = object)

    for index, s in enumerate(subjects):
        file_path[index] = './data/Somatosensory/PlosOne/' + s + '_SEP_prepro_-50_130ms.csv'
    
    for index, path in enumerate(file_path):
        data[index] = read_csv(path)


    selected_indices = list(range(256))
    
    x_coordinates = np.empty(len(subjects), dtype = object)
    y_coordinates = np.empty(len(subjects), dtype = object)
    for i, d in enumerate(data):
        x_coordinates[i] , y_coordinates[i] = plot_mgfp(data[i], selected_indices)
    
    fig, axs = plt.subplots(5, 2)
    for i in range(10):
        axs[i%5,int(np.floor(i/5))].plot(x_coordinates[i], y_coordinates[i])
        axs[i%5,int(np.floor(i/5))].set_title("Subject " + str(i))
    fig.tight_layout()

    plt.show()

if __name__ == "__main__":
    main()