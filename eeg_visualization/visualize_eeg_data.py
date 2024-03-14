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

def plot_average_signal(data, selected_indices):
    num_time_points = len(data[0]) - 1  # Assuming all rows have the same number of columns
    time_points = list(range(num_time_points))
    average_signal = [0] * num_time_points

    valid_indices = [index for index in selected_indices if index < len(data)]
    for index in valid_indices:
        time_series = np.array([float(value) for value in data[index][1:]])
        average_signal = [avg + ts for avg, ts in zip(average_signal, time_series)]
    
    if valid_indices:
        average_signal = [avg / len(valid_indices) for avg in average_signal]
        plt.figure(figsize=(10, 7))
        plt.plot(time_points*1000/2048-50, average_signal, label='Average Signal')
        plt.xlabel('Time Points')
        plt.ylabel('Signal')
        plt.title('Average EEG Electrode Signal')
        plt.legend()
        plt.show()
    else:
        print("No valid indices provided for averaging.")

def plot_mgfp(data, selected_indices):
    num_time_points = len(data[0]) - 1  # Assuming all rows have the same number of columns
    time_points = np.array(range(num_time_points))
    signals = []

    valid_indices = [index for index in selected_indices if index < len(data)]
    for index in valid_indices:
        time_series = np.array([float(value) for value in data[index][1:]])
        signals.append(time_series)
    
    if signals:
        signals = np.array(signals)
        # Calculate MGFP as the root mean square across electrodes for each time point
        mgfp = np.sqrt(np.mean(np.square(signals), axis=0))
        
        plt.figure(figsize=(10, 7))
        plt.plot(time_points*1000/2048-50, mgfp, label='MGFP')
        plt.xlabel('Time Points')
        plt.ylabel('MGFP')
        plt.title('Mean Global Field Power (MGFP)')
        plt.legend()
        plt.show()
    else:
        print("No valid indices provided for MGFP calculation.")

# Main function to run the script
def main():
    #print(f"Current Working Directory: {os.getcwd()}")
    subject = '01'
    file_path = './data/Somatosensory/PlosOne/' + subject + '_SEP_prepro_-50_130ms.csv'
    data = read_csv(file_path)

    selected_indices = list(range(256))
    
    plot_mgfp(data, selected_indices)

if __name__ == "__main__":
    main()