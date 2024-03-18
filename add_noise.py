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

def get_data_points(data, selected_indices, ms_range=(-5, 38)):
    num_time_points = len(data[0]) - 1  # Assuming all rows have the same number of columns
    time_points = np.array(range(num_time_points))
    time_points = time_points*1000/2048-50
    start, end = ms_range
    ms_start = (50 + start)*2
    ms_end = (50 + end)*2

    signals = []


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

def add_gaus_noise(y_coord, mean=0, standard_dev = 3):
    noise = np.random.normal(mean, standard_dev, len(y_coord))
    y_noise = y_coord + noise
    return y_noise

def main():
    #print(f"Current Working Directory: {os.getcwd()}")
    s = '05' # subject
    file_path = './data/Somatosensory/PlosOne/' + s + '_SEP_prepro_-50_130ms.csv'
    
    data = read_csv(file_path)

    selected_indices = list(range(256))

    x_coordinates, y_coordinates = get_data_points(data, selected_indices)
    
    y_noisy_005 = add_gaus_noise(y_coordinates, 0, 0.05)

    y_noisy_01 = add_gaus_noise(y_coordinates, 0, 0.1)

    fig, axs = plt.subplots(3,1)
    axs[0].plot(x_coordinates, y_coordinates)
    axs[0].set_title("Subject " + s)
    axs[1].plot(x_coordinates, y_noisy_005)
    axs[1].set_title("Noisy (0.05) Subject " + s)
    axs[2].plot(x_coordinates, y_noisy_01)
    axs[2].set_title("Noisy (0.1) Subject " + s)
    fig.tight_layout()

    plt.show()

if __name__ == "__main__":
    main()