import csv
import os
import numpy as np
import matplotlib.pyplot as plt
import h5py
import logging
import seaborn as sns
import mne

# Configure logger
logger = logging.getLogger(__name__)

def open_h5(file_name, dataset_name):
    """
    Open an HDF5 file and read a specified dataset.
    
    Parameters:
        file_name (str): The name of the HDF5 file.
        dataset_name (str): The name of the dataset to read.
    
    Returns:
        np.ndarray: The data from the dataset, limited to the first 369 time points.
    """
    file_path = './data/Somatosensory/HDF5/'
    filename = file_path + file_name
    with h5py.File(filename, 'r') as file:
        data_set = file[dataset_name]
        data = data_set[:369]
    return data

def read_csv(file_path):
    """
    Read a CSV file and return its data.
    
    Parameters:
        file_path (str): The path to the CSV file.
    
    Returns:
        list: The data from the CSV file as a list of lists.
    """
    with open(file_path, 'r') as f:
        reader = csv.reader(f)
        data = list(reader)
    return data

def get_data_points(data, selected_channels=256, selected_time_points=369):
    """
    Extract and return selected data points from the provided data.
    
    Parameters:
        data (list): The input data.
        selected_channels (int): The number of channels to select.
        selected_time_points (int): The number of time points to select.
    
    Returns:
        tuple: A tuple containing the time points and the signals.
    """
    selected_channels = list(range(selected_channels)) 
    time_points = np.array(range(selected_time_points))
    time_points = time_points * 1000 / 2048 - 50

    labels = []
    signals = []
    
    valid_indices = [index for index in selected_channels if index < len(data)]
    for index in valid_indices:
        time_series = np.array([float(value) for value in data[index][1:]])
        labels.append(data[index][0])
        signals.append(time_series)

    if signals:
        signals = np.array(signals)      
        return (time_points, signals)
    else:
        print("No valid indices provided.")

def center(data):
    """
    Center the data by subtracting the mean of each channel.
    
    Parameters:
        data (np.ndarray): The input data with dimensions (channels x time points).
    
    Returns:
        np.ndarray: The centered data.
    """
    channel_means = np.mean(data, axis=1, keepdims=True)
    centered_data = data - channel_means
    return centered_data

def normalize(data):
    """
    Normalize the data by centering it and dividing by the standard deviation of each channel.
    
    Parameters:
        data (np.ndarray): The input data with dimensions (channels x time points).
    
    Returns:
        np.ndarray: The normalized data.
    """
    centered_data = center(data)
    channel_std_devs = np.std(centered_data, axis=1, keepdims=True)
    normalized_data = centered_data / channel_std_devs    
    return normalized_data

def create_directory(path):
    """
    Create a directory if it does not already exist.
    
    Parameters:
        path (str): The path of the directory to create.
    """
    if not os.path.exists(path):
        os.makedirs(path)

def plot_butterfly(subject, data, save_path):
    """
    Plot the butterfly plot for the provided data and save it.
    
    Parameters:
        subject (str): The subject identifier.
        data (np.ndarray): The data to plot.
        save_path (str): The path to save the plot.
    """
    fig, ax = plt.subplots(1, 1)
    ax.plot(data.T)
    ax.set_title(f"Original averaged data for all channels for subject {subject}")
    ax.set_xlabel("Time")
    ax.set_ylabel("Amplitude")

    fig.tight_layout()
    plt.show()
    fig.savefig(save_path + "butterfly_plot_all_channels.png", dpi=fig.dpi)

def plot_singular_values(subject, s, save_path):
    """
    Plot the singular values and save the plot.
    
    Parameters:
        subject (str): The subject identifier.
        s (np.ndarray): The singular values.
        save_path (str): The path to save the plot.
    """
    fig, ax = plt.subplots(1, 1)
    ax.plot(s, 'o-')
    ax.set_title(f"Singular values for subject {subject}")
    ax.set_xlabel('Index')
    ax.set_ylabel('Value')

    fig.tight_layout()
    plt.grid(True)
    plt.show()
    fig.savefig(save_path + "singular_values.png")

def plot_variance_explained(subject, s, save_path):
    """
    Plot the variance explained by the singular values and save the plot.
    
    Parameters:
        subject (str): The subject identifier.
        s (np.ndarray): The singular values.
        save_path (str): The path to save the plot.
    """
    var_explained = np.round(s ** 2 / np.sum(s ** 2), decimals=6)
    fig, ax = plt.subplots(1, 1)
    sns.barplot(x=list(range(1, 21)), y=var_explained[0:20], color="dodgerblue", ax=ax)
    ax.set_title(f'Variance explained graph for subject {subject}')
    ax.set_xlabel('Index')
    ax.set_ylabel('Value')

    fig.tight_layout()
    plt.show()
    fig.savefig(save_path + "variance_explained_graph.png")

def plot_reconstruction(subject, U, s, V, save_path):
    """
    Plot the reconstruction of the signal using different numbers of components and save the plot.
    
    Parameters:
        subject (str): The subject identifier.
        U (np.ndarray): The left singular vectors.
        s (np.ndarray): The singular values.
        V (np.ndarray): The right singular vectors.
        save_path (str): The path to save the plot.
    """
    comps = [255, 1, 2, 3, 4, 100]
    num_rows = 2
    num_cols = (len(comps) + 1) // num_rows
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(16, 9))

    for i, comp in enumerate(comps):
        row = i // num_cols
        col = i % num_cols
        low_rank = U[:, :comp] @ np.diag(s[:comp]) @ V[:comp, :]
        fig.suptitle(f"Reconstruction of the signal for subject {subject} with n components")
        axes[row, col].plot(low_rank.T)
        axes[row, col].set_title(f'n_components = {comp}')
        axes[row, col].set_xlabel('Time')
        axes[row, col].set_ylabel('Amplitude')

    for i in range(len(comps), num_rows * num_cols):
        row = i // num_cols
        col = i % num_cols
        fig.delaxes(axes[row, col])

    fig.tight_layout()
    plt.show()
    fig.savefig(save_path + "overview_svd_reconstruction.png")

def plot_v_components(subject, V, s, V_uc, s_uc, save_path):
    """
    Plot the first three components of the V matrix and their phase space, and save the plot.
    
    Parameters:
        subject (str): The subject identifier.
        V (np.ndarray): The centered right singular vectors.
        s (np.ndarray): The centered singular values.
        V_uc (np.ndarray): The uncentered right singular vectors.
        s_uc (np.ndarray): The uncentered singular values.
        save_path (str): The path to save the plot.
    """
    fig, axs = plt.subplots(2, 1)
    s1, s2, s3 = s[:3]
    x_v = V[0, :] * s1
    y_v = V[1, :] * s2
    z_v = V[2, :] * s3

    x_v_uc = V_uc[0, :] * s1
    y_v_uc = V_uc[1, :] * s2

    axs[0].plot(x_v)
    axs[0].plot(y_v)
    axs[0].plot(z_v)
    axs[0].set_title(f"First 3 components of V matrix scaled by respective singular values (subject {subject})")
    axs[0].set_xlabel("Time")
    axs[0].set_ylabel("Amplitude")
    axs[0].legend(["first component", "second component", "third component"])

    axs[1].plot(x_v, y_v)
    axs[1].set_title(f"Phase space of first 2 components scaled by respective singular values (subject {subject})")
    axs[1].set_xlabel("First component of V")
    axs[1].set_ylabel("Second component of V")

    axs[1].plot(x_v_uc, y_v_uc)
    axs[1].legend(['Centered', 'Uncentered'])

    fig.tight_layout()
    plt.show()
    fig.savefig(save_path + "v_components.png")


def main():
    subject = "07"  # Replace with the appropriate subject identifier
    
    file_path = './data/Somatosensory/PlosOne/' + subject + '_SEP_prepro_-50_130ms.csv'
    save_path = "./screenshots/sub" + subject + "/"

    # Create directory for saving results
    create_directory(save_path)
    
    # Read data from CSV file
    logger.info("Reading data from CSV file...")
    org_avg_data = read_csv(file_path)
    
    # Extract time points and signals
    logger.info("Extracting time points and signals...")
    # get uncentered signals
    time_points, uc_signals = get_data_points(org_avg_data)
    if uc_signals is None:
        logger.error("No valid signals found.")
        return
    signals = center(uc_signals)

    # Perform SVD
    logger.info("Performing Singular Value Decomposition...")
    U, s, V = np.linalg.svd(signals, full_matrices=False)
    U_uc, s_uc, V_uc = np.linalg.svd(uc_signals, full_matrices=False)

    # Plot results
    logger.info("Plotting results...")
    plot_butterfly(subject, signals, save_path)
    plot_singular_values(subject, s, save_path)
    plot_variance_explained(subject, s, save_path)
    plot_reconstruction(subject, U, s, V, save_path)
    plot_v_components(subject, V, s, V_uc, s_uc, save_path)

    logger.info("All operations completed successfully.")

if __name__ == "__main__":
    main()