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


def open_h5(file_name,dataset_name):
    file_path = './data/Somatosensory/HDF5/'
    filename = file_path + file_name
    dataset_name = dataset_name
    with h5py.File(filename, 'r') as file:
        data_set = file[dataset_name]
        #make sure we only use the 369 time points of the data
        data=data_set[:369]
    return data


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

def plot_butterfly(subject, data, save_path=None):
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
    if save_path:
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

def plot_v_components(subject, V, s, V_uc=None, s_uc=None, color_plot=False, save_path= None):
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
    s1, s2, s3 = s[:3]
    
    x_v = V[0, :] * s1
    y_v = V[1, :] * s2
    z_v = V[2, :] * s3

    fig, axs = plt.subplots(2, 1)
    
    # Define a colormap that varies according to the index of data points
    cmap = plt.get_cmap('Set1')

    axs[0].plot(x_v)
    axs[0].plot(y_v)
    axs[0].plot(z_v)
    axs[0].set_title(f"First 3 components of V matrix scaled by respective singular values (subject {subject})")
    axs[0].set_xlabel("Time")
    axs[0].set_ylabel("Amplitude")
    axs[0].legend(["first component", "second component", "third component"])
    if color_plot:
        # Plot colored data points
        for i in range(len(x_v)):
            axs[1].plot(x_v[i:i+2], y_v[i:i+2], color=cmap(i/len(x_v)), alpha=0.7)
    else:
        axs[1].plot(x_v, y_v)
    axs[1].set_title(f"Phase space of first 2 components scaled by respective singular values (subject {subject})")
    axs[1].set_xlabel("First component of V")
    axs[1].set_ylabel("Second component of V")

    if not(V_uc is None and s_uc is None):
        s_uc1, s_uc2 = s_uc[:2]
        x_v_uc = V_uc[0, :] * s_uc1
        y_v_uc = V_uc[1, :] * s_uc2

        if color_plot:
            # Plot colored data points
            for i in range(len(x_v_uc)):
                axs[1].plot(x_v_uc[i:i+2], y_v_uc[i:i+2], color=cmap(i/len(x_v_uc)), alpha=0.7)
        else:
            axs[1].plot(x_v_uc, y_v_uc)
            axs[1].legend(['Centered', 'Uncentered'])
    
    
    # Colorbar for reference
    sm_v = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=len(x_v)))
    sm_v.set_array([])
    plt.colorbar(sm_v, ax=axs.ravel().tolist())
    #fig.tight_layout()
    plt.show()

    if save_path:
        fig.savefig(save_path + "v_components.png")


def plot_v_components_with_std(subject, mean_x, mean_y, std_x, std_y):
    # same result, different look
    #plt.errorbar(x=mean_x, y=mean_y, xerr=std_x,yerr=std_y)
    #plt.show()

    # covering the stds in all possible directions
    plt.plot(mean_x, mean_y, "b-", linewidth=2)
    plt.plot(mean_x-std_x, mean_y-std_y, "c--", linewidth=.5)
    plt.plot(mean_x+std_x, mean_y-std_y, "c--", linewidth=.5)
    plt.plot(mean_x-std_x, mean_y+std_y, "c--", linewidth=.5)
    plt.plot(mean_x+std_x, mean_y+std_y, "c--", linewidth=.5)
    plt.plot(mean_x-std_x, mean_y, "c--", linewidth=.5)
    plt.plot(mean_x+std_x, mean_y, "c--", linewidth=.5)
    plt.plot(mean_x, mean_y-std_y, "c--", linewidth=.5)
    plt.plot(mean_x, mean_y+std_y, "c--", linewidth=.5)
    plt.show()


def plot_3D_v_components(subject, V, s):
     ## V components
    ax = plt.figure().add_subplot(projection='3d')

    x_v = V[0,:]*s[0]
    y_v = V[1,:]*s[1]
    z_v = V[2,:]*s[2]

    # Define a colormap that varies according to the index of data points
    cmap = plt.get_cmap('Set1')

    # Plot data points
    for i in range(len(x_v)):
        ax.plot(x_v[i:i+2], y_v[i:i+2], z_v[i:i+2], color=cmap(i/len(x_v)), alpha=0.7)

    # Colorbar for reference
    sm_v = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=len(x_v)))
    sm_v.set_array([])
    plt.colorbar(sm_v, ax=ax)
    ax.set_title(f"Phase space of first 3 components of V-matrix (subject {subject})")
    ax.set_xlabel("First component")
    ax.set_ylabel("Second component")
    ax.set_zlabel("Third component")
    plt.show()


def plot_3D_v_components_with_std(subject, mean_x, mean_y, mean_z, std_x, std_y, std_z):
    ax = plt.figure().add_subplot(projection='3d')
    # same idea, different look
    #ax.errorbar(x=mean_x, y=mean_y, z=mean_z, xerr=std_x, yerr=std_y , zerr=std_z)
    
    ax.plot(mean_x, mean_y, mean_z ,  "b-", linewidth=2)
    ax.plot(mean_x+std_x, mean_y, mean_z,  "c--", linewidth=1)
    ax.plot(mean_x-std_x, mean_y, mean_z,  "c--", linewidth=1)
    ax.plot(mean_x, mean_y+std_y, mean_z,  "c--", linewidth=1)
    ax.plot(mean_x, mean_y-std_y, mean_z,  "c--", linewidth=1)
    ax.plot(mean_x, mean_y, mean_z+std_z,  "c--", linewidth=1)
    ax.plot(mean_x, mean_y, mean_z-std_z,  "c--", linewidth=1)
    
    plt.show()

def main():
    subject = "06"  # Replace with the appropriate subject identifier
    
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
    plot_v_components(subject, V, s, color_plot=True)
    plot_3D_v_components(subject, V, s)

    # creates mean and std. in x, y and z direction for first 3 v components for noisy data to create
    # 3D plot of V components in phase space with mean and std. of all (or at least multiple) trials 
    subject = "6 (noise)"
    filename = "dataset_sub06_data_noise_val.h5"
    ds_name = "data_noise_val"
    all_noisy_x = []
    all_noisy_y = []
    all_noisy_z = []
    for i in range(50):
        # includes 50 noisy trials
        uc_noisy_signals = open_h5(filename, ds_name + str(i)) 
        noisy_signals = center(uc_noisy_signals)
              
        # noisy signal transformed to the space spanned by the V-matrix scaled by singular values of the average signal
        noisy_signal_transformed = np.matmul(U.T, noisy_signals)
        noisy_x, noisy_y, noisy_z = noisy_signal_transformed[:3,:]
        all_noisy_x.append(noisy_x)
        all_noisy_y.append(noisy_y)
        all_noisy_z.append(noisy_z)

    mean_noisy_x = np.mean(all_noisy_x, axis=0)
    std_noisy_x = np.std(all_noisy_x, axis=0)
    mean_noisy_y = np.mean(all_noisy_y, axis=0)
    std_noisy_y = np.std(all_noisy_y, axis=0)
    mean_noisy_z = np.mean(all_noisy_z, axis=0)
    std_noisy_z = np.std(all_noisy_z, axis=0)
    
    plot_3D_v_components_with_std(subject, mean_noisy_x, mean_noisy_y, mean_noisy_z, std_noisy_x, std_noisy_y, std_noisy_z)
    
    plot_v_components_with_std(subject, mean_noisy_x, mean_noisy_y, std_noisy_x, std_noisy_y)

    plot_v_components(subject, V=noisy_signal_transformed, s=np.ones(3), V_uc=noisy_signal_transformed, s_uc=np.ones(3), color_plot=True)
    plot_3D_v_components(subject, noisy_signal_transformed, np.ones(3))
    
    logger.info("All operations completed successfully.")

if __name__ == "__main__":
    main()