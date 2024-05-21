import csv
import os
import numpy as np
import matplotlib.pyplot as plt
import h5py
import logging
import mne

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
filename_val = "dataset_sub" + s + "_data_noise_val.h5"

len_ds = 180 # manually set to length of validation data
selected_channels = [0, 100, 255] # manually set channel to plot

num_time_points = 369 - 1  # Assuming all rows have the same number of columns
time_points = np.array(range(num_time_points))
time_points = time_points*1000/2048-50
 
# channel data contains exactly one channel of every trial
# it has 180 entries (1 per trial) with each 369 elements (time points) 

#######
########## Understanding and analyzing the ORIGINAL DATA ########
#######

#######
## Trial averages of all channels of ORIGINAL DATA (avg.) in butterfly plot
if True:
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
if True:
    U, s, V = np.linalg.svd(org_avg_data, full_matrices=False)

    # Plot the singular values
    plt.plot(s, 'o-')
    plt.title('Singular Values')
    plt.xlabel('Singular Value Index')
    plt.ylabel('Singular Value Magnitude')
    plt.grid(True)
    plt.show()


    ## Variance explanied
    # https://www.geeksforgeeks.org/image-reconstruction-using-singular-value-decomposition-svd-in-python/

    # import module 
    import seaborn as sns 
    
    var_explained = np.round(s**2/np.sum(s**2), decimals=6) 
    
    # Variance explained top Singular vectors 
    print(f'variance Explained by Top 20 singular values:\n{var_explained[0:20]}') 
    
    sns.barplot(x=list(range(1, 21)), 
                y=var_explained[0:20], color="dodgerblue") 
    
    plt.title('Variance Explained Graph') 
    plt.xlabel('Singular Vector') 
    plt.ylabel('Variance Explained') 
    plt.tight_layout() 
    plt.show() 


## Reconstruction
if False:
    # plot images with different number of components 
    comps = [255, 1, 2, 3, 4, 100] 
    
    num_rows = 2
    num_cols = (len(comps) + 1) // num_rows
    fig, axes = plt.subplots(num_rows, num_cols)

    for i, comp in enumerate(comps):
        row = i // num_cols
        col = i % num_cols
        low_rank = U[:, :comp] @ np.diag(s[:comp]) @ V[:comp, :] 
        axes[row, col].plot(low_rank.T)
        axes[row, col].set_title(f'n_components = {comp}')
        axes[row, col].set_xlabel('Sample Index')
        axes[row, col].set_ylabel('Value')

    # Turn off extra subplots
    for i in range(len(comps), num_rows * num_cols):
        row = i // num_cols
        col = i % num_cols
        fig.delaxes(axes[row, col])

    plt.tight_layout()
    plt.show()


## Exploration: plotting the vectors of the matrix V 
if True:
    fig, axs = plt.subplots(2,1)
    s1 = s[0]
    s2 = s[1]
    s3 = s[2]
    x_v = V[0,:]*s1
    y_v = V[1,:]*s2
    z_v = V[2,:]*s3
    axs[0].plot(x_v)
    axs[0].plot(y_v)
    axs[0].plot(z_v)
    axs[0].set_title("First 3 components of V matrix scaled by respective singular values")
    axs[0].set_xlabel("Amplitude")
    axs[0].set_ylabel("Time")
    axs[0].legend(["first component", "second component", "third component"])

    axs[1].plot(x_v, y_v)
    axs[1].set_title("Phase space of first 2 components scaled by respective singular values")
    axs[1].set_xlabel("First component of V")
    axs[1].set_ylabel("Second component of V")

    plt.tight_layout()
    plt.show()


## Exploration: 3D plots of the first three elements (phase space) of the matrices U and V
if False:  
    ## V components
    ax = plt.figure().add_subplot(projection='3d')

    x_v = V[0,:]
    y_v = V[1,:]
    z_v = V[2,:]

    # Define a colormap that varies according to the index of data points
    cmap = plt.get_cmap('plasma')

    # Plot data points
    for i in range(len(x_v)):
        ax.plot(x_v[i:i+2], y_v[i:i+2], z_v[i:i+2], color=cmap(i/len(x_v)), alpha=0.7)

    # Colorbar for reference
    sm_v = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=len(x_v)))
    sm_v.set_array([])
    plt.colorbar(sm_v, ax=ax)
    plt.title("Phase space of first 3 components of V-matrix")
    plt.show()

    ## U components
    ax = plt.figure().add_subplot(projection='3d')
    
    x_u = U[:, 0]
    y_u = U[:, 1]
    z_u = U[:, 2]

    # Define a colormap that varies according to the index of data points
    cmap = plt.get_cmap('viridis')

    # Plot data points
    for i in range(len(x_u)):
        ax.plot(x_u[i:i+2], y_u[i:i+2], z_u[i:i+2], color=cmap(i/len(x_u)), alpha=0.7)

    # Colorbar for reference
    sm_u = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=len(x_u)))
    sm_u.set_array([])
    plt.colorbar(sm_u, ax=ax)
    plt.title("Phase space of first 3 components of U-matrix")
    plt.show()

## Exploration: Topomap of first compnents of U-matrix
if True:
    eeg_file_path = "./data/Somatosensory/Channel_locations/standard_waveguard256_duke.elc"
    mnt = mne.channels.read_custom_montage(eeg_file_path, head_size = 0.131)
    channels = mnt.ch_names
    info = mne.create_info(channels, sfreq=2048, ch_types='eeg')
    info.set_montage(mnt)

    x_u = U[:, 0]
    y_u = U[:, 1]
    z_u = U[:, 2]

    fig, axs = plt.subplots(1,3)
    im,_ = mne.viz.plot_topomap(x_u.tolist(), info, axes=axs[0], contours=0, size=4, show=False)
    axs[0].set_title(f"Topomap of first component of U-matrix")
    im,_ = mne.viz.plot_topomap(y_u.tolist(), info, axes=axs[1], contours=0, size=4, show=False)
    axs[1].set_title(f"Topomap of second component of U-matrix")
    im,_ = mne.viz.plot_topomap(z_u.tolist(), info, axes=axs[2], contours=0, size=4, show=False)
    axs[2].set_title(f"Topomap of third component of U-matrix")

    cbar = plt.colorbar(im, ax=axs.ravel().tolist())
    plt.show()

#######
########## Using same tools to analyze ARTIFICIAL DATA ########
#######

#######
## All trials and trial averages of individual channels of the ARTIFICIAL DATA in comparable plot view 
if False:
    fig, axs = plt.subplots(3,2, sharey='row')
    for i, chan in enumerate(selected_channels):
        channel_data = np.empty((len_ds, 369)) # 369 time points
        for j in range(180):
            ds_name = "data_noise_val" + str(j)
            trial_data = open_h5(filename_val, ds_name)
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
## Trial averages of all channels of ARTIFICIAL DATA in butterfly plot
# if this is actually useful, create matrix instead of list of lists for better efficiency
if False:
    trial_averaged_channels = []
    for i, chan in enumerate(range(255)): # all channels
        channel_data = np.empty((len_ds, 369)) # 369 time points
        for j in range(180):
            ds_name = "data_noise_val" + str(j)
            trial_data = open_h5(filename_val, ds_name)
            channel_data[j] = trial_data[chan]
        trial_averaged_channels.append(np.mean(channel_data, axis = 0))
    # transpose list of lists
    trial_averaged_channels = list(map(list, zip(*trial_averaged_channels)))
    fig, ax = plt.subplots(1,1)

    # plot the single channel for all the trials
    plt.plot(trial_averaged_channels)
    plt.title(f"Artificial averaged data for all channels of subject {s}")

    fig.tight_layout()
    plt.show()

######
## Singular Value Decomposition
if False:
    U, s, V = np.linalg.svd(trial_averaged_channels, full_matrices=False)

    # Plot the singular values
    plt.plot(s, 'o-')
    plt.title('Singular Values')
    plt.xlabel('Singular Value Index')
    plt.ylabel('Singular Value Magnitude')
    plt.grid(True)
    plt.show()


    ## Variance explanied
    # https://www.geeksforgeeks.org/image-reconstruction-using-singular-value-decomposition-svd-in-python/

    # import module 
    import seaborn as sns 
    
    var_explained = np.round(s**2/np.sum(s**2), decimals=6) 
    
    # Variance explained top Singular vectors 
    print(f'variance Explained by Top 20 singular values:\n{var_explained[0:20]}') 
    
    sns.barplot(x=list(range(1, 21)), 
                y=var_explained[0:20], color="dodgerblue") 
    
    plt.title('Variance Explained Graph') 
    plt.xlabel('Singular Vector', fontsize=16) 
    plt.ylabel('Variance Explained', fontsize=16) 
    plt.tight_layout() 
    plt.show() 


## Reconstruction
if False:
    # plot images with different number of components 
    comps = [255, 1, 2, 3, 4, 100] 

    num_rows = 2
    num_cols = (len(comps) + 1) // num_rows
    fig, axes = plt.subplots(num_rows, num_cols)

    for i, comp in enumerate(comps):
        row = i // num_cols
        col = i % num_cols
        low_rank = U[:, :comp] @ np.diag(s[:comp]) @ V[:comp, :] 
        axes[row, col].plot(low_rank)
        axes[row, col].set_title(f'n_components = {comp}')
        axes[row, col].set_xlabel('Sample Index')
        axes[row, col].set_ylabel('Value')

    # Turn off extra subplots
    for i in range(len(comps), num_rows * num_cols):
        row = i // num_cols
        col = i % num_cols
        fig.delaxes(axes[row, col])

    plt.tight_layout()
    plt.show()