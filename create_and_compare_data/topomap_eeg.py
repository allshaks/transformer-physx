import mne
import csv
import os
import time
import numpy as np
from matplotlib import cm
from matplotlib.animation import FuncAnimation, Animation
import matplotlib.pyplot as plt

# Function to read the CSV file and return the data
def read_csv(file_path):
    with open(file_path, 'r') as f:
        reader = csv.reader(f)
        data = list(reader)
    return data

eeg_file_path = "./data/Somatosensory/Channel_locations/standard_waveguard256_duke.elc"

mnt = mne.channels.read_custom_montage(eeg_file_path, head_size = 0.131)

channels = mnt.ch_names
info = mne.create_info(channels, sfreq=2048, ch_types='eeg')
info.set_montage(mnt)


s = '06' # subject
data_file_path = './data/Somatosensory/PlosOne/' + s + '_SEP_prepro_-50_130ms.csv'
data = read_csv(data_file_path)

data_dict = {}
for line in data:
    data_dict[line[0]] = line[1:]

# Plot topomaps for four different time points selected by looking at the peaks of the butterfly plot
if True:
    time_points = [136, 166, 192, 238]
    fig, axs = plt.subplots(1,4)
    for i,tp in enumerate(time_points):
        # extract data of specified time point in the order of the channels of the montage
        data_tp = []
        for ch in channels:
            data_tp.append(float(data_dict[ch][tp]))

        im,_ = mne.viz.plot_topomap(data_tp, info, axes=axs[i], contours=0, size=4, show=False)
        axs[i].set_title(f"Sub006: Time point {tp}")

    cbar = fig.colorbar(im, ax=axs.ravel().tolist(), orientation='horizontal')
    plt.show()

# Make an animated plot of all time points with one consistent colormap
if False:
    # Prepare data for smoother animation
    data_all_tp = []
    for tp in range(0,369):
        data_tp = []
        for ch in channels:
            data_tp.append(float(data_dict[ch][tp]))
        data_all_tp.append(data_tp)
    
    fig, axs = plt.subplots()
    
    cmap = plt.cm.get_cmap('plasma')
    min_value = -4
    max_value = 4
    scalar_mappable = cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=min_value, vmax=max_value))
    start = 100

    for tp in range(start,369):
        # extract data of specified time point in the order of the channels of the montage
        data_plt = data_all_tp[tp]
        im,_ = mne.viz.plot_topomap(data_plt, info, axes=axs, cmap=cmap, contours=0, size=4, show=False)
        
        # Set clim for consistent color mapping
        im.set_clim(vmin=min_value, vmax=max_value)
        
        # Add colorbar to first plot
        if tp == start:
            cbar = plt.colorbar(scalar_mappable, ax=axs)

        plt.draw()
        plt.title(f"Current time point: {tp}")
        plt.pause(0.001)
        # break out of for loop if window is closed
        if not plt.fignum_exists(1):
            break

# Improve animation using matplotlib.animation
if True:
    # Prepare data for smoother animation
    data_all_tp = []
    for tp in range(0, 369):
        data_tp = []
        for ch in channels:
            data_tp.append(float(data_dict[ch][tp]))
        data_all_tp.append(data_tp)

    fig, axs = plt.subplots()
    cmap = plt.cm.get_cmap('plasma')
    min_value = -4
    max_value = 4
    scalar_mappable = cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=min_value, vmax=max_value))

    def update(tp):
        axs.clear()
        data_plt = data_all_tp[tp]
        im, _ = mne.viz.plot_topomap(data_plt, info, axes=axs, cmap=cmap, contours=0, size=4, show=False)
        im.set_clim(vmin=min_value, vmax=max_value)
        if tp == start+1:
            cbar = plt.colorbar(scalar_mappable, ax=axs)
        axs.set_title(f"Current time point: {tp}")

    start = 100
    ani = FuncAnimation(fig, update, frames=range(start, 369), repeat=False)

    plt.show()




