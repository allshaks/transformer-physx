import h5py
import torch
import os

exp_name = "EEG_SEP"
file_path = "./data/Somatosensory/HDF5/dataset_sub06.h5"
batch_size = 512
block_size = 16
n_train = 2048
ndata = -1
stride = 1

print(os.getcwd())

examples = []
with h5py.File(file_path, "r") as f:
    # Iterate through stored time-series
    samples = 0
    for key in f.keys():
        data_series = torch.Tensor(f[key])
        # Stride over time-series
        for i in range(0,  data_series.size(1) - block_size + 1, stride):  # Truncate in block of block_size
            examples.append(data_series[:,i : i + block_size].unsqueeze(0))

        samples = samples + 1
        if(ndata > 0 and samples > ndata): #If we have enough time-series samples break loop
            break

print(examples[0])
# Calculate normalization constants

data = torch.cat(examples, dim=0)
print(data[0])
# Calculate normalization constants
data = torch.cat(examples, dim=0)
""" ## Notes for better understanding of data:
# len(data) is trials x (time_points - block_size + 1)/stride = 1000 * (369 - 16 + 1)/1 = 354.000
#  -> means we have 1000 trials and for each trial we create a number of blocks calculated by 
#     - time_points to not exceed any limit and include all data points
#     - block_size and stride
# len(data[0]) is number of EEG-channels (256): each of the 354.000 entries is one data set containing all 256 channels
# len(data[0][0]) is one block (16):            each of the channels contains exactly one block of time points
# data[0][0][0] is one specific element/value, in this case the value for trial one, channel one, time_point 1
# this means that data[0:353] are all elements for trial one with slightly different blocks (shifted by one)
# this implies, that element data[0][0][10] is the same as element data[10][0][0] (for stride 1) and as element data[5][0][5]
"""
# data seems to be different than original data if we look at the mu and std function -> compare!!

#mu = torch.tensor([torch.mean(data[:,:,0]), torch.mean(data[:,:,1]), torch.mean(data[:,:,2])])
mu_list = [torch.mean(data[:,:,i]) for i in range(256)]
mu = torch.tensor(mu_list)

#std = torch.tensor([torch.std(data[:,:,0]), torch.std(data[:,:,1]), torch.std(data[:,:,2])])
std_list = [torch.mean(data[:,:,i]) for i in range(256)]
std = torch.tensor(std_list)