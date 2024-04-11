# test file to better understand the data format h5py and the structure of the EEG data.


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
        data_series = data_series.transpose(0,1)
        # Stride over time-series
        for i in range(0,  data_series.size(1) - block_size + 1, stride):  # Truncate in block of block_size
            examples.append(data_series[:,i : i + block_size].unsqueeze(0))

        samples = samples + 1
        if(ndata > 0 and samples > ndata): #If we have enough time-series samples break loop
            break

print(examples[0])
data = torch.cat(examples, dim=0)
print(data[0])

# Calculate normalization constants -> takes forever as our data is way too large (356.000x16x256)

#mu = torch.tensor([torch.mean(data[:,:,0]), torch.mean(data[:,:,1]), torch.mean(data[:,:,2])])
mu = torch.mean(data, dim=2)
mu = torch.tensor(mu)
#mu.size() is [354000, 16]
print(mu)

#std = torch.tensor([torch.std(data[:,:,0]), torch.std(data[:,:,1]), torch.std(data[:,:,2])])
std = torch.mean(data, dim=2)
std = torch.tensor(std)
print(std)