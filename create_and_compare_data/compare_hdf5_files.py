import h5py
import torch
import os

exp_name = "EEG_SEP"
file_path = "./data/Somatosensory/HDF5/dataset_sub06_data_noise_train.h5"
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

data = torch.cat(examples, dim=0)

# Calculate normalization constants
mu = torch.mean(data, dim=2)
mu = torch.tensor(mu)

std = torch.mean(data, dim=2)
std = torch.tensor(std)



########
###### Lorenz files ########
########

exp_name = "EEG_SEP"
file_path_1 = "./data/lorenz/lorenz_test_rk.hdf5"
file_path_2 = "./data/lorenz/lorenz_training_rk.hdf5"
file_path_3 = "./data/lorenz/lorenz_valid_rk.hdf5"
batch_size = 512
block_size = 16
n_train = 2048
ndata = -1
stride = 1

print(os.getcwd())

examples = []
with h5py.File(file_path_3, "r") as f:
    # Iterate through stored time-series
    samples = 0
    for key in f.keys():
        data_series = torch.Tensor(f[key])
        # Stride over time-series
        for i in range(0,  data_series.size(0) - block_size + 1, stride):  # Truncate in block of block_size
            examples.append(data_series[i : i + block_size].unsqueeze(0))

        samples = samples + 1
        if(ndata > 0 and samples > ndata): #If we have enough time-series samples break loop
            break


# Calculate normalization constants
data = torch.cat(examples, dim=0)
# for lorenz_test_rk.hdf5:     data.size() is torch.Size([258560, 16, 3]); 256  samples; 1025 time points
#   check: 256 *(1025-16+1)/1 = 258560
# for lorenz_training_rk.hdf5: data.size() is torch.Size([495616, 16, 3]); 2048 samples; 257  time points
#   check: 2048*(257-16+1)/1 = 495616
# for lorenz_valid_rk.hdf5:    data.size() is torch.Size([64640, 16, 3]);  64   samples; 1025 time points
#   check: 64*(1025-16+1)/1 = 258560

# data split: test:       10.810% (with all time points)
#             training:   86.486% (with 25% of all time points)
#             validation:  2.702% (with all time points)
mu = torch.tensor([torch.mean(data[:,:,0]), torch.mean(data[:,:,1]), torch.mean(data[:,:,2])])
std = torch.tensor([torch.std(data[:,:,0]), torch.std(data[:,:,1]), torch.std(data[:,:,2])])

