import mne

file_path = "./data/Somatosensory/Channel_locations/standard_waveguard256_duke.elc"
"""
pos = []
lines = ["Z1L :	105.6	22.5	54.1", "Z2L :	101.7	20.7	71.5"]

for line in lines:
    pos.append(list(map(float, line.split()[-3:])))

print(pos)
"""
mnt = mne.channels.read_custom_montage(file_path, head_size = 0.131)
print(mnt.get_positions())

fig1 = mnt.plot()
fig2 = mnt.plot(kind="3d", show=False)  # 3D
fig2 = fig2.gca().view_init(azim=70, elev=15)


### Get values of channels at given time points
# get positions (x,y,z) of a specific channel: 
ch_pos_l13l =mnt.get_positions()["ch_pos"]['L13L']
#fig3 = mnt.viz.plot_topomap(data_time_0, channel_pos)

# next steps:
#   - create list of all channel names for loops etc
#   - extract data for each channel at a given data point
#   - extract positions for each channel using get_position (possible like this? get_position returns 3 coordinates (x,y,z) but we want to plot it in 2D)
#   - make topomap using data and channel positions