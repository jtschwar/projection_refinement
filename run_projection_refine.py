from pathlib import Path

import scipy.signal as signal
import tomoTV_align
import scipy.io as sio
import numpy as np
import h5py

######################################################################
# Input sinogram data, tilt angles, and cropping

fileName = Path('/mnt/nvme1/microED/uio66_tomo/uio66_tomo2_tomviz.emd')
tilts_file = Path('/mnt/nvme1/microED/uio66_tomo/uio66_tomo2_angles.txt')
outputFname = fileName.with_name('iuo66_tomo_recon.h5')

with h5py.File(fileName,'r') as f0:
    sinogram = f0['data/converted/data'][:]

with open(tilts_file, 'r') as f1:
    theta = np.array([float(ii) for ii in f1.readlines()])

(Nx, Ny, Nangles) = sinogram.shape
Npix = Ny
Nlayers = Nx
Nangles = theta.shape[0]

# Choose reconstructed (cropped) region 
object_ROI = np.array([490,620,0,Ny])

######################################################################
# Useful uning parameters

params = {'min_step_size':0.01, 'max_iter':500, 'use_TV':False, 'high_pass_filter':0.001, 'step_relaxation':0.1}
params.update({'tilt_angle':0, 'momentum_acceleration':False, 'apply_positivity':True, 'refine_geometry':True})
params.update({'filter_type':'ram-lak', 'lamino_angle':90, 'position_update_smoothing':False, 'ROI':object_ROI.astype(int)})
params.update({'showsorted':True,'plot_results':False, 'plot_results_every':5, 'use_gpu':True})
params.update({'filename':outputFname,'alg':'SART','initAlg':'sequential'})

binning = np.array([16, 8, 4, 2, 1], dtype=int)

print('\nAlignment Parameters:\n',params,'\n')
print('Binning factors : {}\n'.format(binning) )

######################################################################
# Consistency-Based Alignment

shift = np.zeros([sinogram.shape[2], 2])

tomoAlign = tomoTV_align.projection_refine.tomo_align(sinogram, theta)
if params['plot_results']: tomoAlign.initialize_plot()

for jj in range(len(binning)):
    params['binning'] = binning[jj]

    # Self Consisency based alignment procedure based on ASTRA toolbox
    (shift, params) = tomoAlign.tomo_consistency_linear(shift, params)
    tomoAlign.sinogram = sinogram
    tomoAlign.angles = theta

print('Finished Alignments, Saving Data..')

# Save the Aligned Projections, Shifts, and Reconstruciton
with h5py.File(params['filename'], 'a') as f2:
    paramGroup = f2.create_group('params')
    for key,item in params.items():
        paramGroup.attrs[key] = item
    f2.create_dataset('aligned_proj', data=tomoAlign.sinogram_shifted)
    f2.create_dataset('aligned_shifts', data=shift)
print(f'Data saved to {params["filename"]}')
