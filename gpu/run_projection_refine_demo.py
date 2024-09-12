from pathlib import Path

import scipy.signal as signali
import projection_refine
import scipy.io as sio
import numpy as np
import h5py

######################################################################
# Input Sinogram and Tilt Angles


fileName = Path('/mnt/nvme1/microED/uio66_tomo/uio66_tomo2_tomviz.emd')
tilts_file = Path('/mnt/nvme1/microED/uio66_tomo/uio66_tomo2_angles.txt')
outputFname = fileName.with_name('iuo66_tomo_recon.h5')

with h5py.File(fileName, 'r') as f0:
    sinogram = f0['data/converted/data'][:]

with open(tilts_file, 'r') as f1:
    theta = np.array([float(ii) for ii in f1.readlines()])

# The shape of the data file needs to be (X, Y, num)
(Nx, Ny, Nangles) = sinogram.shape
assert Nangles == theta.shape[0]

# Crop to an object or smaller region 
object_ROI = np.array([490,620,0,Ny])
width_sinogram = Ny
Nlayers = Nx
Npix = width_sinogram

# Initialize array to hold the shifts
shift = np.zeros([sinogram.shape[2],2])

######################################################################
# Useful Tuning Parameters

params = {'min_step_size':0.01, 'max_iter':500, 'use_TV':False, 'high_pass_filter':0.01, 'step_relaxation':0.5}
params.update({'tilt_angle':0, 'momentum_acceleration':False, 'apply_positivity':True, 'refine_geometry':True})
params.update({'lamino_angle':90, 'position_update_smoothing':False, 'ROI':object_ROI.astype(int)})
params.update({'showsorted':True,'plot_results':False, 'plot_results_every':15, 'use_gpu':True})
params.update({'filename':outputFname})

# Chose Reconstruction Algorithm:
# Option 1: FBP / WBP great for fast alignment and X-Ray Datasets
params.update({'alg': 'FBP', 'initAlg':'ram-lak'})

#Option 2: SART ~10x slower but accurately aligns electron tomography datasets with missing wedge 
# params.update({'alg': 'SART', 'initAlg':'sequential'})

#max_binning = 2**(np.ceil(np.log2(np.max(Npix))-np.log2(100)))
#binning = 2**(np.arange(max_binning-1)[::-1])
max_binning = 16
binning = [16,8,4,2,1] # manually define binning values. Start with a large value and go down

print('\nAlignment Parameters:\n',params,'\n')
print('Binning factors : {}\n'.format(binning) )

######################################################################
# Consistency-Based Alignment

tomoAlign = projection_refine.tomo_align(sinogram, theta)
if params['plot_results']: tomoAlign.initialize_plot()

for bb in binning:
    print(bb)
    params['binning'] = bb

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
