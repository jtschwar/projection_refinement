from pathlib import Path
import scipy.io as sio
import tomoTV_align
import numpy as np
import h5py, os

######################################################################

# Input Sinogram and Tilt Angles
fileName = Path('misaligned_xray_dataset.h5')
outputFname = fileName.with_name('aligned_xray_dataset.h5')

# manually define binning values. 
# we want to pick a minumum binning factors that
# will downsample the sinogram to roughly 64 pixels 
binning = [4,2,1] 

data = h5py.File(fileName,'r')
sinogram = data['tiltSeries'][:]
theta = data['tiltAngles'][:]
data.close()

# The shape of the data file needs to be (X, Y, num)
(Nx, Ny, Nangles) = sinogram.shape
assert Nangles == theta.shape[0]

# (Optional) Crop to an object or smaller region 
object_ROI = np.array([0,Nx,0,Ny])

width_sinogram = Ny
Nlayers = Nx
Npix = width_sinogram

######################################################################
# Useful Tuning Parameters

params = {'min_step_size':0.01, 'max_iter':500, 'use_TV':False, 'high_pass_filter':0.01, 'step_relaxation':0.5}
params.update({'tilt_angle':0, 'momentum_acceleration':False, 'apply_positivity':True, 'refine_geometry':True})
params.update({'lamino_angle':90, 'position_update_smoothing':False, 'ROI':object_ROI.astype(int)})
params.update({'showsorted':True,'plot_results':False, 'plot_results_every':15, 'use_gpu':True})
params.update({'filename': str(outputFname)})

# Chose Reconstruction Algorithm:
# Option 1: FBP / WBP great for fast alignment and X-Ray Datasets
params.update({'alg': 'FBP', 'initAlg':'ram-lak'})

#Option 2: SART ~10x slower but accurately aligns electron tomography datasets with missing wedge 
# params.update({'alg': 'SART', 'initAlg':'sequential'})

print('\nAlignment Parameters:\n',params,'\n')
print('Binning factors : {}\n'.format(binning) )

######################################################################
# Consistency-Based Alignment

# Initialize array to hold the shifts
shift = np.zeros([sinogram.shape[2],2])

# Initialize Alignment Class
tomoAlign = tomoTV_align.projection_refine.tomo_align(sinogram, theta)

if params['plot_results']: 
    params = tomoAlign.initialize_plot(params)

for currBin in binning:

    # Self Consisency based alignment procedure based on ASTRA toolbox
    params['binning'] = currBin
    (shift, params) = tomoAlign.tomo_consistency_linear(shift, params)
    tomoAlign.sinogram = sinogram
    tomoAlign.angles = theta

print('Finished Alignments, Saving Data..')

# Save the Aligned Projections, Shifts, and Reconstruction
if os.path.isfile(params['filename']):
    save_h5flag = 'w'
    print(f"{params['filename']} already exists, overwriting current H5 File")
else:
    save_h5flag = 'a'

with h5py.File(params['filename'], save_h5flag) as f2:
    paramGroup = f2.create_group('params')
    for key,item in params.items():
        paramGroup.attrs[key] = item
    f2.create_dataset('aligned_proj', data=tomoAlign.sinogram_shifted)
    f2.create_dataset('aligned_shifts', data=shift)

print(f'Data saved to {params["filename"]}')
