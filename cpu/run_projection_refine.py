import scipy.signal as signal
import projection_refine
import scipy.io as sio
import numpy as np
import h5py

######################################################################
# Input Sinogram and Tilt Angles

sinogram = sio.loadmat('sinogram.mat')
sinogram = sinogram['sinogram']
(Nx, Ny, Nangles) = sinogram.shape

# Choose reconstructed (cropped) region 
object_ROI = np.array([0,-1,0,-1])
width_sinogram = Ny
Nlayers = Nx
Npix = width_sinogram

theta = sio.loadmat('theta.mat')
theta = theta['theta'].flatten()
Nangles = theta.shape[0]

shift = np.zeros([sinogram.shape[2],2])

######################################################################
# Useful Tuning Parameters

params = {'min_step_size':0.1, 'max_iter':500, 'use_TV':False, 'high_pass_filter':0.01, 'step_relaxation':0.5}
params.update({'tilt_angle':0, 'momentum_acceleration':False, 'apply_positivity':True, 'refine_geometry':False})
params.update({'filter_type':'ram-lak', 'lamino_angle':90, 'position_update_smoothing':False, 'ROI':object_ROI.astype(int)})
params.update({'showsorted':True,'plot_results':True, 'plot_results_every':5, 'use_gpu':True})
params.update({'filename':'simulation.h5'})

max_binning = 2**(np.ceil(np.log2(np.max(Npix))-np.log2(100)))
binning = 2**(np.arange(max_binning-1)[::-1])

print('\nAlignment Parameters:\n',params,'\n')
print('Binning factors : {}\n'.format(binning) )

######################################################################
# Consistency-Based Alignment

tomoAlign = projection_refine.tomo_align(sinogram, theta)
if params['plot_results']: tomoAlign.initialize_plot()

for jj in range(len(binning)):
    params['binning'] = binning[jj]

    # Self Consisency based alignment procedure based on ASTRA toolbox
    (shift, params) = tomoAlign.tomo_consistency_linear(shift, params)
    tomoAlign.sinogram = sinogram
    tomoAlign.angles = theta

# Save the Aligned Projections, Shifts, and Reconstruciton
file = h5py.File(params['filename'], 'a')
paramGroup = file.create_group('params')
for key,item in params.items():
    paramGroup.attrs[key] = item
file.create_dataset('aligned_proj', data=tomoAlign.sinogram_shifted)
file.create_dataset('aligned_shifts', data=shift)
file.close()
