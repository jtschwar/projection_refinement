import scipy.signal as signal
import projection_refine
import scipy.io as sio
import numpy as np

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

params = {'min_step_size':0.01, 'max_iter':500, 'use_TV':False, 'high_pass_filter':0.01, 'step_relaxation':0.5}
params.update({'tilt_angle':0, 'momentum_acceleration':False, 'apply_positivity':False, 'refine_geometry':False})
params.update({'filter_type':'ram-lak', 'lamino_angle':90, 'position_update_smoothing':False, 'ROI':object_ROI.astype(int)})
params.update({'showsorted':True,'plot_results':True, 'plot_results_every':25})

max_binning = 2**(np.ceil(np.log2(np.max(Npix))-np.log2(100)))
binning = 2**(np.arange(max_binning-1)[::-1])

print('\nAlignment Parameters:\n',params,'\n')
print('Binning factors : {}\n'.format(binning) )

######################################################################
# Cross-Correlation Alignment of Raw Data - only rough guess to ease the alignment (TODO)

# par = {'filter_pos':Nangles/4, 'filter_data':0.01, 'max_iter':10, 'precision': 0.1, 'binning':binning[-1]} 

## Implement Tomviz XCorr Method instead
# (sinogram, xcor_shift) = tomoAlign.align_tomo_Xcorr(sinogram, theta, par)

######################################################################
# Consistency-Based Alignment

ker = signal.gaussian(np.maximum(3,np.ceil(params['high_pass_filter']*width_sinogram)), 1/2.5) * signal.windows.hann(5)[1:4].reshape(3,1)
sino_weights = np.maximum(0,1-signal.convolve(np.abs(sinogram) > 2, ker.reshape(3,3,1), 'same'))
tomoAlign = projection_refine.tomo_align(sinogram, theta, sino_weights)
tomoAlign.initialize_plot()

for jj in range(len(binning)):
    params['binning'] = binning[jj]

    # Self Consisency based alignment procedure based on ASTRA toolbox
    (shift, err) = tomoAlign.tomo_consistency_linear(shift, params)
    tomoAlign.sinogram = sinogram

    # Plotting
    # TODO 
