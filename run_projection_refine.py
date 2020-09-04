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
vert_crop, horiz_crop = Nx / 8, Ny / 8
object_ROI = np.array([vert_crop, Nx - vert_crop, horiz_crop, Ny - horiz_crop])
width_sinogram = np.int(object_ROI[3] - object_ROI[2]) #(Npix = width_sinogram)
Nlayers = np.int(object_ROI[1] - object_ROI[0])

theta = sio.loadmat('theta.mat')
theta = theta['theta']
Nangles = angles.shape[0]

shift = np.zeros([sinogram.shape[2],2])

######################################################################
# Useful Tuning Parameters

params = {'min_step_size':0.01, 'max_iter':500, 'use_TV':False, 'high_pass_filter':0.01, 'step_relaxation':0.5, 'filter_type':'ram-lak', 'lamino_angle':90}
params.update({'tilt_angle':0, 'momentum_acceleration':False, 'apply_positivity':True, 'refine_geometry':True, 'ROI':object_ROI})

max_binning = 2**(np.ceil(np.log2(np.max(Npix))-np.log2(100)))
binning = 2**(np.arange(max_binning-1)[::-1])

print( 'Binning factors : {}'.format(binning) )

tomoAlign = projection_refine.tomo_align(sinogram, theta)

######################################################################
# Cross-Correlation Alignment of Raw Data - only rough guess to ease the alignment (TODO)

# par = {'filter_pos':Nangles/4, 'filter_data':0.01, 'max_iter':10, 'precision': 0.1, 'binning':binning[-1]} 

# (sinogram, xcor_shift) = tomoAlign.align_tomo_Xcorr(sinogram, theta, par)

######################################################################
# Consistency-Based Alignment

for jj in range(len(binning)):
    params['binning'] = binning[jj]
    print('Binning: ' + str(binning[jj]))

    # Self Consisency based alignment procedure based on ASTRA toolbox
    (shift, err) = tomoAlign.align_tomo_consistency_linear(shift, params)

    # Plotting
    # TODO 