import scipy.io as sio
import numpy as np
import shifts 

sinogram = sio.loadmat('sinogram.mat')
sinogram = sinogram['sinogram']
shift = np.zeros([sinogram.shape[2],2])

theta = sio.loadmat('theta.mat')
theta = theta['theta']

Npix = sinogram.shape
binning = 4
smooth = 5
interp_sign = 1
ROI = np.array([31,191,0,288],dtype=int)

linearShifts = shifts.shifts()
sinogram = linearShifts.imshift_generic(sinogram, shift, Npix, smooth, ROI, binning, interp_sign)

