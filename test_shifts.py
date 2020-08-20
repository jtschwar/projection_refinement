import scipy.io as sio
import numpy as np
import shifts 

sinogram = sio.loadmat('sinogram.mat')
sinogram = sinogram['sinogram']
Npix = sinogram.shape

shift = np.zeros([sinogram.shape[2],2])

theta = sio.loadmat('theta.mat')
theta = theta['theta']

binning = 4
smooth = 5
interp_sign = 1
ROI = np.array([31,191,0,288],dtype=int)

linearShifts = shifts.shifts()
sinogram = linearShifts.imshift_generic(sinogram, shift, Npix, smooth, ROI, binning, interp_sign)

print('Finished')

sinoDic = {'sino':sinogram}
sio.savemat('/Users/hovdengroup/cSAXS_matlab_tomo_shared/bin_sino.mat',sinoDic)


