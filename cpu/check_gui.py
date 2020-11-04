import scipy.io as sio
import numpy as np
import h5py 

#sinogram = sio.loadmat('sinogram.mat')
#sinogram = sinogram['sinogram']
#(Nx, Ny, Nangles) = sinogram.shape

file = h5py.File('bowtie_tiltseries.h5','r')
sinogram = file['tiltSeries'][:,128:512-128,:]
print(sinogram.shape)

from guiViewer import sliceZ
sliceZ(sinogram)

print('Viewed Sinogram!')
