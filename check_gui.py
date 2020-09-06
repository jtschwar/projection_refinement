import scipy.io as sio
import numpy as np

sinogram = sio.loadmat('sinogram.mat')
sinogram = sinogram['sinogram']
(Nx, Ny, Nangles) = sinogram.shape

# Choose reconstructed (cropped) region 
vert_crop, horiz_crop = Nx / 8, Ny / 8
object_ROI = np.array([vert_crop, Nx - vert_crop, horiz_crop, Ny - horiz_crop])

from guiViewer import sliceZ
sliceZ_wROI(sinogram)

print('Viewed Sinogram!')