from cupy.fft import fft, ifft, fft2, ifft2, fftshift, ifftshift
from cupyx.scipy.ndimage import convolve
import numpy as np
import cupy as cu

# These scripts are part of the cSAXS_toolbox

class shifts: 

	def __init__(self):
		pass

## --------------------------------------------------------------------------------------------------------
	
	# Apply imshift_fft on provided image that was first upsampled to cuix (if needed) and 
	# cropped to region ROI. After shifting, the image is downsampled.
	def imshift_generic(self, img, shift, smooth, ROI, downsample, interp_sign):
		## Icuuts:
		## **img            2D stacked image
		## **shift          Nx2 vector of shifts applied on the image
		## **cuix           2x1 int, size of the image to be upsampled before shift
		## **smooth         How many pixels around edges will be smoothed before shifting the array
		## **ROI            Array, used to crop the array to smaller size
		## **downsample     Downsample factor, 1 == no downsampling
		## **intep_sign   Interpolation method: (1) linear (2)

		## Outputs:
		## ++img            2D Stacked Image

		img = cu.array(img)
		real_img = cu.all(cu.isreal(img))

		# Debugger
		# import pdb; pdb.set_trace()

		if np.any(shift != 0):
			smooth_axis = np.nonzero(np.any(shift != 0,axis=0))[0]
			img = self.smooth_edges(img, smooth, smooth_axis)
			img = self.imshift_fft(img, shift[:,0], shift[:,1], True)

		# Crop the FOV after shift and before "downsample"
		if ROI.size:
			# Crop to smaller ROI if provided 
			img = img[ ROI[0]:ROI[1], ROI[2]:ROI[3] ]	
		# Apply crop after imshift_fft

		Np = img.shape

		# perform interpolation instead of downsample , it provides more accurate results 
		if downsample > 1:
			img = self.imgaussfilt3_conv(img, [downsample, downsample, 0])

			#Correct for boundary effects of the convolution based smoothing
			img = img / self.imgaussfilt3_conv(cu.ones([Np[0],Np[1],1]), [downsample, downsample, 0])

			outShape = np.array([int(cu.ceil(Np[0]/downsample/2)*2), int(cu.ceil(Np[1]/downsample/2)*2)])
			img = self.interpolateFT_centered(self.smooth_edges(img, int(2*downsample), np.array([0,1],dtype=int)), outShape, interp_sign)

		if real_img:
			img = cu.real(img)

		return cu.asnumpy(img)

## --------------------------------------------------------------------------------------------------------

	## Take stake of 2D images and smooths boundaries to avoid sharp edge artifacts during imshift_fft
	def smooth_edges(self, img, win_size, dims):
		#   Icuuts:
	    #       **img - 2D stacked array, smoothing is done along first two dimensions 
	    #       **win_size - size of the smoothing region, default is 3 
	    #       **dims - list of dimensions along which will by smoothing done 
	    #   Outputs: 
		#       ++img - smoothed array 

		if np.isscalar(dims):
			dims = np.array([dims], dtype=int)

		Npix = img.shape
		
		for ii in dims:

			win_size = max(win_size,3)

			# Get indidces of edge regions
			ind = np.concatenate([np.arange(Npix[ii]-win_size,Npix[ii],dtype=int), np.arange(win_size,dtype=int)])
			ker_size = np.array([1,1])
			ker_size[ii] = win_size

			if ii == 0:
				img_temp = img[ind,]
			else:
				img_temp = img[:,ind,:]

			kernel = self.gausswin(win_size,2.5)

			if ii == 0:
				kernel = kernel.reshape(win_size,1,1)		
			else:
				kernel = kernel.reshape(1,win_size,1)
			kernel = cu.array(kernel)

			# smooth across the image edges
			img_tmp = convolve(img_temp, kernel)
			
			#avoid boundary issues form convolution
			boundary_shape = np.array([1,1,1])
			boundary_shape[ii] = len(ind)
			boundary_shape = tuple(boundary_shape)

			img_tmp = img_tmp / convolve(cu.ones(boundary_shape),kernel)

			if ii == 0:
				img[ind,:,:] = img_tmp
			else:
				img[:,ind,:] = img_tmp

		return img
		
## --------------------------------------------------------------------------------------------------------		
	
	## Apply shifts with subpixel accuracy that can be different for each frame.
	def imshift_fft(self,img, xShifts, yShifts, apply_fft):
		## Icuuts:
		## **img            icuut image / stack of images
		## **shifts         applied shifts for each frame (Nz x 2)
		## **Weights        Apply Importance weighting to avoid noise in low reliability regions

		## Outputs:
		## ++img            shifted image / stack of images

		Np = img.shape

		# Shift Only Along One Axis -> Faster (TODO)
		if np.all(xShifts == 0):
			print('Tbd: Shift Along One Axis (X)')
			# img = self.imshift_fft_ax(img, y, 1)
		elif np.all(yShifts == 0):
			print('Tbd: Shift Along One Axis  (Y)')
			# img = self.imshift_fft_ax(img, x, 2)

		# 2D FFT Shifting
		else:

			real_img = cu.all(cu.isreal(img))

			if apply_fft:
				img = fft2(img,axes=(0,1))

			# Shift Along X-Direction
			arrayMin = -np.floor(Np[1]/2)
			arrayMax = np.ceil(Np[1]/2)
			xGrid = ifftshift(cu.arange(arrayMin,arrayMax))/Np[1]

			# import pdb; pdb.set_trace()

			# Output shape : 1 x ny x nz 
			if np.isscalar(xShifts):
				X = (xGrid.reshape(Np[1],1) * cu.array(xShifts)).reshape(1,Np[1],1)
			else:
				X = (xGrid.reshape(Np[1],1) @ cu.array(xShifts).reshape(1,Np[2])).reshape(1,Np[1],Np[2])
			X = cu.exp((-2*(1j)*cu.pi)*X)

			img = img * X

			# Shift Along Y-Direction
			arrayMin = -np.floor(Np[0]/2)
			arrayMax = np.ceil(Np[0]/2)
			yGrid = ifftshift(cu.arange(arrayMin,arrayMax))/Np[0]

			# Output shape : nx x 1 x nz 
			if cu.isscalar(yShifts):
				Y = (yGrid.reshape(Np[0],1) * cu.array(yShifts)).reshape(Np[0],1,1)
			else:
				Y = (yGrid.reshape(Np[0],1) @ cu.array(yShifts).reshape(1,Np[2])).reshape(Np[0],1,Np[2])
			Y = cu.exp((-2*(1j)*cu.pi)*Y)

			img = img * Y

			if apply_fft:
				img = ifft2(img,axes=(0,1))

			if real_img:
				img = cu.real(img)

		return img

## --------------------------------------------------------------------------------------------------------
	
	# Perform FT interpolation of provided stack of images using FFT so that the center of mass is not
	# modified after the resolution change. This function is critical for subpixel accurate up/downsampling.
	def interpolateFT_centered(self, img, Np_new, interp_sign):
		## Icuuts:
		## **img            - icuut image / stack of images
		## **cu_new			-(2x1 vector) Size of Interpolated Array
		## **interp_sign    - +1 or -1, sign that adds extra 1px shift. 

		## Outputs:
		## ++img            - Output complex image	

		Np = img.shape
		Np_new = 2 + Np_new
		real_img = cu.all(cu.isreal(img))

		scale = np.prod(Np_new - 2) / np.prod(Np[:2])
		downsample = int(np.ceil( np.sqrt(1/scale) ))

		# Apply Padding in X/Y to Account for Boundary Issues
		padShape = ((downsample,downsample+1),(downsample,downsample+1),(0,0))
		img = cu.pad(img, padShape, 'symmetric')

		# Go to the fourier space
		img = fft2(img,axes=(1,0))

		# Apply +/- 0.5 px Shift
		img = self.imshift_fft(img, interp_sign*-0.5, interp_sign*-0.5, False)

		# Crop in Fourier Space
		img = self.ifftshift_2D(self.crop_pad(self.fftshift_2D(img), Np_new))

		# Apply +/- 0.5 px Shift in Cropped Space
		img = self.imshift_fft(img, interp_sign*0.5, interp_sign*0.5, False)
		# img = self.imshift_fft(img, interp_sign*-0.5, interp_sign*-0.5, False)

		# Return to the Real Space
		img = ifft2(img,axes=(0,1))

		# Scale to keep the average constant
		img = img * scale

		# Remove the Padding 
		img = img[ 1:-1, 1:-1, :]

		# Perserve Complexity
		if real_img:
			img = cu.real(img)

		return img

## --------------------------------------------------------------------------------------------------------
	
	# Fast version of fftshift for stack of 2D images.
	def fftshift_2D(self,img):
		## Icuuts:
		## **img 		- Stack of 2D images

		## Outputs:
		## ++img 		- Stack of 2D images after fftshift along first 2 dimensions

		numDims = 2
		idx = []
		for kk in range(numDims):
			m = img.shape[kk]
			p = int(np.ceil(m/2))
			idx.append( np.concatenate((np.arange(p,m), np.arange(p))) )

		return img[idx[0],:,:][:,idx[1],:]

## --------------------------------------------------------------------------------------------------------
	
	# Fast version of ifftshift for stack of 2D images.
	def ifftshift_2D(self, img):
		## Icuuts:
		## **img 		- Stack of 2D images

		## Outputs:
		## ++img 		- Stack of 2D images after fftshift along first 2 dimensions

		numDims = 2
		idx = []
		for kk in range(numDims):
			m = img.shape[kk]
			p = int(np.floor(m/2))
			idx.append( np.concatenate((np.arange(p,m), np.arange(p))) )

		return img[idx[0],:,:][:,idx[1],:]

## --------------------------------------------------------------------------------------------------------
	
	# Adjusts the size by zero padding or cropping.
	def crop_pad(self, img, outSize):
		# Icuuts: 
		#   **img                icuut image
		#   **outsize            size of final image
		# *optional:*
		#   **fill               value to fill padded regions (Default = 0)
		
		# Outputs: 
		#   ++imout              cropped image 

		Nin = np.array([img.shape[0], img.shape[1], img.shape[2]])
		Nout = outSize

		center = np.floor(Nin[:2]/2) 
		centerOut = np.floor(Nout/2) 
		cenout_cen = centerOut - center

		imout = cu.zeros( np.append(Nout, Nin[2]), dtype=type(img))

		# import pdb; pdb.set_trace()
		# xCrop = np.arange( np.maximum(cenout_cen[0],0), np.minimum(cenout_cen[0]+Nin[0],Nout[0]) )
		# yCrop = np.arange( np.maximum(cenout_cen[1],0), np.minimum(cenout_cen[1]+Nin[1],Nout[1]) )

		xCrop = np.arange( np.maximum(-cenout_cen[0],0), np.minimum(-cenout_cen[0]+Nout[0],Nin[0]), dtype=int )
		yCrop = np.arange( np.maximum(-cenout_cen[1],0), np.minimum(-cenout_cen[1]+Nout[1],Nin[1]), dtype=int )

		imout = img[xCrop,:,:][:,yCrop,:]

		return imout

## --------------------------------------------------------------------------------------------------------
	
	# apply gaussian smoothing along all three dimensions using convolution, 
	# faster than matlab alternative
	def imgaussfilt3_conv(self, X, filter_size):
		# Icuuts: 
		#   **A           		3D volume to be smoothed 
		#   **filter_size       gaussian smoothing constant, scalar or use vector for anizotropic kernel smoothing
		# returns: 
		#   ++A           		smoothed volume 

		ker = cu.array(self.get_kernel(filter_size[0]))
		shape = ker.shape[0]

		X = convolve(X,ker.reshape(shape,1,1))
		X = convolve(X,ker.reshape(1,shape,1))

		return X

	def get_kernel(self, filter_size):

		grid = np.arange(-np.ceil(2*filter_size),np.ceil(2*filter_size)+1)/filter_size
		ker = np.exp(-grid**2)
		ker /= np.sum(ker)

		return ker

## --------------------------------------------------------------------------------------------------------

	# Compute Gaussian window
	def gausswin(self, L,a):
		N = L - 1
		n = np.arange(L) - N/2
		return np.exp(-0.5*(a*n/(N/2))**2)

## --------------------------------------------------------------------------------------------------------

	# Apply Accurate Affine Deformation on image 
	def imdeform_affine_fft(self, img, affine_matrix, shift):
		# Icuuts:  
		#     **img             - 2D or stack of 2D images 
		#     **affine_matrix   - 2x2xN affine matrix 
		#     **shift           - Nx2 vector of shifts to be applied 
		# *returns*: 
		#     ++img             - deformed image 

		img = cu.array(img)

		if np.any(shift != 0):
			img = self.imshift_fft(img, shift[:,0], shift[:,1], True)

		# if np.all(shift != 0):
		# 	img = self.imshift_fft(img, shift[:,0], shift[:,1], True)

		scale = affine_matrix[0,:]
		rotation = affine_matrix[1,:]
		shear = affine_matrix[2,:]

		# Apply rescaling if non-square pixel sizes. 
		# if cu.any(cu.abs(scale[:]-1) > 1e-5):
		# 	img = self.imrescale_frft(img,scale)

		# Apply shear transformation
		# if np.any(np.abs(shear[:]) > 1e-5):
		# 	img = self.imshear_fft(img,shear,0)

		# Apply rotation 
		if np.any(np.abs(rotation[:])> 1e-5):
			img = self.imrotate_ax_fft(img,rotation,3)

		return cu.asnumpy(img)

## --------------------------------------------------------------------------------------------------------

	# FFT-based image rotation for a stack of images along given axis
	# for the accurate rotation of sampled images (Optic Communications, 1997)
	def imrotate_ax_fft(self, img, theta, axis):
		# Icuuts: 
		#   **img       - stacked array of images to be rotated 
		#   **theta     - rotation angle 
		# *optional*
		#   **axis      - rotation axis (default=3)
		# returns:
		#   ++img       - rotated image 

		if np.all(theta == 0):
			return img

		real_img = cu.all(cu.isreal(img))

		# if axis == 1:
		# 	img = permute(img, [3,2,1])
		# 	theta = -theta 
		# elif: axis == 2:
		# 	img = permute(img, [1,3,2])

		# angle_90_offset = cu.round(theta/90)

		# if angle_90_offset != 0:
		# 	img = cu.rot90(img,angle_90_offset)
		# 	theta = theta - 90*angle_90_offset

		(M,N,_) = img.shape

		#Make possible to rotate each slice with different angle
		Nangles = theta.shape[0]
		theta = theta.reshape(1,1,Nangles)
		xgrid = ifftshift( cu.arange(-np.floor(M/2),np.ceil(M/2) )) / M
		ygrid = ifftshift( cu.arange(-np.floor(N/2),np.ceil(N/2) )) / N
 
		# the 0.5px offset is important to make the rotation equivalent ot matlab imrotate
		Mgrid = cu.arange(M) - np.floor(M/2) + 0.5 
		Ngrid = cu.arange(N) - np.floor(N/2) + 0.5

		Mgrid = Mgrid.reshape(M, 1, 1)
		Ngrid = Ngrid.reshape(1, N, 1)

		(M1, M2) = self.aux_fun(theta, xgrid, ygrid, Mgrid, Ngrid)

		img = ifft( fft(img, axis=1) * M1, axis=1 )
		img = ifft( fft(img, axis=0) * M2, axis=0 )
		img = ifft( fft(img, axis=1) * M1, axis=1 )

		if real_img:
			img = cu.real(img)

		# if axis == 1:
		# 	img = permute(img, [3,2,1])
		# elif: axis == 2:
		# 	img = permute(img, [1,3,2])

		return img


	# Auxiliarly function to be used for GPU kernel merging
	def aux_fun(self, theta, xgrid, ygrid, Mgrid, Ngrid):

		Nx = - cu.array(np.sin(np.deg2rad(theta))) * xgrid.reshape(xgrid.shape[0],1,1)
		Ny = cu.array(np.tan(np.deg2rad(theta/2))) * ygrid.reshape(1,ygrid.shape[0],1)

		M1 = cu.exp( (-2*(1j)*cu.pi) * Mgrid * Ny )
		M2 = cu.exp( (-2*(1j)*cu.pi) * Ngrid * Nx )

		return (M1, M2)

## --------------------------------------------------------------------------------------------------------

	# IMSHEAR_FFT fft-based image shearing function for a stack of images along given axis 
	def imshear_fft(self, img, theta, shear_axis):
		# Inputs: 
		#   **img_stack  - stack of 2D images 
		#   **theta      - shear angle, scalar 
		#   **shear_axis - image axis along which the image will be shared 
		# *returns*: 
		#   ++img        - shreared image 

		if np.all(theta == 0): return img

		real_img = np.all(np.isreal(img))

		if np.any(np.abs(theta) > 45): 
			print('Out of Valid angle range [-45,45], use rot90 to get into valid range')
			return img

		(M,N,_) = img.shape

		# allow different theta for each slice 
		Nangles = theta.shape[0]
		theta = theta.reshape(1,1,Nangles)

		# Rotate images by a combination of shears
		if shear_axis == 0: 
			Ny =  cu.array(np.tan(np.deg2rad(theta/2))) * (ifftshift(cu.arange(-np.floor(N/2), np.ceil(N/2)))/N).reshape(1,N,1)			
			Mgrid = 2*(1j)*cu.pi * (cu.arange(M)-np.floor(M/2)+1)
			Mgrid = Mgrid.reshape(M, 1, 1)
			img = ifft(fft(img, axis=1) * cu.exp(-Mgrid * Ny), axis=1)
		elif shear_axis == 1: 
			Nx = -cu.array(np.sin(np.deg2rad(theta)))   * (ifftshift(cu.arange(-np.floor(M/2), np.ceil(M/2)))/M).reshape(1,M,1)
			Ngrid = 2*(1j)*cu.pi * (cu.arange(N)-np.floor(N/2)+1)
			Ngrid = Ngrid.reshape(N, 1, 1)
			img = ifft(fft(img,axis=0)  * cu.exp(Ngrid * Nx), axis=0)
		else: print('Shear axis has to be 1 or 2')

		if real_img: img = np.real(img)

		return img		
		
