from scipy.signal import convolve
from numpy.fft import fft2, ifft2, fftshift, ifftshift
import scipy.io as sio
import numpy as np
import FFTW

# These scripts are part of the cSAXS_toolbox

class shifts: 

	def __init__(self):
		self.sinogram = None
		self.shifts = None
		self.FFTW = None

## --------------------------------------------------------------------------------------------------------
	
	# Apply imshift_fft on provided image that was first upsampled to Npix (if needed) and 
	# cropped to region ROI. After shifting, the image is downsampled.
	def imshift_generic(self, img, shift, Npix, smooth, ROI, downsample, interp_sign):
		## Inputs:
		## **img            2D stacked image
		## **shift          Nx2 vector of shifts applied on the image
		## **Npix           2x1 int, size of the image to be upsampled before shift
		## **smooth         How many pixels around edges will be smoothed before shifting the array
		## **ROI            Array, used to crop the array to smaller size
		## **downsample     Downsample factor, 1 == no downsampling
		## **intep_sign   Interpolation method: (1) linear (2)

		## Outputs:
		## ++img            2D Stacked Image

		real_img = np.all(np.isreal(img))

		# Debugger
		# import pdb; pdb.set_trace()

		if np.any(shift != 0):
			smooth_axis = 3 - np.nonzero(np.any(shift != 0,axis=0))[0]
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
			img = img / self.imgaussfilt3_conv(np.ones([Np[0],Np[1],1]), [downsample, downsample, 0])

			outShape = np.array([int(np.ceil(Np[0]/downsample/2)*2), int(np.ceil(Np[1]/downsample/2)*2)])
			img = self.interpolateFT_centered(self.smooth_edges(img, 2*downsample), outShape, interp_sign)

		if real_img:
			img = np.real(img)

		return img

## --------------------------------------------------------------------------------------------------------

	## Take stake of 2D images and smooths boundaries to avoid sharp edge artifacts during imshift_fft
	def smooth_edges(self, img, win_size):
	

		dims = np.array([0,1],dtype=np.int)
		Npix = img.shape
		
		for ii in dims:
			win_size = max(win_size,3)

			# Get indidces of edge regions
			ind = np.concatenate([np.arange(Npix[ii]-win_size,Npix[ii]), np.arange(win_size)])
			ker_size = np.array([1,1])
			ker_size[ii] = win_size

			if ii == 0:
				img_temp = img[ind,]
			else:
				img_temp = img[:,ind,:]

			kernel = self.gausswin(win_size,2.5)
			kernel = kernel.reshape(win_size,1,1)
			
			# smooth across the image edges
			img_tmp = convolve(img_temp, kernel,mode='same')
			
			#avoid boundary issues form convolution
			boundary_shape = np.array([1,1,1])
			boundary_shape[ii] = len(ind)
			boundary_shape = tuple(boundary_shape)

			img_tmp = img_tmp / convolve(np.ones(boundary_shape),kernel,mode='same')

			if ii == 0:
				img[ind,] = img_tmp
			else:
				img[:,ind,:] = img_tmp

		return img
		
## --------------------------------------------------------------------------------------------------------		
	
	## Apply shifts with subpixel accuracy that can be different for each frame.
	def imshift_fft(self,img, xShifts, yShifts, apply_fft):
		## Inputs:
		## **img            input image / stack of images
		## **shifts         applied shifts for each frame (Nz x 2)
		## **Weights        Apply Importance weighting to avoid noise in low reliability regions

		## Outputs:
		## ++img            shifted image / stack of images

		Np = img.shape

		# Shift Only Along One Axis -> Faster (TODO)
		if np.all(xShifts == 0):
			print('Tbd: Shift Alone One Axis')
			# img = imshift_fft_ax(img, y, 1)
		elif np.all(yShifts == 0):
			print('Tbd: Shift Alone One Axis')
			# img = imshift_fft_ax(img, x, 2)

		# 2D FFT Shifting
		else:

			real_img = np.all(np.isreal(img))

			if apply_fft:
				img = fft2(img,axes=(0,1))

			# Shift Along X-Direction
			arrayMin = -np.floor(Np[1]/2)
			arrayMax = np.ceil(Np[1]/2)
			xGrid = ifftshift(np.arange(arrayMin,arrayMax))/Np[1]

			# Output shape : 1 x ny x nz 
			if np.isscalar(xShifts):
				X = (xGrid.reshape(Np[1],1) * xShifts).reshape(1,Np[1],1)
			else:
				X = (xGrid.reshape(Np[1],1) @ xShifts.reshape(1,Np[2])).reshape(1,Np[1],Np[2])
			X = np.exp((-2*(1j)*np.pi)*X)

			img = img * X

			# Shift Along Y-Direction
			arrayMin = -np.floor(Np[0]/2)
			arrayMax = np.ceil(Np[0]/2)
			yGrid = ifftshift(np.arange(arrayMin,arrayMax))/Np[0]

			# Output shape : nx x 1 x nz 
			if np.isscalar(yShifts):
				Y = (yGrid.reshape(Np[0],1) * yShifts).reshape(Np[0],1,1)
			else:
				Y = (yGrid.reshape(Np[0],1) @ yShifts.reshape(1,Np[2])).reshape(Np[0],1,Np[2])
			Y = np.exp((-2*(1j)*np.pi)*Y)

			img = img * Y

			if apply_fft:
				img = ifft2(img,axes=(0,1))

			if real_img:
				img = np.real(img)

		return img

## --------------------------------------------------------------------------------------------------------
	
	# Perform FT interpolation of provided stack of images using FFT so that the center of mass is not
	# modified after the resolution change. This function is critical for subpixel accurate up/downsampling.
	def interpolateFT_centered(self, img, Np_new, interp_sign):
		## Inputs:
		## **img            - input image / stack of images
		## **Np_new			-(2x1 vector) Size of Interpolated Array
		## **interp_sign    - +1 or -1, sign that adds extra 1px shift. 

		## Outputs:
		## ++img            - Output complex image	

		Np = img.shape
		Np_new = 2 + Np_new
		real_img = np.all(np.isreal(img))

		scale = np.prod(Np_new - 2) / np.prod(Np[:2])
		downsample = int(np.ceil( np.sqrt(1/scale) ))

		# Apply Padding in X/Y to Account for Boundary Issues
		padShape = ((downsample,downsample),(downsample,downsample),(0,0))
		img = np.pad(img, padShape, 'symmetric')

		# Go to the fourier space
		img = fft2(img,axes=(0,1))

		# Apply +/- 0.5 px Shift
		img = self.imshift_fft(img, interp_sign*-0.5, interp_sign*-0.5, False)

		# Crop in Fourier Space
		img = self.ifftshift_2D(self.crop_pad(self.fftshift_2D(img), Np_new))

		# Apply +/- 0.5 px Shift in Cropped Space
		img = self.imshift_fft(img, interp_sign*-0.5, interp_sign*-0.5, False)

		# Return to the Real Space
		img = ifft2(img,axes=(0,1))

		# Scale to keep the average constant
		img = img * scale

		# Remove the Padding 
		img = img[ 1:-1, 1:-1, :]

		# Perserve Complexity
		if real_img:
			img = np.real(img)

		return img

## --------------------------------------------------------------------------------------------------------
	
	# Fast version of fftshift for stack of 2D images.
	def fftshift_2D(self,img):
		## Inputs:
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
		## Inputs:
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
		# Inputs: 
		#   **img                input image
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

		imout = np.zeros( np.append(Nout, Nin[2]), dtype=type(img))

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
		# Inputs: 
		#   **A           		3D volume to be smoothed 
		#   **filter_size       gaussian smoothing constant, scalar or use vector for anizotropic kernel smoothing
		# returns: 
		#   ++A           		smoothed volume 

		ker = self.get_kernel(filter_size[0])
		shape = ker.shape[0]

		X = convolve(X,ker.reshape(shape,1,1),mode='same')
		X = convolve(X,ker.reshape(1,shape,1),mode='same')

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

