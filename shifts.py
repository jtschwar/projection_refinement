from numpy.fft import fftshift, ifftshift
import numpy as np
import FFTW

# These scripts are part of the cSAXS_toolbox

class shifts: 

	def __init__():
		self.sinogram = None
		self.shifts = None
		self.FFTW = None

## --------------------------------------------------------------------------------------------------------
	
	# Apply imshift_fft on provided image that was first upsampled to Npix (if needed) and 
	# cropped to region ROI. After shifting, the image is downsampled.
	def imshift_generic(Npix, smooth, ROI, downsample):
		## Inputs:
		## **img            2D stacked image
		## **shift          Nx2 vector of shifts applied on the image
		## **Npix           2x1 int, size of the image to be upsampled before shift
		## **smooth         How many pixels around edges will be smoothed before shifting the array
		## **ROI            Array, used to crop the array to smaller size
		## **downsample     Downsample factor, 1 == no downsampling
		## **intep_method   Interpolation method: (1) linear (2)

		## Outputs:
		## ++img            2D Stacked Image

		# Necessary ??
		interp_sign = 0

		real_img = np.isreal(img)

		if any(shift[:] != 0):
			smooth_axis = 3 - find(any(shift != 0))
			img = self.smooth_edges(img, smooth, smooth_axis)
			img = self.imshift_fft(img, shift[:,0], shift[:,1], True)

		# Crop the FOV after shift and before "downsample"
		if ROI.size:
			# Crop to smaller ROI if provided 
			img = img[ROI,:]	
		# Apply crop after imshift_fft

		Np = img.shape

		# perform interpolation instead of downsample , it provides more accurate results 
		if downsample > 1:
			img = imgaussfilt3_conv(img, [downsample, downsample, 0])

			#Correct for boundary effects of the convolution based smoothing
			img = img / imgaussfilt3_conv(np.ones([Np[0],Np[1]]), [downsample, downsample, 0])

			img = interpolateFT_centered(smooth_edges(img, 2*downsample, [1,2]), np.ceil(Np[:2]/downsample/2)*2)

		if real_img:
			img = np.real(img)

		return img

## --------------------------------------------------------------------------------------------------------

	## Take stake of 2D images and smooths boundaries to avoid sharp edge artifacts during imshift_fft
	def smooth_edges(img, win_size, dims):
	

		dims = np.array([0,1],dtype=np.int)
		Npix = img.shape
		
		for ii in dims:
			win_size = max(win_size,3)

			# Get indidces of edge regions
			ind = np.concatenate([np.arange(Npix[ii]-win_size,Npix[ii]), np.arange(win_size)])
			ker_size = np.array([1,1])
			ker_size[ii] = win_size
			
			if ii = 0:
				img_temp = img[ind,]
			else:
				img_temp = img[:,ind,:]

			kernel = gausswin(win_size,2.5)
			# smooth across the image edges

			# TODO :: Figure Out 2D Convolution 
			img_tmp = convn(img_temp, kernel,'same')
			#avoid boundary issues form convolution

			boundary_shape = np.array([1,1])
			boundary_shape[ii] = len(ind)

			# TODO :: Figure Out 1D Convolution 
			img_tmp = bsxfun(@rdivide, img_tmp, conv(ones(boundary_shape), kernel, 'same'));
			img(ind{:}) = img_temp

		return img
		
## --------------------------------------------------------------------------------------------------------		
	
	## Apply shifts with subpixel accuracy that can be different for each frame.
	def imshift_fft(img, xShifts, yShifts, apply_fft):
		## Inputs:
		## **img            input image / stack of images
		## **shifts         applied shifts for each frame (Nz x 2)
		## **Weights        Apply Importance weighting to avoid noise in low reliability regions

		## Outputs:
		## ++img            shifted image / stack of images

		Np = img.shape

		# Shift Only Along One Axis -> Faster (TODO)
		if all(x==0):
			print('Tbd: Shift Alone One Axis')
			# img = imshift_fft_ax(img, y, 1)
		elif all(y==0):
			print('Tbd: Shift Alone One Axis')
			# img = imshift_fft_ax(img, x, 2)

		# 2D FFT Shifting
		else:

			real_img = np.isreal(img)

			if apply_fft:
				img = fftw.fft(img)

			# Shift Along X-Direction
			arrayMin = -np.floor(Np[1]/2)
			arrayMax = np.ceil(Np[1]/2)
			xGrid = ifftshift(np.arange(arrayMin-1,arrayMax-1))/Np[1]
			
			# Output shape : 1 x ny x nz 
			X = (xGrid.reshape(Np[1],1) @ xShifts.reshape(1,Np[2]))
			X = np.exp((-2*j*np.pi)*X)

			# Todo: figure out how to run proper multiplication (??)
			img = img * X

			# Shift Along Y-Direction
			arrayMin = -np.floor(Np[0]/2)
			arrayMax = np.ceil(Np[0]/2)
			yGrid = ifftshift(np.arange(arrayMin-1,arrayMax-1))/Np[0]

			# Output shape : nx x 1 x nz 
			Y = (yGrid.reshape(Np[1],1) @ yShifts.reshape(1,Np[2]))
			Y = np.exp((-2*j*np.pi)*Y)

			# Todo: figure out how to run proper multiplication (??)
			img = img * Y

			if apply_fft:
				img = fftw.ifft(img)

			if real_img:
				img = np.real(img)

## --------------------------------------------------------------------------------------------------------
	
	# Perform FT interpolation of provided stack of images using FFT so that the center of mass is not
	# modified after the resolution change. This function is critical for subpixel accurate up/downsampling.
	def interpolateFT_centered(img, Np_new):
		## Inputs:
		## **img            - input image / stack of images
		## **Np_new			-(2x1 vector) Size of Interpolated Array
		## **interp_sign    - +1 or -1, sign that adds extra 1px shift. 

		## Outputs:
		## ++img            - Output complex image	

		Np = img.shape
		Np_new = 2 + Np_new
		real_img = np.isreal(img)

		scale = np.prod(Np_new - 2) / np.prod(Np[:1])
		downsample = np.ceil( np.sqrt(1/scale) )

		# Apply Padding to Account for Boundary Issues
		img = np.pad(img, (downsample,downsample), 'symmetric')

		# Go to the fourier space
		img = fftw.fft(img)

		# Apply +/- 0.5 px Shift
		img = self.imshift_fft(img, interp_sign*-0.5, interp_sign*-0.5, False)

		# Crop in Fourier Space
		img = self.ifftshift_2D(self.crop_pad(self.fftshift_2D(img), Np_new))

		# Apply +/- 0.5 px Shift in Cropped Space
		img = self.imshift_fft(img, interp_sign*-0.5, interp_sign*-0.5, False)

		# Return to the Real Space
		img = fftw.ifft(img)

		# Scale to keep the average constant
		img = img * scale

		# Remove the Padding 
		img = img[2:Np[0],2:Np[1],:]

		# Perserve Complexity
		if real_img:
			img = np.real(img)

## --------------------------------------------------------------------------------------------------------
	
	# Fast version of ifftshift for stack of 2D images.
	def fftshift_2D(img):
		## Inputs:
		## **img 		- Stack of 2D images

		## Outputs:
		## ++img 		- Stack of 2D images after fftshift along first 2 dimensions

## --------------------------------------------------------------------------------------------------------
	
	# Fast version of ifftshift for stack of 2D images.
	def ifftshift_2D(img):
		## Inputs:
		## **img 		- Stack of 2D images

		## Outputs:
		## ++img 		- Stack of 2D images after fftshift along first 2 dimensions

## --------------------------------------------------------------------------------------------------------

	# Compute Gaussian window
	def gausswin(L,a):
		N = L - 1
		n = np.arange(L) - N/2
		return np.exp(-0.5*(a*n/(N/2))**2)

