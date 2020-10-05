from numpy.fft import fft, ifft, fft2, ifft2, fftshift, ifftshift
import numpy as np
import shifts

class refinements:

    def __init__(self,inPar):
        self.par = inPar
        self.FFTW_full = None
        self.FFTW_downsample = None
        
## --------------------------------------------------------------------------------------------------------

    def refine_geometry(self, sinogram_measured, sino_model, geom, iter):

        # Debugger
        # import pdb; pdb.set_trace()

        if iter == 0:
            geometry_corr = np.array([ np.mean(self.par['lamino_angle']), np.mean(geom['tilt_angle']), np.mean(geom['skewness_angle']) ])
 
        resid_sino = self.get_resid_sino(sino_model,sinogram_measured, self.par['high_pass_filter'])

        step_relation = 0.01
        (dX, dY) = self.get_img_grad(sino_model)

        # get tilt angle correction
        Dvec = dX * np.linspace(-1,1,dX.shape[0]).reshape(dX.shape[0],1,1) - dY * np.linspace(-1,1,dY.shape[1]).reshape(1,dX.shape[1],1)
        optimal_shift = self.get_GD_update(Dvec, resid_sino, self.par['high_pass_filter'])
        geom['tilt_angle'] = geom['tilt_angle'] + step_relation * np.rad2deg(optimal_shift)

        # get shear gradient
        Dvec = dY * np.linspace(-1,1,dY.shape[1]).reshape(1,dX.shape[1],1)
        optimal_shift = self.get_GD_update(Dvec, resid_sino, self.par['high_pass_filter'])
        geom['skewness_angle'] = geom['skewness_angle'] + step_relation * np.rad2deg(optimal_shift)

        return geom

## --------------------------------------------------------------------------------------------------------

    # Calculate the optimal step length for optical flow based
    # image alignment
    def get_GD_update(self, dX, resid, filter):

        dX = self.imfilter_high_pass_1d(dX, 1, filter, True)
        optimal_shift = np.sum(resid * dX, axis=(0,1)) / np.sum(dX**2, axis=(0,1))

        return optimal_shift

## --------------------------------------------------------------------------------------------------------

    # Get vertical and horizontal gradient of image using FFT
    def get_img_grad(self, img):
    # Inputs:
    #   **img   - stack of images 
    #   **axis  - direction of the derivative 
    # *returns*
    #  ++[dX, dY]  - image gradients 

        isReal = np.all(np.isreal(img))
        Np = img.shape

        X = 2 * (1j) * np.pi * ifftshift( np.arange(-np.floor(Np[1]/2), np.ceil(Np[1]/2)) ) / Np[1]
        dX = fft(img, axis = 1) * X.reshape(1,Np[1],1)
        dX = ifft(dX, axis = 1)

        Y = 2 * (1j) * np.pi * ifftshift( np.arange(-np.floor(Np[0]/2), np.ceil(Np[0]/2)) ) / Np[0]
        dY = fft(img, axis = 0) * Y.reshape(Np[0],1,1)
        dY = ifft(dY, axis = 0)

        if isReal: 
            dX = np.real(dX)
            dY = np.real(dY)

        return (dX, dY)

## --------------------------------------------------------------------------------------------------------

    # given the sinogram_model, measured sinogram, and importance weight for each pixel it tries to
    # estimate the most optimal shift betweem sinogram_model and
    # sinogram to minimize weighted difference || W * (sinogram_model - sinogram + alpha * d(sino)/dX )^2 ||
    def find_optimal_shift(self, sinogram_model, sinogram, mass):

        shift_x = np.zeros(sinogram_model.shape[2])
        shift_y = np.zeros(sinogram_model.shape[2])

        resid_sino = self.get_resid_sino(sinogram_model, sinogram, self.par['high_pass_filter'])
        resid_sino = self.imfilter_high_pass_1d(resid_sino,1,self.par['high_pass_filter'], True)

        # Align Horizontal
        dX = self.get_img_grad_filtered(sinogram_model, 0, self.par['high_pass_filter'], 5)
        dX = self.imfilter_high_pass_1d(dX, 1, self.par['high_pass_filter'], True)
        shift_x = - np.sum(dX * resid_sino, axis=(0,1)) / np.sum(dX**2, axis=(0,1))

        # Align Vertical
        dY = self.get_img_grad_filtered(sinogram_model, 1, self.par['high_pass_filter'], 5)
        dY = self.imfilter_high_pass_1d(dY, 0, self.par['high_pass_filter'], True)
        shift_y = - np.sum(dY * resid_sino, axis=(0,1)) / np.sum(dY**2, axis=(0,1) )
        
        shift = np.array([shift_x, shift_y]).T
        err = np.sqrt(np.mean(resid_sino**2,axis=(0,1))) / mass

        return (shift, err)

## --------------------------------------------------------------------------------------------------------

    # calculate filtered difference between sinogram_model  and sinogram 
    # || (sinogram_model  - sinogram) \ast ker ||
    # filtering is used for suppresion of low spatial freq. errors 
    def get_resid_sino(self, sinogram_model, sinogram, high_pass_filter):

        # Calculate residuum
        resid_sino = sinogram_model - sinogram

        # Apply high pass filter => get rid of phase artifacts
        resid_sino = self.imfilter_high_pass_1d(resid_sino, 1, high_pass_filter, True)

        return resid_sino

## --------------------------------------------------------------------------------------------------------

    # calculate filtered image gradient  between sinogram_model  and sinogram 
    # filtering is used for suppresion of low spatial freq. errors 
    def get_img_grad_filtered(self, img, axis, high_pass_filter, smooth_win):

        linearShifts = shifts.shifts()

        img = linearShifts.smooth_edges(img, smooth_win, axis%2)
        isReal = np.all(np.isreal(img))
        Np = img.shape

        if axis == 0:
            X = 2 * (1j) * np.pi * (np.fft.fftshift( np.arange(Np[1])/Np[1] ) - 0.5)
            d_img = fft(img,axis=1)
            d_img = d_img * X.reshape(1,Np[1],1)

            # Apply filter in horizontal direction
            d_img = self.imfilter_high_pass_1d(d_img, 1, high_pass_filter, False)
            d_img = ifft(d_img, axis=1)

        if axis == 1:
            X = 2 * (1j) * np.pi * (np.fft.fftshift( np.arange(Np[0])/Np[0] ) - 0.5)
            d_img = fft2(img, axes=(0,1))
            d_img = d_img * X.reshape(Np[0],1,1)

            # Apply filter in horizontal direction
            d_img = self.imfilter_high_pass_1d(d_img, 1, high_pass_filter, False)
            d_img = ifft2(d_img, axes=(0,1))

        if isReal:
            img = np.real(d_img)

        return img

## --------------------------------------------------------------------------------------------------------

    # IMFILTER_HIGH_PASS_1D applies fft filter along AX dimension that
    # removes SIGMA ratio of the low frequencies 
    def imfilter_high_pass_1d(self, img, ax, sigma, apply_fft):
    # Inputs:
    #   **img           - ndim filtered image 
    #   **ax            - filtering axis 
    #   **sigma         - filtering intensity [0-1 range],  sigma <= 0 no filtering 
    #   **padding       - pad the array to avoid edge artefacts (in pixels) [default==0]
    #   **apply_fft     - if true assume that img is in real space, [default == true]
    # *returns*
    #   ++img           - highpass filtered image 

        Ndims = len(img.shape)
        np.seterr(divide='ignore', invalid='ignore')

        # padding = np.ceil(padding)

        # if padding > 0:
        #   pad_vec = np.zeros([Ndims,1])
        #   pad_vec[ax] = padding
        #   img = np.pad(img, pad_vec, 'symmetric')

        Npix = img.shape
        shape = np.ones(Ndims,dtype=int)
        shape[ax] = Npix[ax]
        isReal = np.all(np.isreal(img))

        if apply_fft:
            img = np.fft.fft(img, axis=ax)

        x = np.arange(-Npix[ax]/2, Npix[ax]/2)/Npix[ax]
        sigma = 256 / (Npix[ax]) * sigma

        if sigma == 0:
            # Use Derivative Filter
            spectral_filter = 2*(1j) * np.pi * (np.fft.fftshift( np.arange(Npix[ax])/Npix[ax] ) - 0.5)
        else:
            spectral_filter = fftshift( np.exp(1/(-x**2/sigma**2))  )

        img = img * spectral_filter.reshape(shape)

        if apply_fft:
            img = np.fft.ifft(img,axis=ax)

        if isReal:
            img = np.real(img)

        return img 

## --------------------------------------------------------------------------------------------------------

    # function for accelerated momentum gradient descent. 
    # the function measured momentum of the subsequent updates and if the
    # correlation between then is high, it will use this information to
    # accelerate the update in the direction of average velocity 
    def add_momentum(self, shifts_memory, velocity_map, acc_axes):
        import scipy.optimize as optimize
        
        shift = shifts_memory[-1,:,:]

        Nmem = shifts_memory.shape[0] - 1

        # apply only for horizontal, vertical seems to be too unstable         
        for jj in acc_axes:

            if np.all(shift[:,jj] == 0):
                continue

            for ii in range(Nmem):
                C[ii] = np.correlate(shift[:,jj], shifts_memory[ii,:,jj].T)

            # estimate optimal friction from previous steps 
            funOptimize = lambda z: np.norm( C - np.exp(-x * np.arange(Nmem, 1,- 1)))
            decay = optimize.fmin( funOptimize, 0 )

            ######################################
            alpha = 2                                          # scaling of the friction , larger == less memory 
            gain = 0.5                                         # smaller -> lower relative speed (less momentum)
            friction = np.min( 1, np.max(0,alpha*decay) )    # smaller -> longer memory, more momentum 
            ######################################

            # update velocity map 
            velocity_map[:,jj] = (1-friction) * velocity_map[:,jj] + shift[:,jj]
            
            # update shift estimates 
            shift[:,jj] = (1-gain) * shift[:,jj]  + gain * velocity_map[:,jj]

        return (shift, velocity_map)

