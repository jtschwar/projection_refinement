from scipy import signal
import numpy as np
import shifts
import tqdm

class tomo_align:
    def __init__(input_sino, input_angles, par):
        self.sinogram = input_sino
        self.angles = input_angles
        self.par = par
	       self.binning = par['binning']
		      self.min_step_size = par['min_step_size']
		      self.max_iter = par['max_iter']
		      self.use_TV = par['use_TV']
        self.shifts = par['shifts']

    def main_loop():
        print('TBD')

    def tomo_consistency_linear():
		      print('Starting align_tomo_consistency_linear')
		      Nangles = angles.shape[0]
		      (Nlayers,width_sinogram) = input_sino.shape[:1]

		      binFactor = self.params['binning']
		      print('Shifting Sinograms and binning = ' + str(binFactor))
        # Downsample sinogram and shifts


		      # Shift to the Last Optimal Position + Remove Edge Issues (Line: 235) + Downsample data
		      self.sinogram_shifted = shifts.imshift_generic(self.sinogram, self.shifts, 5, self.ROI, binFactor)

		      # Sort by angle if Requested, but only after binning to make it faster

		      # Initialize ASTRA FBP
        # Determine the optimal size based on downsampled sinogram

		      # Prepare some Auxiliary Variables
		      win = signal.tukeywin(width_sinogram,0.2)
		      win = signal.tukeywin(Nlayers,0.2)*win

		      # Main Loop
		      for ii in tqdm(self.max_iter):
            print('Iteration %d/%d'.format(ii, self.max_iter) )
            
            #step 1: shift (downsampled) sinogram     
            #self.sinogram_shifted = ...

            #step 2: tomo recon (using ASTRA)    
            #self.rec = ...

            #optional step: regularization
            #if self.use_TV:

            #step 3: get reprojections from current tomogram (using ASTRA)
            #self.sinogram_model = 

            #step 4: calculate updated shifts from sinogram and sinogram_model 
            #self.shift_upd = find_optimal_shift(self.sinogram_model, sinogram)
            #step 4.5: add momentum acceleration to improve convergence
            #optional step: add smoothing to shifts to prevent trapping at the local solutions

            #step 5: update shifts
            self.shifts = self.shifts + self.shift_upd;

        # after main loop: prepare outputs
        self.shifts = self.shifts*self.binning
