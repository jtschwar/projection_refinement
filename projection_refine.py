from scipy import signal
from tqdm import tqdm
import astra_ctvlib
import numpy as np
import shifts

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

    def tomo_consistency_linear(self):
      print('Starting align_tomo_consistency_linear')
      Nangles = angles.shape[0]
      (Nlayers,width_sinogram) = input_sino.shape[:1]

      binFactor = self.params['binning']
      print('Shifting Sinograms and binning = ' + str(binFactor))
      # Downsample sinogram and shifts

      # Shift to the Last Optimal Position + Remove Edge Issues (Line: 235) + Downsample data
      linearShifts = shifts.shifts()
      self.sinogram = linearShifts.imshift_generic(self.sinogram, self.shifts, 5, self.ROI, binFactor)

      # Sort by angle if Requested, but only after binning to make it faster (Line: 247)

      # Prepare some Auxiliary Variables
      (Nlayers,width_sinogram, _) = sinogram.shape 
      win = signal.tukeywin(width_sinogram,0.2)
      if Nlayers > 10:
        win = signal.tukey(Nlayers,0.2).reshape(Nlayers,1) @ win.reshape(1,width_sinogram)

      # Initialize ASTRA FBP
      # Determine the optimal size based on downsampled sinogram
      b = np.zeros([Nlayers, width_sinogram*Nangles])
      tomoObj = astra_ctvlib.astra_ctvlib(Nlayers,Nray,Nangles, np.deg2rad(self.angles))
      tomo_obj.initializeFBP('ram-lak')

      # geometry parameters structure 
      geom = {'tilt_angle':0, 'skewness_angle':0, 'asymmetry':0}

      #ASTRA needs the reconstruction to be dividable by 32 othewise there
      #will be artefacts in left corner  (Line: 321)
      Npix = np.ceil(Npix/par['binning'])

      # Main Loop
      for ii in tqdm(self.max_iter):
        
        #step 1: shift (downsampled) sinogram (Line: 371) 
        sinogram_shifted = singoram
        geom_mat = np.array([np.ones((Nangles)), np.ones((Nangles))*geom['asymmetry'], np.ones((Nangles)) * -geom['tilt_angle'], np.ones((Nangles)) * -geom['skewness_angle']])
        self.sinogram_shifted = linearShifts.imdeform_affine_fft(sinogram_shifted, geom_mat , shift_total)

        if ii == 1:
          mass = np.mean(np.abs(sinogram_shifted))

        #step 2: tomo recon (using ASTRA)    
        for s in range(Nslice):
          b[s,:] = self.sinogram_shifted[s,:,:].transpose().ravel()
        tomo_obj.setTiltSeries(b)
        tomo_obj.FBP()

        #optional step: regularization
        # if self.use_TV:
        #   tomo_obj.tv_chambolle(tvIter)

        for s in range(Nlayers):
          rec[s,:,:] = tomo_obj.getRecon(s)

        #step 3: get reprojections from current tomogram (using ASTRA) (Line 455)
        tomo_obj.forwardProjection()
        self.sinogram_model = tomo_obj.get_model_projections()

        # Refine Geometry (ie tilt_angle) (Line 482)
        #geom = refine_geometry(rec, sinogram_shifted, sinogram_model, angles, par, geom)

        #step 4: calculate updated shifts from sinogram and sinogram_model (Line 494 & 884)
        (self.shift_upd, err) = find_optimal_shift(self.sinogram_model, sinogram, mass, self.par)
        
        #step 4.5: add momentum acceleration to improve convergence
        #optional step: add smoothing to shifts to prevent trapping at the local solutions

        #step 5: update shifts
        self.shifts = self.shifts + self.shift_upd

        # after main loop: prepare outputs
        self.shifts = self.shifts*self.binning
