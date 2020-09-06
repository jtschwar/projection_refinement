from scipy import signal
from tqdm import tqdm
import astra_ctvlib
import refinements
import numpy as np
import shifts

class tomo_align:
	def __init__(self, input_sino, input_angles):
		self.sinogram = input_sino
		self.angles = input_angles

	def tomo_consistency_linear(self, optimal_shift, params):
		print('Starting align_tomo_consistency_linear')

		interp_sign = -1
		binFactor = params['binning']
		Nangles = self.angles.shape[0]
		print('Shifting Sinograms and binning = ' + str(binFactor))

		# Shift to the Last Optimal Position + Remove Edge Issues (Line: 235) + Downsample data
		linearShifts = shifts.shifts()
		self.sinogram = linearShifts.imshift_generic(self.sinogram, optimal_shift, 5, binFactor, 'fft', interp_sign)

		# Shift also the weights to correspond to the sinogram
		# weights = np.ones(self.sinogram.shape)
		# weights = linearShifts.imshift_generic(weights, optimal_shift, Np_sinogram, 0, binFactor, 'linear', 0)

		# Prepare some Auxiliary Variables
		(Nlayers,width_sinogram) = self.sinogram.shape[:1]
		win = signal.tukeywin(width_sinogram,0.2)
		if Nlayers > 10:
			win = signal.tukey(Nlayers,0.2).reshape(Nlayers,1) @ win.reshape(1,width_sinogram)

		shift_upd_all = np.zeros([params['max_iter'],Nangles,2], dtype=np.float32)
		shift_velocity =  np.zeros([Nangles,2], dtype=np.float32)
		shift_total =  np.zeros([Nangles,2], dtype=np.float32)
		err = np.zeros(Nangles,dtype=np.float32)

		# Initialize ASTRA FBP
		# Determine the optimal size based on downsampled sinogram
		b = np.zeros([Nlayers, width_sinogram*Nangles])
		tomoObj = astra_ctvlib.astra_ctvlib(Nlayers,width_sinogram, Nangles, np.deg2rad(self.angles))
		tomo_obj.initializeFBP(parms['filter_type'])

		# geometry parameters structure 
		geom = {'tilt_angle':0, 'skewness_angle':0}

		# Class for Calculating Projection Refinements
		calcRefine = refinements.refinements(params)

		# Main Loop
		for ii in tqdm(params['max_iter']):

			#step 1: shift (downsampled) sinogram (Line: 371) 
			sinogram_shifted = self.sinogram
			# weights_shifted = weights

			# geom_mat = [scale, rotation, shear]
			geom_mat = np.array([np.ones((Nangles)), np.ones((Nangles)) * -geom['tilt_angle'], np.ones((Nangles)) * -geom['skewness_angle']])
			sinogram_shifted = linearShifts.imdeform_affine_fft(sinogram_shifted, geom_mat , shift_total)

			# Shift Weights (Lines: 376 & 379)
			# weights_shifted = linearShifts.imshift_linear(weights_shifted, shift_total, 'linear', tomo_obj)
			# weights_shifted = np.maximum(0, weights_shifted * win)

			if ii == 0:
				mass = np.mean(np.abs(sinogram_shifted))

			#step 2: tomo recon (using ASTRA)    
			for s in range(Nslice):
				b[s,:] = sinogram_shifted[s,:,:].transpose().ravel()
			tomo_obj.setTiltSeries(b)
			tomo_obj.FBP()

			#optional step: regularization
			# if self.use_TV:
			#   tomo_obj.tv_chambolle(tvIter)

			for s in range(Nlayers):
				rec[s,:,:] = tomo_obj.getRecon(s)

			#step 3: get reprojections from current tomogram (using ASTRA) (Line 455)
			tomo_obj.forwardProjection()
			sinogram_model = tomo_obj.get_model_projections().reshape(Nlayers, width_sinogram, Nangles)

			# Refine Geometry (ie tilt_angle & shear) (Line 482)
			if params['refine_geometry']:
				geom = calcRefine.refine_geometry(sinogram_shifted, sinogram_model, geom, ii)

			#step 4: calculate updated shifts from sinogram and sinogram_model (Line 494 & 884)
			(shift_upd, err) = calcRefine.find_optimal_shift(sinogram_model, self.sinogram, mass)

			# Do not allow more than 0.5px per iteration (Line 502)
			shift_upd = np.minimum(0.5, np.abs(shift_upd)) * np.sign(shift_upd) * params['step_relaxation']

			# Store update history for momentum gradient acceleration
			shift_upd_all[ii,] = shift_upd

			#step 4.5: add momentum acceleration to improve convergence
			#optional step: add smoothing to shifts to prevent trapping at the local solutions
			if params['momentum_acceleration']:
				momentum_memory = 2
				max_update = np.quantile(np.abs(shift_upd), 0.995)
				if ii > momentum_memory:
					(shift_upd, velocity_map) = calcRefine.add_momentum(shifts_memory, velocity_map, acc_axes)

			shift_upd -= np.median(shift_upd)

			# Prevent outliers when the code decides to quickly oscillate around the solution 
			max_step = np.minimum(np.quantile(np.abs(shift_upd),0.99), 0.5)

			# Do not allow more than 0.5px per iteration (multiplied by binning factor)
			shift_upd = np.minimum(max_step, np.abs(shift_upd)) * np.sign(shift_upd)

			#step 5: update shifts
			shift_total += shift_upd

			# Enforce smoothness of the estimate position update -> in each iteration 
			# smooth the accumlated position udate, this helps against discontinuites 
			# in the update (Line: 558)
			if params['position_update_smoothing']:
				for kk in range(2):
					shift_total[:,kk] = smooth(shift_total(:,kk), max(0, min(1, par.position_update_smoothing)) * Nangles); 

			# Check for Maximal Update

			# # Plot results (Line 572)
			# if params['plot_results']:
			# 	plot_alignment(rec, sinogram_shifted, err, shift_upd, shift_total, angles, par)

		# Prepare outputs to be returned
		params['tilt_angle']     = geom['tilt_angle']
		params['skewness_angle'] = geom['skewness_angle']

		optimal_shift += shifts * binFactor

		# after main loop: prepare outputs
		shifts = shifts * binning

		return (shifts, err)

	def align_tomo_Xcorr(self, params):

		print('TODO: Cross Correlation Alignment')

		return -1 
