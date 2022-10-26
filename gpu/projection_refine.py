from pytvlib import initialize_algorithm, run
from pyqtgraph.Qt import QtCore, QtGui
import pyqtgraph as pg
from scipy import signal
from tqdm import tqdm
import astra_ctvlib
import numpy as np

class tomo_align:

	def __init__(self, input_sino, input_angles):
		self.sinogram = input_sino
		self.angles = input_angles
		#self.weights = input_weights

	def tomo_consistency_linear(self, optimal_shift, params):
		print('[align] : Starting align_tomo_consistency_linear')

		# Debugger
		# import pdb; pdb.set_trace()

		interp_sign = -1
		binFactor = params['binning']
		Nangles = self.angles.shape[0]
		print('[align] : Shifting Sinograms and binning = ' + str(binFactor))

		if np.any(self.sinogram.shape[:2] // binFactor > 128): import shifts_gpu as shifts; use_gpu = True
		else: import shifts_cpu as shifts; use_gpu = False

		# Shift to the Last Optimal Position + Remove Edge Issues (Line: 235) + Downsample data
		linearShifts = shifts.shifts()
		self.sinogram = linearShifts.imshift_generic(self.sinogram, optimal_shift, 5, params['ROI'], binFactor, interp_sign)
		print('Sinogram Shape: ' + str(self.sinogram.shape))

		# Sort by angle if requested, but only after binning to make it faster
		if params['showsorted']:
			ang_order = np.argsort(self.angles)
			self.sinogram = self.sinogram[:,:,ang_order]
			self.angles = self.angles[ang_order]
			optimal_shift = optimal_shift[ang_order,:]
		else:
			ang_order = np.arange(Nangles)

		# Prepare some Auxiliary Variables
		(Nlayers,width_sinogram) = self.sinogram.shape[:2]

		shift_upd_all = np.zeros([params['max_iter'],Nangles,2], dtype=np.float32)
		shift_velocity =  np.zeros([Nangles,2], dtype=np.float32)
		shift_total =  np.zeros([Nangles,2], dtype=np.float32) 			#total update of the optimal shifts
		err = np.zeros([params['max_iter'], Nangles], dtype=np.float32) #error evolution 

		# Initialize ASTRA FBP
		# Determine the optimal size based on downsampled sinogram
		b = np.zeros([Nlayers, width_sinogram*Nangles])
		sinogram_model = np.zeros([Nlayers, width_sinogram, Nangles])

		if params['plot_results']:
			rec = np.zeros([Nlayers, width_sinogram, width_sinogram], dtype=np.float32)

		tomo = astra_ctvlib.astra_ctvlib(Nlayers,width_sinogram, np.deg2rad(self.angles))
		initialize_algorithm(tomo, params['alg'], params['initAlg'])
		tomo.restart_recon()

		# geometry parameters structure 
		geom = {'tilt_angle':0, 'skewness_angle':0}

		# Class for Calculating Projection Refinements
		if use_gpu: import refinements_gpu as refinements
		else: import refinements_cpu as refinements
		calcRefine = refinements.refinements(params)

		# Main Loop
		for ii in tqdm(range(params['max_iter'])):

			#step 1: shift (downsampled) sinogram (Line: 371) 
			sinogram_shifted = self.sinogram

			# geom_mat = [scale, rotation, shear]
			if hasattr(params['tilt_angle'],"__len__"):
				geom_mat = np.array([np.ones((Nangles)), params['tilt_angle'], np.ones((Nangles)) * -geom['skewness_angle']])				
			else: # load original rotations if available
				geom_mat = np.array([np.ones((Nangles)), np.ones((Nangles)) * -geom['tilt_angle'], np.ones((Nangles)) * -geom['skewness_angle']])				
			# geom_mat = np.array([np.ones((Nangles)), params['refine_mask'] * -geom['tilt_angle'], params['refine_mask'] * -geom['skewness_angle']])
			sinogram_shifted = linearShifts.imdeform_affine_fft(sinogram_shifted, geom_mat , shift_total)

			if ii == 0: mass = np.median(np.mean(np.abs(sinogram_shifted),axis=(0,1)))

			#step 2: tomo recon (using ASTRA)    
			for s in range(Nlayers):
				b[s,:] = sinogram_shifted[s,:,:].transpose().ravel()
			tomo.set_tilt_series(b)
			run(tomo, params['alg'])

			#optional step: regularization
			if params['use_TV']: tomo.tv_gd(20, 0.1)

			#step 3: get reprojections from current tomogram (using ASTRA) (Line 455)
			tomo.forward_projection()

			reproj = tomo.get_model_projections()
			for s in range(Nangles):
				sinogram_model[:,:,s] = reproj[:,s*width_sinogram:(s+1)*width_sinogram]

			# Refine Geometry (ie tilt_angle & shear) (Line 482)
			if params['refine_geometry']:
				geom = calcRefine.refine_geometry(sinogram_shifted, sinogram_model, geom, ii)

			#step 4: calculate updated shifts from sinogram and sinogram_model (Line 494 & 884)
			(shift_upd, err[ii,:]) = calcRefine.find_optimal_shift(sinogram_model, sinogram_shifted, mass)

			# Do not allow more than 0.5px per iteration (Line 502)
			shift_upd = np.minimum(0.5, np.abs(shift_upd)) * np.sign(shift_upd) * params['step_relaxation']

			# Store update history for momentum gradient acceleration
			shift_upd_all[ii,] = shift_upd

			#step 4.5: add momentum acceleration to improve convergence
			#optional step: add smoothing to shifts to prevent trapping at the local solutions
			if params['momentum_acceleration']:
				momentum_memory = 2
				max_update = np.quantile(np.abs(shift_upd), 0.995, axis=0)
				if ii >= momentum_memory:
					(shift_upd, shift_velocity) = calcRefine.add_momentum(shift_upd_all[ii-momentum_memory:ii,:,:], shift_velocity, max_update * binFactor < 0.5 )

			shift_upd[:,1] = shift_upd[:,1] - np.median(shift_upd[:,1])

			# Prevent outliers when the code decides to quickly oscillate around the solution 
			max_step = np.minimum(np.quantile(np.abs(shift_upd),0.99, axis=0), 0.5)

			# Do not allow more than 0.5px per iteration (multiplied by binning factor)
			shift_upd = np.minimum(max_step, np.abs(shift_upd)) * np.sign(shift_upd)

			#step 5: update total position shift
			shift_total += shift_upd

			# Enforce smoothness of the estimate position update -> in each iteration 
			# smooth the accumlated position udate, this helps against discontinuites 
			# in the update (Line: 558) (TODO)
			if params['position_update_smoothing']:
				for kk in range(2):
					shift_total[:,kk] = self.smooth(shift_total[:,kk], np.maximum(0, np.minimum(1, params['position_update_smoothing'])) * Nangles); 

			# # Plot results (Line 572)
			if params['plot_results'] and (ii % params['plot_results_every'] == 0):
				recSlice = tomo.get_recon(int(Nlayers//2))
				sinoSlice = sinogram_shifted[int(Nlayers//2),]
				self.plot_alignment(recSlice, sinoSlice, err, shift_upd, shift_total, self.angles, params, ii)
				self.removeImageItems()

			# Check for Maximal Update
			max_update = np.max( np.quantile(np.abs(shift_upd), 0.995, axis=0) )
			if max_update * binFactor < params['min_step_size']: break

		# Save the Final Reconstruction and Aligned Tilt Series
		if binFactor == 1: tomo.save_recon(params['filename'])
		self.sinogram_shifted = sinogram_shifted

		# Prepare outputs to be returned
		params['tilt_angle']     = params['tilt_angle'] + geom['tilt_angle']
		params['skewness_angle'] = params['tilt_angle'] + geom['skewness_angle']

		# vertical offset is a degree of freedom => minimize of offset 
		shift_total[:,1] = shift_total[:,1] - np.median(shift_total[:,1]) 

		optimal_shift[ang_order,:] = optimal_shift + shift_total * binFactor

		return (optimal_shift, params)

	def smooth(self, signal, window_len):
#		if signal.size < window_len:
#			raise ValueError, "Input vector needs to be bigger than window size."
		w = np.ones(window_len)
		return np.convolve(w/w.sum(),s,mode='valid')

	# plot results for the current iteration
	def plot_alignment(self, recSlice, sinoSlice, err, shift_upd, shift_total, angles, par, iter):

		# # Show Reconstruction and Sinogram Slice
		self.iItem1 = pg.ImageItem(sinoSlice)
		self.imPtr1.addItem(self.iItem1)

		self.iItem2 = pg.ImageItem(recSlice)
		self.imPtr2.addItem(self.iItem2)

		# # Show Evolution of Position Correction
		self.hShiftUpdCurve.setData(angles, shift_upd[:,0]*par['binning'])
		self.vShiftUpdCurve.setData(angles, shift_upd[:,1]*par['binning'])

		self.hShiftTotCurve.setData(angles, shift_total[:,0]*par['binning'])
		self.vShiftTotCurve.setData(angles, shift_total[:,1]*par['binning'])

		# # Show Evolution of Errors
		self.mseCurve.setData(np.arange(iter),np.mean(err[:iter,:],axis=1))
		self.errCurve.setData(angles, err[iter,:]) # Scatter Plot

		self.app.processEvents()

	def initialize_plot(self):

		self.app = QtGui.QApplication([])

		self.win = pg.GraphicsLayoutWidget(show=True)
		self.win.resize(1000,600)

		self.imPtr1 = self.win.addPlot(title='Filtered Shifted Sinogram')
		self.imPtr1.hideAxis('left')
		self.imPtr1.hideAxis('bottom')

		self.pPtr1 = self.win.addPlot(title='Current Position Update')
		self.pPtr1.setLabel('bottom','Angle [deg]')
		self.pPtr1.setLabel('left', 'Shift x Downsampling [px]')
		self.pPtr1.showGrid(x=True, y=True)
		self.hShiftUpdCurve = self.pPtr1.plot(pen='r', name="horiz")
		self.vShiftUpdCurve = self.pPtr1.plot(pen='b', name="vert")

		self.pPtr2 = self.win.addPlot(title='Total Position Update')
		self.pPtr2.setLabel('bottom','Angle [deg]')
		self.pPtr2.setLabel('left', 'Shift x Downsampling [px]')
		self.pPtr2.showGrid(x=True, y=True)
		self.hShiftTotCurve = self.pPtr2.plot(pen='r', name="horiz")
		self.vShiftTotCurve = self.pPtr2.plot(pen='b', name="vert")

		self.win.nextRow()

		self.imPtr2 = self.win.addPlot(title='Current Reconstruction')
		self.imPtr2.hideAxis('left')
		self.imPtr2.hideAxis('bottom')

		self.pPtr3 = self.win.addPlot(title='MSE Evolution') 
		self.pPtr3.setLabel('bottom','Iteration')
		self.pPtr3.setLabel('left', 'Mean Square Error')
		self.pPtr3.showGrid(x=True, y=True)
		self.mseCurve = self.pPtr3.plot()

		self.pPtr4 = self.win.addPlot(title='Current Error') # Scatter Plot
		self.pPtr4.setLabel('bottom','Angle [deg]')
		self.pPtr4.showGrid(x=True, y=True)
		self.errCurve = self.pPtr4.plot(pen='g')

	def removeImageItems(self):
		self.imPtr1.removeItem(self.iItem1)
		self.imPtr2.removeItem(self.iItem2)
