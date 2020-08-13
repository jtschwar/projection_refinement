from scipy import signal
import numpy as np
import shifts
import tqdm

class angle_refine:

	def __init__(input_sino, input_angles, params):
		self.sinogram = input_sino
		self.angles = input_angles
		self.params = params
	
	def main_loop():
		print('TBD')

	def tomo_consistency_linear():
		print('[align] : Starting align_tomo_consistency_linear')
		Nangles = angles.shape[0]
		(Nlayers,width_sinogram) = input_sino.shape[:1]

		binFactor = self.params['binning']
		print('[align] : Shifting Sinograms and binning = ' + str(binFactor))

		# Shift to the Last Optimal Position + Remove Edge Issues (Line: 235)
		shifts.imshift_generic(self.sinogram, self.shifts, 5, self.ROI, binFactor)

		# Sort by angle if Requested, but only after binning to make it faster

		# Initialize ASTRA FBP

		# Prepare some Auxiliary Variables
		win = signal.tukeywin(width_sinogram,0.2)
		win = signal.tukeywin(Nlayers,0.2)*win

		# Main Loop
		for ii in tqdm(range(params['max_iter'])):

			if params['verbose']:
				print('[align] : Iteration %d/%d'.format(ii, params['max_iter']) )

			sinogram_shifted = self.sinogram_shifted


	def default_param_values():
		self.params['align_vertical'] = True
		self.params['align_horizontal'] = True
		self.params['high_pass_filter'] = 0.02
		self.params['min_step_size'] = 0.01
		self.params['max_iter'] = 50
		self.params['verbose'] = True
		self.params['refine_geometry'] = True
		self.params['momentum_acceleration'] = True
		self.params['use_TV'] = True
		self.params['filter_type'] = 'ram-lak'
		self.params['binning'] = 1
