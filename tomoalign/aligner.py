from pathlib import Path
import scipy.io as sio
import tomoalign
import numpy as np
import h5py
import os

class AlignmentWorkflow:
    """
    A class for performing tomographic alignment on X-ray datasets using consistency-based methods.
    
    This class encapsulates the alignment workflow including data loading, parameter configuration,
    multi-scale alignment processing, and result saving.
    """
    
    def __init__(self, tiltSeries, tiltAngles):
        """
        Initialize the TomoAlignmentProcessor.
        
        Parameters:
        -----------
        input_file : str or Path
            Path to input H5 file containing 'tiltSeries' and 'tiltAngles' datasets
        output_file : str or Path, optional
            Path for output file. If None, will use input filename with 'aligned_' prefix
        binning_factors : list, optional
            List of binning factors for multi-scale alignment. Default: [4, 2, 1]
        """

        # Validate data shapes
        self.sinogram = tiltSeries
        self.theta = tiltAngles
        Nx, Ny, Nangles = self.sinogram.shape
        assert Nangles == self.theta.shape[0], "Mismatch between sinogram angles and theta array"
        
        # Initialize data attributes
        self.shift = None
        self.tomo_align = None
       
        # Default parameters
        roi = np.array([0, Nx, 0, Ny])
        self.params = {
            'min_step_size': 0.01,
            'max_iter': 500,
            'use_TV': False,
            'high_pass_filter': 0.01,
            'step_relaxation': 0.5,
            'tilt_angle': 0,
            'momentum_acceleration': False,
            'apply_positivity': True,
            'refine_geometry': True,
            'lamino_angle': 90,
            'position_update_smoothing': False,
            'ROI': roi,
            'showsorted': True,
            'plot_results': True,
            'plot_results_every': 10,
            'use_gpu': True,
            'alg': 'sart',
            'initAlg': 'ram-lak',
            'filename': 'tmp.h5'
        }
        
    def run(self, binning_factors=[4,2,1]):
        """
        Perform the consistency-based alignment procedure.
        
        Returns:
        --------
        dict : Dictionary containing aligned sinogram, shifts, and parameters
        """

        # Display the current parameters
        print('Current Parameters:')
        for key, value in self.params.items():
            print(f'  {key}: {value}')

        # Update binning factors
        self.binning_factors = binning_factors
        
        # Initialize shift array
        self.shift = np.zeros([self.sinogram.shape[2], 2])
        
        # Initialize Alignment Class
        self.tomo_align = tomoalign.projection_refine.ProjectionMatcher(self.sinogram, self.theta)
        
        if self.params['plot_results']:
            self.params = self.tomo_align.initialize_plot(self.params)
            
        # Multi-scale alignment loop
        for curr_bin in self.binning_factors:
            print(f"Processing with binning factor: {curr_bin}")
            
            # Self-consistency based alignment procedure
            self.params['binning'] = curr_bin
            self.shift, self.params = self.tomo_align.tomo_consistency_linear(self.shift, self.params)
            self.tomo_align.sinogram = self.sinogram
            self.tomo_align.angles = self.theta
            
        print('Finished Alignments')
        
        return {
            'aligned_sinogram': self.tomo_align.sinogram_shifted,
            'shifts': self.shift,
            'parameters': self.params
        }
        
    def save_results(self, results=None):
        """
        Save aligned projections, shifts, and reconstruction parameters to H5 file.
        
        Parameters:
        -----------
        results : dict, optional
            Results dictionary from run_alignment(). If None, uses internal state.
        """
        if results is None:
            if self.tomo_align is None or self.shift is None:
                raise ValueError("No results to save. Run alignment first or provide results dictionary.")
            aligned_proj = self.tomo_align.sinogram_shifted
            shifts = self.shift
            params = self.params
        else:
            aligned_proj = results['aligned_sinogram']
            shifts = results['shifts']
            params = results['parameters']
            
        # Determine file mode
        save_h5flag = 'w' if self.output_file.exists() else 'a'
        if self.output_file.exists():
            print(f"{self.output_file} already exists, overwriting current H5 File")
            
        # Save data
        with h5py.File(self.output_file, save_h5flag) as f2:
            # Save parameters as attributes
            param_group = f2.create_group('params')
            for key, item in params.items():
                try:
                    param_group.attrs[key] = item
                except (TypeError, ValueError):
                    # Handle non-serializable parameters
                    param_group.attrs[key] = str(item)
                    
            # Save datasets
            f2.create_dataset('aligned_proj', data=aligned_proj)
            f2.create_dataset('aligned_shifts', data=shifts)
            
        print(f'Data saved to {self.output_file}')
        
    
