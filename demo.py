from tomoalign.aligner import AlignmentWorkflow
from pathlib import Path
import h5py

# Load the data
data = h5py.File(Path('misaligned_xray_dataset.h5'), 'r')
sinogram = data['tiltSeries'][:]
theta = data['tiltAngles'][:]
data.close()

# Initialize the Alignment Class
aligner = AlignmentWorkflow(sinogram, theta)
aligner.run()
