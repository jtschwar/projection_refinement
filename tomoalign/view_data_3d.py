from pyqtgraph.Qt import QtWidgets
import argparse, os, h5py
import pyqtgraph as pg
import numpy as np


def sliceZ():

	# Create the argument parser
	parser = argparse.ArgumentParser(description='Run tomoTV alignment process.')

	# Add arguments
	parser.add_argument('--file-path', type=str, required=True, help='Path to the input file')
	parser.add_argument('--group', type=str, required=True, help='Group to process')

	# Parse the arguments
	args = parser.parse_args()

	# Access arguments like args.file_path and args.group
	file_path = args.file_path
	group = args.group

	# Example of using the arguments
	print(f"Processing file at: {file_path} in group: {group}")

	# Check if file exists
	if not os.path.isfile(file_path):
		print(f"Error: The file at '{file_path}' does not exist.")
		return

	# Check if the group exists in the HDF5 file
	try:
		with h5py.File(file_path, 'r') as h5file:
			if group not in h5file:
				print(f"Error: The group '{group}' does not exist in the file '{file_path}'.")
				return
			else:
				data = h5file[group][:]
			
	except OSError as e:
		print(f"Error: Unable to open file '{file_path}'. Please check if it's a valid HDF5 file.")
		return

	## Always start by initializing Qt (only once per application)
	app = QtWidgets.QApplication([])

	# Create the ImageView widget for displaying the 3D data as 2D slices
	imv = pg.ImageView()

	# Display the window
	imv.show()
	imv.setWindowTitle(f'Viewing Slices of {file_path} - Group: {group}')

	# Display the 3D data as 2D slices (assuming the 3D data is in the shape (z, y, x))
	imv.setImage(data.transpose(2,0,1))  # Data should be ordered as (z, y, x) where z is the slice dimension

	# Set the range for the histogram (optional, for better contrast)
	imv.setHistogramRange(np.min(data), np.max(data))

	# Set the default slice to the middle
	imv.setCurrentIndex( int(data.shape[2] // 2))  # Set the default slice to the middle	

	# Start the Qt event loop
	app.exec_()
