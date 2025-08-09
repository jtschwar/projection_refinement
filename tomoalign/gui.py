from pyqtgraph.Qt import QtWidgets
import pyqtgraph as pg
import numpy as np

class visualize:
	"""
	Visualze the Projection Alignment Process  - plot results for the current iteration
	"""

	def __init__(self, parent_instance=None):
		self.parent = parent_instance        

	def plot_alignment(self, recSlice, sinoSlice, err, shift_upd, shift_total, angles, par, iter):

		# Show Reconstruction and Sinogram Slice
		self.iItem1 = pg.ImageItem(sinoSlice)
		self.imPtr1.addItem(self.iItem1)

		self.iItem2 = pg.ImageItem(recSlice)
		self.imPtr2.addItem(self.iItem2)

		# Show Evolution of Position Correction
		self.hShiftUpdCurve.setData(angles, shift_upd[:,0]*par['binning'])
		self.vShiftUpdCurve.setData(angles, shift_upd[:,1]*par['binning'])

		self.hShiftTotCurve.setData(angles, shift_total[:,0]*par['binning'])
		self.vShiftTotCurve.setData(angles, shift_total[:,1]*par['binning'])

		# Show Evolution of Errors
		self.mseCurve.setData(np.arange(iter),np.mean(err[:iter,:],axis=1))
		# Scatter Plot
		self.errCurve.setData(angles, err[iter,:])

		self.app.processEvents()

	def initialize_plot(self):

		self.app = QtWidgets.QApplication([])

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
