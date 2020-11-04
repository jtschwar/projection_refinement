from pyqtgraph.Qt import QtCore, QtGui
import pyqtgraph as pg
import numpy as np
import time


def sliceZ(data):

	## Always start by initializing Qt (only once per application)
	app = QtGui.QApplication([])

	## Create window with ImageView widget
	win = QtGui.QMainWindow()
	win.resize(800,800)
	win.setWindowTitle('Sinogram, min: {:+.2f} & max: {:+.2f}'.format(np.min(data), np.max(data)))

	cw = QtGui.QWidget()
	win.setCentralWidget(cw)
	l = QtGui.QGridLayout()
	cw.setLayout(l)

	imv1 = pg.ImageView()
	l.addWidget(imv1, 0, 0)
	win.show()

	# Display Data
	imv1.setImage(data.transpose(2,0,1))
	imv1.setHistogramRange(np.min(data), np.max(data))

	# Add ROI. 
	# viewROI = pg.ROI([roi[0],roi[2]], [roi[1],roi[3]])
	# viewROI.addScaleHandle([0.5, 0], [0.5, 0.5])
	# viewROI.addScaleHandle([0, 0.5], [0.5, 0.5])
	# imv1.addItem(viewROI)

	# Start Qt event loop
	app.exec_()	


