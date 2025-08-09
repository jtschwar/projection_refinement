import tomoalign, h5py, os

def load_demo():
    # Load the data
    tomoalign_path = os.path.dirname(tomoalign.__file__)
    data_file = os.path.join(tomoalign_path, "data", "misaligned_xray_dataset.h5")
    data = h5py.File(data_file, 'r')
    sinogram = data['tiltSeries'][:]
    theta = data['tiltAngles'][:]
    data.close()

    return sinogram, theta


