"""A setuptools based setup module.
See https://packaging.python.org/en/latest/distributing.html
Addapted from https://github.com/pypa/sampleproject
"""

# Always prefer setuptools over distutils
from codecs import open
from os import path
import subprocess
import sys

from setuptools import setup , find_packages

# To use a consistent encoding
here = path.abspath(path.dirname(__file__))

def detect_cuda_version():
    """Detect CUDA version and return appropriate CuPy package."""
    try:
        # Try to get CUDA version from nvidia-smi
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True, timeout=10)
        if result.returncode == 0 and 'CUDA Version:' in result.stdout:
            # Extract CUDA version from nvidia-smi output
            for line in result.stdout.split('\n'):
                if 'CUDA Version:' in line:
                    version_str = line.split('CUDA Version:')[1].strip().split()[0]
                    major_version = int(version_str.split('.')[0])
                    
                    if major_version >= 12:
                        print("Detected CUDA 12+, installing cupy-cuda12x")
                        return 'cupy-cuda12x>=12.0.0'
                    elif major_version == 11:
                        print("Detected CUDA 11, installing cupy-cuda11x")
                        return 'cupy-cuda11x>=12.0.0'
                    else:
                        print(f"Detected CUDA {major_version}, using generic cupy")
                        return 'cupy>=12.0.0'
    except (subprocess.TimeoutExpired, subprocess.CalledProcessError, FileNotFoundError, ValueError):
        pass
    
    try:
        # Alternative: try nvcc --version
        result = subprocess.run(['nvcc', '--version'], capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            for line in result.stdout.split('\n'):
                if 'release' in line and 'V' in line:
                    version_str = line.split('V')[1].strip().split(',')[0]
                    major_version = int(version_str.split('.')[0])
                    
                    if major_version >= 12:
                        print("Detected CUDA 12+ via nvcc, installing cupy-cuda12x")
                        return 'cupy-cuda12x>=12.0.0'
                    elif major_version == 11:
                        print("Detected CUDA 11 via nvcc, installing cupy-cuda11x")
                        return 'cupy-cuda11x>=12.0.0'
    except (subprocess.TimeoutExpired, subprocess.CalledProcessError, FileNotFoundError, ValueError):
        pass
    
    # Fallback: no CUDA detected
    print("WARNING: No CUDA installation detected!")
    print("GPU acceleration will not be available.")
    print("Install CuPy manually if you have CUDA:")
    print("  - For CUDA 11: pip install cupy-cuda11x")
    print("  - For CUDA 12: pip install cupy-cuda12x")
    print("  - Or use: pip install tomoalign[cuda11] or pip install tomoalign[cuda12]")
    return None

# Detect CUDA and set appropriate CuPy dependency
cuda_cupy_package = detect_cuda_version()

# Base dependencies (always required)
base_requires = ['numpy', 'scipy', 'matplotlib', 'h5py>=3', 'tqdm', 'scikit-image']

# Add CuPy if CUDA detected
install_requires = base_requires.copy()
if cuda_cupy_package:
    install_requires.append(cuda_cupy_package)

# Setup extras_require
extras_require = {
    'gui': ['PyQt5', 'pyqtgraph'],
}

# Add GPU options for manual installation
if not cuda_cupy_package:  # Only add these if auto-detection failed
    extras_require.update({
        'cuda11': ['cupy-cuda11x>=12.0.0'],
        'cuda12': ['cupy-cuda12x>=12.0.0']
    })

extras_require['all'] = list(set([item for sublist in extras_require.values() for item in sublist]))

setup(
    name='tomoalign',

    # Versions should comply with PEP440.  For a discussion on single-sourcing
    # the version across setup.py and the project code, see
    # https://packaging.python.org/en/latest/single_source_version.html
    version='0.1',

    description='Align Tomography Tilt Series',
    long_description='tomoalign Iterative alignment of tomogrpahic tilt series',

    # The project's main homepage.
    url='https://github.com/jtschwar/projection_refinement',

    # Author details
    author='J. Schwartz, P. Ercius',
    author_email='percius@lbl.gov',

    # Choose your license
    license='GPLv3+, MIT',

    # See https://pypi.python.org/pypi?%3Aaction=list_classifiers
    classifiers=[
        # How mature is this project? Common values are
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        'Development Status :: 3 -Alpha',

        # Indicate who your project is intended for
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering',

        # Pick your license as you wish (should match "license" above)
        'License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)',

        # Specify the Python versions you support here. In particular, ensure
        # that you indicate whether you support Python 2, Python 3 or both.
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
    ],

    # What does your project relate to?
    keywords='electron microscopy tomogrpahy alignment',

    # You can just specify the packages manually here if your project is
    # simple. Or you can use find_packages().
    packages=find_packages(exclude=['contrib', 'docs', 'test', 'data']),

    # Alternatively, if you want to distribute just a my_module.py, uncomment
    # this:
    #   py_modules=["my_module"],

    # List run-time dependencies here.  These will be installed by pip when
    # your project is installed. For an analysis of "install_requires" vs pip's
    # requirements files see:
    # https://packaging.python.org/en/latest/requirements.html

    install_requires=install_requires,

    # List additional groups of dependencies here (e.g. development
    # dependencies). You can install these using the following syntax,
    # for example:
    # $ pip install -e .[gui]
    extras_require=extras_require,

    # If there are data files included in your packages that need to be
    # installed, specify them here.  If using Python 2.6 or less, then these
    # have to be included in MANIFEST.in as well.
    # package_data={
    #     'edstomo': ['Elam/ElamDB12.txt'],
    # },
    #include_package_data=True,

    # Although 'package_data' is the preferred approach, in some case you may
    # need to place data files outside of your packages. See:
    # http://docs.python.org/3.4/distutils/setupscript.html#installing-additional-files # noqa
    # In this case, 'data_file' will be installed into '<sys.prefix>/my_data'
    # data_files=[('ncempy', ['ncempy/edstomo/Elam/ElamDB12.txt'])],
    
    # To provide executable scripts, use entry points in preference to the
    # "scripts" keyword. Entry points provide cross-platform support and allow
    # pip to create the appropriate form of executable for the target platform.
    entry_points={
       'console_scripts': [
           'data_viewer=tomoalign.view_data_3d:sliceZ',
       ],
    },
)