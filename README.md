# Python Implementation of [Alignment methods for nanotomography with deep subpixel accuracy](https://opg.optica.org/oe/fulltext.cfm?uri=oe-27-25-36637)

A python implemenation of the projection matching method (PMA) developed by the Paul Scherrer Institute ([PSI](https://www.psi.ch/en/sls)). When I created this repository, there was no python implementation available, which limited the capability of aligning large datasets with supercomupting resources. Although there is an [official Matlab implementation](https://www.psi.ch/en/sls/csaxs/software), this is an available Python package that is a useful platform to test and develop more advanced alignment algorithms. 

## Installation

To build the alignment package, first build the [tomo_TV reconstruction package](https://github.com/jtschwar/tomo_TV). Once the reconstruction package is available, we can build `tomoalign` with pip install:

```bash
pip install -e . 
```

## Quick Start

To run the alignment script, we simply need to provide the the tilt series and tilt angles from the experiment. 

```python
from tomoalign.aligner import AlignmentWorkflow
from tomoalign import load_demo

# Initialize the Alignment Class
(tiltSeries, tiltAngles) = load_demo()
aligner = AlignmentWorkflow(tiltSeries, tiltAngles)

# We can play with different reconstruction algorithms 
# (1) aligner.params['alg'] = 'sart'; aligner.params['initAlg'] = 'sequential'
# (2) aligner.params['alg'] = 'sirt'; aligner.params['initAlg'] = None
# (3) aligner.params['alg'] = 'wbp'; aligner.params['initAlg'] = 'ram-lak'

# Results is a dictionary with the aligned sinogram, measured shifts, and parameters metadata.
aligner.run(binning_factors=[4,2,1])

# Save the Results to the given h5 file name.
aligner.save('aligned.h5')
```

## References
If you use `tomoalign` for your research, we would appreciate it if you cite to the following papers:

- [Real-time 3D analysis during electron tomography using tomviz](https://www.nature.com/articles/s41467-022-32046-0)
- [Imaging 3D Chemistry at 1 nm resolution with fused multi-modal electron tomography](https://www.nature.com/articles/s41467-024-47558-0)

## Contact

email: [jtschwar@gmail.com](jtschwar@gmail.com)
website: [https://jtschwar.github.io](https://jtschwar.github.io)