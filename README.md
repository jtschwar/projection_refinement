# Python Implementation of [Alignment methods for nanotomography with deep subpixel accuracy](https://opg.optica.org/oe/fulltext.cfm?uri=oe-27-25-36637)

A python implemenation of the projection matching method (PMA) developed by the Paul Scherrer Institute ([PSI](https://www.psi.ch/en/sls)). When I created this repository, there was no python implementation available, which limited the capability of aligning large datasets with supercomupting resources. Although there is an [official Matlab implementation](https://www.psi.ch/en/sls/csaxs/software), this is an available Python package that is a useful platform to test and develop more advanced alignment algorithms. 

## Requirements
* [tomo_TV](https://github.com/jtschwar/tomo_TV)
* cupy (for GPU accelerated alignment)
* pyqtgraph
* python3.x
* numpy
* scipy
* tqdm
* h5py

Please build all these necessary dependencies, prior to running these scripts. I'd recommend adding tomo_TV to the Python path. Refer to `setup_tomo_tv.sh` for an example shell script. 

## Contact

email: [jtschw@umich.edu](jtschw@umich.edu)
website: [https://jtschwar.github.io](https://jtschwar.github.io)