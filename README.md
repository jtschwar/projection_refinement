# Python Implementation of [Alignment methods for nanotomography with deep subpixel accuracy](https://opg.optica.org/oe/fulltext.cfm?uri=oe-27-25-36637)

A python implemenation of the projection matching method (PMA) developed by the Paul Scherrer Institute ([PSI](https://www.psi.ch/en/sls)). When I created this repository, there was no python implementation available, which limited the capability of aligning large datasets with supercomupting resources. Although there is an [official Matlab implementation](https://www.psi.ch/en/sls/csaxs/software), this is an available Python package useful as a backend to test and develop more advanced alignment algorithms. 

## Requirements
* python3.x
* numpy
* cupy (for GPU accelerated alignment)
* scipy
* pyqtgraph
* [tomo_TV](https://github.com/jtschwar/tomo_TV)
