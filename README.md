# README


Unstable code to perform simulations of RNA diffusion in the nucleus in the presence of chromatin of a particular architecture.
## Requirements & Installation for Phase-field package

### TLDR Recipe (details below)

`conda create --name <MYFIPYENV> --channel conda-forge python=3.12.8 numpy scipy matplotlib fipy=3.4.5 gmsh=4.13.1 python-gmsh=4.13.1 moviepy`

`activate <MYFIPYENV>`

### Running the code

`./scripts/diffusion.sh -i $infile -o $outfolder [-r $outsuffix] [-p $parameter_file] [-n $pnumber]`

$infile contains the text-file with the input parameters (see input/input_params.txt for reference) in the following format

`param,value \n`

$outfolder contains the path to output_folder

$outsuffix is an optional argument & it contains the suffix to the path to output_folder 

$parameter_file is an optional input file & it contains a list of parameters to iterate over in the following format:

`param`

`value1`

`value2`

$pnumber is an optional argument & it contains the index of the parameter value from \$parameter_file to be used


### Long version
This code is written to employ the latest version of fipy (3.4.5 ) interfacing with py 3.12.8.

The [FiPy webpage](https://www.ctcms.nist.gov/fipy/INSTALLATION.html) has instructions on setting up a specific conda environment with important packages from conda-forge, and installing latest fipy through pip.

_P.S. It should be actually faster with python2.7, but it's not supported yet.
