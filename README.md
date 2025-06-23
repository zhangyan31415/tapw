# TAPW (Twisted Angle Plane Wave)

A Python package for calculating and plotting band structures of twisted materials.

## Features

- Band structure calculation for twisted bilayer systems
- Support for various valleys (K1, K2, Gamma, M points)
- C3 symmetry consideration
- GPU acceleration support
- Parallel computation capabilities

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/tapw.git
cd tapw
```

2. Create a conda environment and install dependencies:
```bash
conda env create -f environment.yml
conda activate tapw
```

3. Install the package in development mode:
```bash
pip install -e .
```

## Usage

1. Prepare your configuration file (config.yaml):
```bash
tapw-config -o output_dir
```
and then you can edit the `config.yaml` and `bands.yaml` files in the output_dir, especially the `config.yaml` file, you must set the `H.dat` ,`S.dat` and `openmx.dat` files path.

2. Run the calculation:
```bash
cd output_dir
tapw-calc --config config.yaml
```

3. Plot the band structure:
```bash
cd output_dir/Q_shell_{n_g}/band_data
tapw-plot --config ../../bands.yaml
```

Or with command line arguments:

```bash
tapw-plot --kpath-in KPATH.in --kpath-out KPATH.out --bands band1.dat band2.dat --labels "Band 1" "Band 2" --fermi 0.0
```
4. Calculate the Chern number:
```bash
cd output_dir
tapw-calc --config config.yaml --mode chern --n_g 4 --num_processes 100 --num_chern 20
#or you can directly edit the config.yaml file
#and then run the following command
tapw-calc --config config.yaml
cd output_dir/Q_shell_{n_g}
tapw-chernpost --config config.yaml -b -1,-2 -v 1 > tapw_chern.log
```


## Configuration

See example configuration files in the `examples` directory. 