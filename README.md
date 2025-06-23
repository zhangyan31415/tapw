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
```yaml
# Twist parameters
twist:
  twist_index_m: 3          # Twist index m
  num_layers: 2             # Number of layers
  type_structure: [2, 2]    # Structure type for each layer
  twist_layer: [1, 1]       # Twist configuration
  spin: true               # Include spin

# File paths
paths:
  base_path: "path/to/your/data"
  input_file: "openmx.dat"
  output_dir: "output"
  kpath: "path/to/KPATH_GMKG.in"
  kpath_out: "path/to/KPATH_GMKG.out"

# Computation parameters
compute:
  valleys: [1,2]           # Valleys to calculate
  mode: "band"            # "band" or "chern"
  efermi: -0.17           # Fermi energy
  n_g: 3                  # Harmonic of G vectors
  num_processes: 61       # Number of parallel processes
  num_bands_cal: 50      # Number of bands
  gpu: false             # GPU acceleration
```

2. Run the calculation:
```bash
tapw-calc --config config.yaml
```

3. Plot the band structure:
```bash
tapw-plot --config plot_config.yaml
```

Or with command line arguments:

```bash
tapw-plot --kpath-in KPATH.in --kpath-out KPATH.out --bands band1.dat band2.dat --labels "Band 1" "Band 2" --fermi 0.0
```

## Configuration

See example configuration files in the `examples` directory. 