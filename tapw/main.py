#!/usr/bin/env python3
import argparse
import logging
import sys
import time
from pathlib import Path

from .config import Config
from .cal_ham_01 import BandStructureCalculator
from .read_pos_01 import LayeredLatticeAnalyzer, StructureProcessor,OpenMXFile
from .read_kpath_01 import KPathGenerator
from .read_hr_01 import HrSparseHandler

def setup_logging(log_file: str = None):
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_file) if log_file else logging.NullHandler()
        ]
    )
    return logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Twisted Material Band Structure Calculator')
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--twist-index', type=int,
                       help='Twist index (overrides config file)')
    parser.add_argument('--output-dir', type=str,
                       help='Output directory (overrides config file)')
    parser.add_argument('--valleys', type=int, nargs='+',
                       help='List of valleys to calculate (overrides config file)')
    parser.add_argument('--mode', choices=['band', 'chern'],
                       help='Calculation mode (overrides config file)')
    return parser.parse_args()

def main():
    """Main program"""
    # Parse arguments and load configuration
    args = parse_args()
    config = Config.from_yaml(args.config)
    
    # Override config with command line arguments
    if args.twist_index:
        config.twist.twist_index_m = args.twist_index
    if args.output_dir:
        config.paths.output_dir = args.output_dir
    if args.valleys:
        config.compute.valleys = args.valleys
    if args.mode:
        config.compute.mode = args.mode
    
    # Setup logging
    log_file = Path(config.paths.output_dir) / f"run_{time.strftime('%Y%m%d_%H%M%S')}.log"
    logger = setup_logging(str(log_file))
    logger.info("Starting calculation with configuration:")
    logger.info(f"Twist index: {config.twist.twist_index_m}")
    logger.info(f"Output directory: {config.paths.output_dir}")
    logger.info(f"Valleys to calculate: {config.compute.valleys}")
    logger.info(f"Calculation mode: {config.compute.mode}")

    try:
        # Initialize structure
        structure = OpenMXFile(
            file_path=str(Path(config.paths.base_path) / config.paths.input_file),
            twist_index=config.twist.twist_index_m,
            spin=config.twist.spin
        )
        structure.display_properties()

        # Initialize lattice analyzer
        analyzer = LayeredLatticeAnalyzer(
            input_data=structure.sorted_species_coordinates,
            num_layers=config.twist.num_layers,
            type_structure=config.twist.type_structure,
            twist_layer=config.twist.twist_layer
        )
        analyzer.process()

        # Plot results
        analyzer.plot_lattice(save=True, save_path=config.paths.output_dir)
        analyzer.plot_nearest_vectors_phase(save=True, save_path=config.paths.output_dir)

        # Process structure
        processor = StructureProcessor(
            input_data=analyzer.input_data,
            num_layers=config.twist.num_layers,
            monolayer_reciprocal_list=analyzer.reciprocal_vectors,
            twist_layer=config.twist.twist_layer,
            layer_eps=config.cluster.layer_eps,
            layer_min_samples=config.cluster.layer_min_samples,
            sublayer_eps=config.cluster.sublayer_eps,
            sublayer_min_samples=config.cluster.sublayer_min_samples,
            atom_eps=config.cluster.atom_eps,
            atom_min_samples=config.cluster.atom_min_samples,
            period=config.cluster.period,
            k_max=config.cluster.k_max,
            spin=structure.spin,
            twist_index=config.twist.twist_index_m,
            Tmat=structure.Tmat,
            reciprocal_Tmat=structure.reciprocal_Tmat,
        )
        processor.process()

        # Plot clustering results
        processor.plot_clusters_loc(save=True, save_path=config.paths.output_dir)
        processor.plot_clusters_phase(save=True, save_path=config.paths.output_dir)

        # Handle Hamiltonian
        H_handler = HrSparseHandler(
            file_name=config.paths.H_path,
            npz_file_name=str(Path(config.paths.base_path) / 'H.npz'),
            A=processor.transformed_index_matrix,
            read_from_npz=False
        )
        hr = H_handler.get_hr_sparse()

        # Handle overlap matrix
        S_handler = HrSparseHandler(
            file_name=config.paths.S_path,
            npz_file_name=str(Path(config.paths.base_path) / 'S.npz'),
            A=processor.transformed_index_matrix,
            read_from_npz=False
        )
        sr = S_handler.get_hr_sparse()

        # Initialize k-path if needed
        kpath_config = None
        if config.compute.mode == "band":
            kpath_config = KPathGenerator(structure.Tmat)
            kpath_config.read_and_generate_kpath(config.paths.kpath, config.paths.kpath_out)

        # Calculate for each valley
        for valley in config.compute.valleys:
            logger.info(f"Starting calculation for valley {valley}")
            config.compute.valley = valley
            
            band_calculator = BandStructureCalculator(
                hr_supercell=hr,
                sr_supercell=sr,
                structure=processor,
                config=config.compute,
                kpath_config=kpath_config
            )
            
            out_path = Path(config.paths.output_dir) / f"Q_shell_{config.compute.n_g}"
            out_path.mkdir(exist_ok=True)
            
            band_calculator.run_calculation(str(out_path))
            logger.info(f"Completed calculation for valley {valley}")

    except Exception as e:
        logger.error(f"Error during calculation: {str(e)}", exc_info=True)
        sys.exit(1)

    logger.info("Calculation completed successfully")

if __name__ == "__main__":
    main() 