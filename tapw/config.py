from dataclasses import dataclass, field
from typing import List, Dict, Optional
import os
import yaml
from pathlib import Path

@dataclass
class TwistConfig:
    """Configuration for twisted materials"""
    twist_index_m: int
    num_layers: int = 2
    type_structure: List[int] = field(default_factory=lambda: [2, 2])
    twist_layer: List[int] = field(default_factory=lambda: [1, 1])
    spin: bool = True

@dataclass
class PathConfig:
    """Configuration for file paths"""
    H_file: str
    S_file: str
    input_file: str
    output_dir: str
    kpath_in: str
    kpath_out: str

    def __post_init__(self):
        self.H_file = os.path.abspath(self.H_file)
        self.S_file = os.path.abspath(self.S_file)
        self.output_dir = os.path.abspath(self.output_dir)
        os.makedirs(self.output_dir, exist_ok=True)

@dataclass
class ClusterConfig:
    """Configuration for clustering parameters with fixed values"""
    layer_eps: float = 0.5
    layer_min_samples: int = 1
    sublayer_eps: float = 0.5
    sublayer_min_samples: int = 1
    atom_eps: float = 0.3
    atom_min_samples: int = 2
    period: float = 2 * 3.14159
    k_max: int = 2

    def __post_init__(self):
        # These parameters are fixed and should not be changed
        pass

@dataclass
class ComputeConfig:
    """Configuration for computation parameters"""
    valleys: List[int] = field(default_factory=lambda: [31, 32, 33])  # List of valleys to calculate
    mode: str = "band"  # Calculation mode: "band" for band structure, "chern" for Chern number
    efermi: float = -0.17
    n_g: int = 6
    num_processes: int = 50
    num_bands_cal: int = 50
    num_chern: int = 40
    gpu: bool = False
    gpu_index: List[int] = field(default_factory=lambda: [0, 1])
    delay_time: int = 4
    hamk_save: bool = False
    TAPW: bool = True
    eigsh_cal: bool = True
    C3_H: bool = False
    ge: bool = False
    eig_vec_cal: bool = True
    valley: int = field(init=False)  # Current valley being calculated
    valley_flag: str = field(init=False)  # Valley flag for display
    solve_flag: str = field(init=False)  # Solver flag
    eq_flag: str = field(init=False)  # Equation flag
    symm_flag: str = field(init=False)  # Symmetry flag
    gpu_num: int = field(init=False)  # Number of GPUs

    def __post_init__(self):
        # Valley mapping
        valley_flag = {
            1: "K1", 2: "K2", 
            11: "K1_120", 12: "K1_240", 
            5: "Gamma",
            31: "M1", 32: "M2", 33: "M3"
        }
        # Solver mapping
        solve_flag = {True: "eigsh", False: "lapack"}
        # Equation mapping
        eq_flag = {True: "ge", False: "st"}
        # Symmetry mapping
        symm_flag = {True: "symm", False: "nsymm"}

        # Set initial valley if not already set
        if not hasattr(self, 'valley') and self.valleys:
            self.valley = self.valleys[0]

        # Set flags
        self.valley_flag = valley_flag.get(self.valley, "unknown")
        self.solve_flag = solve_flag[self.eigsh_cal]
        self.eq_flag = eq_flag[self.ge]
        self.symm_flag = symm_flag[self.C3_H]
        
        # Set GPU number
        self.gpu_num = len(self.gpu_index) #if self.gpu else 0

        # Set GPU environment if using GPU
        if self.gpu:
            import os
            os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(idx) for idx in self.gpu_index)
            print(f"Using GPU with index {self.gpu_index}")

@dataclass
class Config:
    """Main configuration class"""
    twist: TwistConfig
    paths: PathConfig
    compute: ComputeConfig
    cluster: ClusterConfig = field(default_factory=ClusterConfig)  # Use default values if not provided

    @classmethod
    def from_yaml(cls, yaml_path: str) -> 'Config':
        """Load configuration from YAML file"""
        with open(yaml_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        twist_config = TwistConfig(**config_dict.get('twist', {}))
        paths_config = PathConfig(**config_dict.get('paths', {}))
        compute_config = ComputeConfig(**config_dict.get('compute', {}))
        # Use default cluster config if not provided
        cluster_config = ClusterConfig(**config_dict.get('cluster', {})) if 'cluster' in config_dict else ClusterConfig()
        config_obj = cls(
            twist=twist_config,
            paths=paths_config,
            compute=compute_config,
            cluster=cluster_config
        )
        # 如果config.yaml没有n_g字段，则自动调用update_ng
        if 'n_g' not in config_dict.get('compute', {}):
            config_obj.update_ng()
        return config_obj

    def save_yaml(self, yaml_path: str):
        """Save configuration to YAML file"""
        config_dict = {
            'twist': self.twist.__dict__,
            'paths': {k: v for k, v in self.paths.__dict__.items() if not k.startswith('_')},
            'compute': {k: v for k, v in self.compute.__dict__.items() if k != 'valley'}
            # Don't save cluster config as it uses fixed values
        }
        
        with open(yaml_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False)

    def update_ng(self):
        """Update n_g based on twist angle"""
        angle = float(self.twist.twist_angle)
        if angle > 9:
            self.compute.n_g = 3
        elif angle > 6:
            # self.compute.efermi = -0.17
            self.compute.n_g = 6
        elif angle > 4:
            # self.compute.efermi = -0.172
            self.compute.n_g = 7
        elif angle > 2.8:
            # self.compute.efermi = -0.178
            self.compute.n_g = 8
        elif angle > 2:
            self.compute.efermi = -0.18
            self.compute.n_g = 9
        elif angle > 1:
            self.compute.efermi = -0.18
            self.compute.n_g = 10 