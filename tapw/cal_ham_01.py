import numpy as np
import scipy
from scipy.linalg import lapack
import scipy.linalg
import scipy.sparse
from scipy.sparse.linalg import eigsh
from scipy.linalg import det
import time
import os
import sys
from joblib import Parallel, delayed
from .C3_symm_01 import C3_MoTe2_all,C3_G_matrix
from tqdm import tqdm
from functools import wraps
# import cupy as cp
from .config import ComputeConfig
from .read_pos_01 import StructureProcessor
from .read_kpath_01 import KPathGenerator
import psutil
import memory_profiler
from datetime import datetime
# import cupy as cp
from numba import njit
import re
from numpy.linalg import LinAlgError, eigh
try:
    import cupy as cp
except ImportError:
    pass
hartree = 27.211386245988
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
def timing_decorator_factory_1(process_id):
    def timing_decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if process_id == 0:
                start_time = time.time()
                mem_before = memory_profiler.memory_usage()[0]

                result = func(*args, **kwargs)

                end_time = time.time()
                mem_after = memory_profiler.memory_usage()[0]

                duration = end_time - start_time
                mem_usage_mib = mem_after - mem_before
                mem_usage_mb = mem_usage_mib * 1024 * 1024 # Convert from MiB to GB

                print(f"Function '{func.__name__}' executed in {duration:.6f} seconds and used {mem_usage_mb:.6f} GB of memory")
            else:
                result = func(*args, **kwargs)

            return result
        return wrapper
    return timing_decorator

def timing_decorator_factory(process_id):
    def timing_decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if process_id == 0:
                start_time = time.time()
                process = psutil.Process()
                mem_before = process.memory_info().rss / (1024 * 1024 * 1024)  # Convert to GB

                result = func(*args, **kwargs)

                mem_after = process.memory_info().rss / (1024 * 1024 * 1024)  # Convert to GB
                end_time = time.time()
                duration = end_time - start_time
                mem_peak = mem_after - mem_before

                current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                print(f"[{current_time}] Function '{func.__name__}' executed in {duration:.6f} seconds, Memory peak: {mem_peak:.6f} GB")
                sys.stdout.flush() 
            else:
                result = func(*args, **kwargs)
            return result
        return wrapper
    return timing_decorator

class Config_old:
    def __init__(self, 
                 efermi=-0.16617994221559068, TAPW=True, eigsh_cal=True, valley=1,
                 C3_H=False, ge=False, n_g=4, num_processes=2, eig_vec_cal=True,
                 num_bands_cal=30, end_flag='GMKG_no_phase',gpu=False,hamk_save=False,gpu_index=[],delay_time=0):

        self.efermi = efermi
        self.TAPW = TAPW
        self.eigsh_cal = eigsh_cal
        self.valley = valley
        self.C3_H = C3_H
        self.ge = ge
        self.n_g = n_g
        self.num_processes = num_processes
        self.eig_vec_cal = eig_vec_cal
        self.num_bands_cal = num_bands_cal
        self.end_flag = end_flag
        self.gpu = gpu
        self.hamk_save = hamk_save
        self.gpu_index = np.array(gpu_index)
        self.gpu_num = len(gpu_index)
        self.delay_time = delay_time







        if self.gpu:
            # import cupy as cp
            os.environ["CUDA_VISIBLE_DEVICES"] = ",".join([str(index) for index in self.gpu_index])
            print(f"Using GPU with index {self.gpu_index}")


    @classmethod
    def default(cls):
        return cls()

    @classmethod
    def from_dict(cls, config_dict):
        return cls(**config_dict)
    
    def update(self, **kwargs):
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise AttributeError(f"{key} is not a valid attribute of {self.__class__.__name__}")
        valley_flag = {1: "K1", 2: "K2", 11: "K1_120", 12: "K1_240", 5: "Gamma",
                       31: "M1", 32: "M2", 33: "M3"}
        solve_flag = {True: "eigsh", False: "lapack"}
        eq_flag = {True: "ge", False: "st"}
        symm_flag = {True: "symm", False: "nsymm"}
        self.valley_flag = valley_flag[self.valley]
        self.solve_flag = solve_flag[self.eigsh_cal]
        self.eq_flag = eq_flag[self.ge]
        self.symm_flag = symm_flag[self.C3_H]

class TAPW_parameters:
    def __init__(self, structure:StructureProcessor, config:ComputeConfig):

        self.config = config
        self.n_g = self.config.n_g
        self.valley = self.config.valley
        self.structure = structure

        self.g_matrix = None
        self.C3_matrix = None
        self.symm_matrix = None
        self.symm_matrix_inv = None
        self.g_symm_matrix = None
        self.g_symm_matrix_inv = None

        self.K1 = None
        self.K2 = None
        self.g_vec_list_K1 = None
        self.g_vec_list_K2 = None
        

    def calculate_K_points(self):
        """Calculate the K1 and K2 points."""
        m_g_unitvec_1 = self.structure.reciprocal_Tmat[0][:2]
        m_g_unitvec_2 = self.structure.reciprocal_Tmat[1][:2]
        if self.structure.reciprocal_Tmat[0]@self.structure.reciprocal_Tmat[1] < 0:
            m_g_unitvec_1 = - self.structure.reciprocal_Tmat[0][:2]
        else:
            m_g_unitvec_1 = self.structure.reciprocal_Tmat[0][:2]
        n_moire = self.structure.twist_index
        offset = -n_moire*m_g_unitvec_1+n_moire*m_g_unitvec_2

        m_K1 = -1/3*m_g_unitvec_1 + 2/3*m_g_unitvec_2
        m_K2 = -2/3*m_g_unitvec_1 + 1/3*m_g_unitvec_2
        
        # m_M1 = 1/2*m_g_unitvec_1
        # m_M2 = 1/2*m_g_unitvec_2
        # m_M3 = self.rotate_vector(m_M2,120)
        if n_moire % 2 == 1:
            offset_1 = (n_moire+1)*(m_g_unitvec_1+m_g_unitvec_2)/2
            offset_2 = self.rotate_vector(offset_1,120)
            offset_3 = self.rotate_vector(offset_1,240)
            m_M1 = -1/2*m_g_unitvec_2
            m_M2 = -1/2*m_g_unitvec_1
            m_M3 = self.rotate_vector(m_M2,120)
        else:
            offset_1 = n_moire*(m_g_unitvec_1+m_g_unitvec_2)/2
            offset_2 = self.rotate_vector(offset_1,120)
            offset_3 = self.rotate_vector(offset_1,240)
            m_M1 = 1/2*m_g_unitvec_1
            m_M2 = 1/2*m_g_unitvec_2
            m_M3 = self.rotate_vector(m_M2,120)

        mono_basis_1layer = self.structure.monolayer_reciprocal_list[0]
        mono_basis_2layer = self.structure.monolayer_reciprocal_list[1]

        layer_basis = np.array([self.structure.monolayer_reciprocal_list[i] for i in range(len(self.structure.monolayer_reciprocal_list))])
        print("layer_basis = ",layer_basis)

        valley_dict = {1: (m_K1 + offset, m_K2 + offset),
                    2: (-m_K1 - offset, -m_K2 - offset),
                    11: (self.rotate_vector(m_K1 + offset, 120), self.rotate_vector(m_K2 + offset, 120)),
                    12: (self.rotate_vector(m_K1 + offset, 240), self.rotate_vector(m_K2 + offset, 240)),
                    5: (np.zeros(2), np.zeros(2)),
                    31: (m_M1 + offset_1, m_M2 + offset_1),
                    32: (self.rotate_vector(m_M1 + offset_1, 120), self.rotate_vector(m_M2 + offset_1, 120)),
                    33: (self.rotate_vector(m_M1 + offset_1, 240), self.rotate_vector(m_M2 + offset_1, 240)),
                    }
        offset_dict = {1:offset,
                       2:-offset,
                       11:self.rotate_vector(offset,120),
                       12:self.rotate_vector(offset,240),
                       5:np.zeros(2),
                       31:offset_1,32:self.rotate_vector(offset_1,120),33:self.rotate_vector(offset_1,240)}
        mK_dict = {1:(m_K1,m_K2),
                   2:(-m_K1,-m_K2),
                   11:(self.rotate_vector(m_K1,120),self.rotate_vector(m_K2,120)),
                   12:(self.rotate_vector(m_K1,240),self.rotate_vector(m_K2,240)),
                   5:(np.zeros(2),np.zeros(2)),
                   31:(m_M1,m_M2),
                   32:(self.rotate_vector(m_M1,120),self.rotate_vector(m_M2,120)),
                   33:(self.rotate_vector(m_M1,240),self.rotate_vector(m_M2,240))}

        if self.valley not in mK_dict:
            raise ValueError("Valley error in set_const_mtrx_diff_Gn")

        K1, K2 = valley_dict[self.valley]
        m_K1,m_K2 = mK_dict[self.valley]
        offset = offset_dict[self.valley]
        print("K1 ,K2,m_K1,m_K2,offset = ",K1, K2, m_K1, m_K2, offset)
        return K1, K2, m_K1, m_K2, offset
    
    def unique_sorted(self, arr, tolerance=0.02):
        """Return a sorted array with unique elements within a specified tolerance."""
        sorted_arr = np.sort(arr)
        unique_list = []

        if len(sorted_arr) > 0:
            unique_list.append(sorted_arr[0])

        for element in sorted_arr:
            if abs(element - unique_list[-1]) > tolerance:
                unique_list.append(element)

        return unique_list

    def generate_g_vec_list(self):

        def find_common_elements(arr1, arr2, arr3, tol=0.01):
            common_elements = []
            common_indices_arr1 = []
            common_indices_arr2 = []
            common_indices_arr3 = []
            
            for idx, coord1 in enumerate(arr1):
                for idx2, coord2 in enumerate(arr2):
                    for idx3, coord3 in enumerate(arr3):
                        if np.allclose(coord1, coord2, atol=tol) and np.allclose(coord1, coord3, atol=tol):
                            common_elements.append(coord1)
                            common_indices_arr1.append(idx)
                            common_indices_arr2.append(idx2)
                            common_indices_arr3.append(idx3)
            
            return np.array(common_elements), np.array(common_indices_arr1), np.array(common_indices_arr2), np.array(common_indices_arr3)

        def get_C3_g_list_K_valley(g_vec_list,K1,K2):

            g_vec_list_K1_C0 = g_vec_list
            g_vec_list_K1_C1 = (rot(g_vec_list.T-K1[:,np.newaxis],120)+K1[:,np.newaxis]).T
            g_vec_list_K1_C2 = (rot(g_vec_list.T-K1[:,np.newaxis],240)+K1[:,np.newaxis]).T
            g_vec_list_K2_C0 = g_vec_list
            g_vec_list_K2_C1 = (rot(g_vec_list.T-K2[:,np.newaxis],120)+K2[:,np.newaxis]).T
            g_vec_list_K2_C2 = (rot(g_vec_list.T-K2[:,np.newaxis],240)+K2[:,np.newaxis]).T
            g_vec_list_K1 ,index_K1_C0,index_K1_C1,index_K1_C2 = find_common_elements(g_vec_list_K1_C0,g_vec_list_K1_C1,g_vec_list_K1_C2)
            g_vec_list_K2 ,index_K2_C0,index_K2_C1,index_K2_C2 = find_common_elements(g_vec_list_K2_C0,g_vec_list_K2_C1,g_vec_list_K2_C2)
            return g_vec_list_K1,g_vec_list_K2,index_K1_C0,index_K2_C0
        


        def rot(vec,theta):
            theta = np.pi/180*theta
            rot_mat = np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]])
            return np.dot(rot_mat,vec)


        """Generate the g_vec_list for K1 and K2."""
        m_g_unitvec_1 = -self.structure.reciprocal_Tmat[0][:2]
        m_g_unitvec_2 = self.structure.reciprocal_Tmat[1][:2]
        if self.structure.reciprocal_Tmat[0]@self.structure.reciprocal_Tmat[1] < 0:
            m_g_unitvec_1 = self.structure.reciprocal_Tmat[0][:2]
        else:
            m_g_unitvec_1 = -self.structure.reciprocal_Tmat[0][:2]

        K1, K2, m_K1, m_K2, offset = self.calculate_K_points()
        self.K1, self.K2 = K1, K2

        g_vec_list = []
        g_3 = -m_g_unitvec_1 - m_g_unitvec_2
        num_add = 2
        for i in range(self.n_g + num_add):
            for j in range(self.n_g + num_add):
                g_vec_list.append(i * m_g_unitvec_1 + j * m_g_unitvec_2)

        for i in range(1, self.n_g + num_add):
            for j in range(self.n_g + num_add):
                g_vec_list.append(i * g_3 + j * m_g_unitvec_1)

        for i in range(1, self.n_g + num_add):
            for j in range(1, self.n_g + num_add):
                g_vec_list.append(j * g_3 + i * m_g_unitvec_2)

        o_g_vec_list = np.array(g_vec_list)

        o_g_vec_list_m_K1 = o_g_vec_list - m_K1
        o_g_vec_list_m_K2 = o_g_vec_list - m_K2

        # print("o_g_vec_list_m_K1 = ", o_g_vec_list_m_K1)


        K1_distance = self.unique_sorted(np.linalg.norm(o_g_vec_list_m_K1, axis=1), tolerance=0.1 * np.linalg.norm(m_g_unitvec_1))
        K2_distance = self.unique_sorted(np.linalg.norm(o_g_vec_list_m_K2, axis=1), tolerance=0.1 * np.linalg.norm(m_g_unitvec_1))

        # print("K1_distance = ", K1_distance)
        # print("K2_distance = ", K2_distance)

        #ori ========================================
        g_vec_list_K1 = []
        g_vec_list_K2 = []
        if self.valley == 5:
            n_g = self.n_g - 1
        else:
            n_g = self.n_g
        for i, vec in enumerate(o_g_vec_list_m_K1):
            if np.linalg.norm(vec) < K1_distance[n_g - 1] + 0.001:
                g_vec_list_K1.append(o_g_vec_list[i] + offset)
        for i, vec in enumerate(o_g_vec_list_m_K2):
            if np.linalg.norm(vec) < K2_distance[n_g - 1] + 0.001:
                g_vec_list_K2.append(o_g_vec_list[i] + offset)
        
        #new outside ===================================
        # o_g_vec_list_new = []
        # for i, vec in enumerate(o_g_vec_list_m_K1):
        #     if np.linalg.norm(vec) > K1_distance[8] + 0.001:
        #         o_g_vec_list_new.append(o_g_vec_list[i])
        # o_g_vec_list_new = np.array(o_g_vec_list_new)

        # g_vec_list_K1 = o_g_vec_list + offset
        # g_vec_list_K2 = o_g_vec_list + offset
        # g_vec_list_K1 = o_g_vec_list_new + offset
        # g_vec_list_K2 = o_g_vec_list_new + offset
        # ===============================================

        # just C3
        # ============================================================
        # g_vec_list_K1,g_vec_list_K1,_,_ = get_C3_g_list_K_valley(o_g_vec_list+offset,K1,K2)

        self.g_vec_list_K1 = np.array(g_vec_list_K1)
        self.g_vec_list_K2 = np.array(g_vec_list_K2)
        # self.g_vec_list_K1 = np.load("/work/zy/software/TAPW_tmdc/2.13/new_10step/2soc_200/g_vec_list_5_K1_1layer.npy")
        # self.g_vec_list_K2 = np.load("/work/zy/software/TAPW_tmdc/2.13/new_10step/2soc_200/g_vec_list_5_K1_2layer.npy")
         
        print(self.g_vec_list_K1)
        print("======================")
        print(self.g_vec_list_K2)
        print("num G vectors per layer = ",len(self.g_vec_list_K1))
    
    @timing_decorator_factory(process_id=0)
    def generate_gr_matrix(self):
        # if not self.config.gpu:
        g_vec_list = np.array([self.g_vec_list_K1, self.g_vec_list_K2])
        self.g_matrix = self.generate_gr_matrix_cpu(self.structure.df, g_vec_list, spin=self.structure.spin)
        self.g_matrix_conj = self.g_matrix.T.conj()
        # else:
        #     self.generate_gr_matrix_gpu()
        
        
        delta = np.abs(det(self.g_matrix @ self.g_matrix_conj))
        self.g_matrix = scipy.sparse.csr_matrix(self.g_matrix)
        self.g_matrix_conj = scipy.sparse.csr_matrix(self.g_matrix_conj)
        if delta < 1.0e-2:
            raise Exception("np.abs(det(gr_mtrx @ gr_mtrx.T.conj())) < 1.0e-2")
        else:
            print("np.abs(det(gr_mtrx @ gr_mtrx.T.conj())) = ", delta)
        
    @staticmethod
    def generate_gr_matrix_cpu(structure_df, g_vec_list, spin=None):
        """
        生成并返回转换后的 gr_matrix 及其共轭转置 gr_matrix_conj。
        
        参数:
        - structure_df (pd.DataFrame): 包含结构信息的 DataFrame, 必须包含以下列:
            - 'atom_type': 原子类型
            - 'orb_num': 每个原子的轨道数量
            - 'layer': 每个原子的层级索引
            - 'shifted_x', 'shifted_y': 原子的平移坐标
        - g_vec_list (np.ndarray): 一个二维数组，形状为 (num_layers, num_g_per_layer, dim)，代表每层的 g 向量列表
        
        返回:
        - gr_matrix (np.ndarray): 生成的 gr_matrix, 形状为 (dim_gr_1, dim_gr_2)
        - gr_matrix_conj (np.ndarray): gr_matrix 的共轭转置，形状为 (dim_gr_2, dim_gr_1)
        """
        
        # 复制 DataFrame 以避免修改原始数据
        df_temp = structure_df.copy()
        
        # 获取唯一的原子类型
        atom_type_list = np.unique(df_temp['atom_type'].values)
        
        # 获取每种原子的轨道数量（假设每种原子的轨道数量相同）
        atom_orb_num_list = np.array([
            np.unique(df_temp[df_temp['atom_type'] == atom_type]['orb_num'].values)[0]
            for atom_type in atom_type_list
        ])
        
        # 获取所有原子的轨道数量列表
        atom_orb_num_list_all_atom = np.concatenate([
            df_temp[df_temp['atom_type'] == atom_type]['orb_num'].values
            for atom_type in atom_type_list
        ]).flatten()
        
        # 获取每种原子的数量
        atom_num_list = np.array([
            len(df_temp[df_temp['atom_type'] == atom_type])
            for atom_type in atom_type_list
        ])
        
        # 获取每种原子的层级索引（假设每种原子的层级相同）
        atom_layer_list = np.array([
            np.unique(df_temp[df_temp['atom_type'] == atom_type]['layer'].values)[0]
            for atom_type in atom_type_list
        ])
        
        # 获取所有原子的层级索引列表
        atom_layer_list_all_atom = np.concatenate([
            df_temp[df_temp['atom_type'] == atom_type]['layer'].values
            for atom_type in atom_type_list
        ]).flatten()
        
        # 获取所有原子的原子类型列表
        atom_type_list_all_atom = df_temp['atom_type'].values
        
        atom_orb_name_list = np.array([
            np.unique(df_temp[df_temp['atom_type'] == atom_type]['orb_name'].values)[0]
            for atom_type in atom_type_list
        ])
        
        # 计算每种原子的因子
        factor_list = np.array([
            1 / np.sqrt(atom_num)
            for atom_num in atom_num_list
        ])
        
        # 打印相关信息
        print("atom_type_list = ", atom_type_list)
        print("atom_orb_num_list = ", atom_orb_num_list)
        print("atom_num_list = ", atom_num_list)
        print("atom_layer_list = ", atom_layer_list)
        print("atom_orb_name_list = ", atom_orb_name_list)
        print("atom_type_list_all_atom = ", atom_type_list_all_atom)
        print("atom_orb_num_list_all_atom = ", atom_orb_num_list_all_atom)
        
        # 获取 g_vec_list 的维度信息
        num_layers = g_vec_list.shape[0]
        num_g_per_layer = g_vec_list.shape[1]
        dim_gr_1 = np.sum(atom_orb_num_list[atom_layer_list == i].sum() * len(g_vec_list[i]) for i in range(np.max(atom_layer_list) + 1))
        dim_gr_2 = np.sum(atom_num_list * atom_orb_num_list)
        
        print("dim_gr_1 = ", dim_gr_1)
        print("dim_gr_2 = ", dim_gr_2)
        
        # 初始化 gr_matrix
        gr_matrix = np.zeros((dim_gr_1, dim_gr_2), dtype=np.complex128)
        
        # 生成 index_list_g
        index_list_g = []
        g_list_index = 0
        orb_layer_num = np.bincount(atom_layer_list, weights=atom_orb_num_list).astype(int)
        g_layern_num = np.array([g_vec_list.shape[1]] * g_vec_list.shape[0])
        
        for ilayer in range(np.max(atom_layer_list) + 1):
            for i in range(len(g_vec_list[ilayer])):
                for j, atom_orb_num in enumerate(atom_orb_num_list):
                    if atom_layer_list[j] != ilayer:
                        continue
                    g_list_index += 1
                    shift1 = atom_orb_num_list[:j][atom_layer_list[:j] == ilayer].sum()
                    shift2 = i * atom_orb_num_list[atom_layer_list == ilayer].sum()
                    shift3 = np.dot(g_layern_num[:ilayer], orb_layer_num[:ilayer])
                    index_list = np.arange(atom_orb_num) + shift1 + shift2 + shift3
                    index_list_g.append(index_list)
                    # print("ilayer = ", ilayer, "i = ", i, "j = ", j, "atom_orb_num = ", atom_orb_num, "len(index_list) = ", len(index_list))
        
        # index_list_g = np.array(index_list_g) # cant be array as the length of each element may be different due to different atom_orb_num
        
        # 获取原子的位置信息
        pos_array = np.array(df_temp[['shifted_x', 'shifted_y']].values)
        
        # 分配 gr_matrix 的对角元素
        g_list_index = -1
        for ilayer in range(np.max(atom_layer_list) + 1):
            for i in range(len(g_vec_list[ilayer])):
                for j, atom_orb_num in enumerate(atom_orb_num_list):
                    if atom_layer_list[j] != ilayer:
                        continue
                    g_list_index += 1
                    for iatom in range(len(atom_type_list_all_atom)):
                        if (atom_layer_list_all_atom[iatom] != ilayer or
                            atom_type_list_all_atom[iatom] != j):
                            continue
                        r_vec = pos_array[iatom]
                        g_vec = g_vec_list[ilayer][i]
                        gr_i_index_include = index_list_g[g_list_index]
                        gr_j_start = np.sum(atom_orb_num_list_all_atom[:iatom])
                        gr_j_end = gr_j_start + atom_orb_num_list_all_atom[iatom]
                        gr_j_index_include = np.arange(gr_j_start, gr_j_end)
                        
                        # 计算指数值
                        exp_val = np.exp(-1j * np.dot(g_vec, r_vec))
                        
                        # 只赋值对角部分
                        # print("g_list_index = ", g_list_index, "iatom = ", iatom, gr_i_index_include, gr_j_index_include,g_vec, r_vec)
                        gr_matrix[gr_i_index_include, gr_j_index_include] = [exp_val] * atom_orb_num_list_all_atom[iatom]
        
        # 应用因子
        print("factor_list = ", factor_list)
        gr_matrix = gr_matrix * factor_list[0]
        
        # 计算共轭转置
        gr_matrix_conj = gr_matrix.T.conj()
        
        # 打印 gr_matrix @ gr_matrix_conj 的实部
        print(np.real(gr_matrix @ gr_matrix_conj))
        print(np.linalg.det(gr_matrix @ gr_matrix_conj))
        
        return scipy.linalg.block_diag(gr_matrix, gr_matrix) if spin else gr_matrix

    def generate_gr_matrix_gpu(self):
        """Generate the g_matrix for TAPW using GPU."""
        n_wann_perlayer = int(len(self.structure.sort_wann) / 2)
        n_wann = n_wann_perlayer * 2

        num_orbs = self.structure.num_orbs_per_unit_cell * 2
        sel_orbs = np.arange(num_orbs)
        len_type = len(sel_orbs) * 2
        num_g_vec = len(self.g_vec_list_K1)
        gr_mtrx = cp.zeros((num_g_vec * len_type, n_wann), dtype=cp.complex128)

        iter = -1
        print(f"num of orb per cell = {num_orbs}, len type= {len_type} num g vec = {num_g_vec}")

        sort_wann = cp.asarray(self.structure.sort_wann)
        g_vec_list_K1 = cp.asarray(self.g_vec_list_K1)
        g_vec_list_K2 = cp.asarray(self.g_vec_list_K2)

        for ilayer in range(2):
            for i in tqdm(range(num_g_vec)):
                for k, iorb in enumerate(sel_orbs):
                    iter += 1
                    for j in range(n_wann):
                        if j % num_orbs == iorb and int(j / n_wann_perlayer) == ilayer:
                            wann_coord = sort_wann[j, :2]
                            if ilayer == 0:
                                gr_mtrx[iter, j] = cp.exp(-1j * cp.dot(g_vec_list_K1[i], wann_coord))
                            elif ilayer == 1:
                                gr_mtrx[iter, j] = cp.exp(-1j * cp.dot(g_vec_list_K2[i], wann_coord))

        factor = 1 / cp.sqrt(self.structure.num_unit_cell)
        gr_mtrx = factor * gr_mtrx

        delta = cp.abs(det(gr_mtrx @ gr_mtrx.T.conj()))
        if delta < 1.0e-2:
            raise Exception("cp.abs(det(gr_mtrx @ gr_mtrx.T.conj())) < 1.0e-2")
        else:
            print("cp.abs(det(gr_mtrx @ gr_mtrx.T.conj())) = ", delta)

        self.g_matrix = scipy.sparse.csr_matrix(cp.asnumpy(gr_mtrx))

    def generate_C3_matrix(self):
        """Generate the C3_matrix."""
        # Implement the logic to generate C3_matrix
        df_temp = self.structure.df.copy()
        
        atom_type_list = np.unique(df_temp['atom_type'].values)
        
        atom_layer_list = np.array([
            np.unique(df_temp[df_temp['atom_type'] == atom_type]['layer'].values)[0]
            for atom_type in atom_type_list
        ])
        
        atom_orb_name_list = np.array([
            np.unique(df_temp[df_temp['atom_type'] == atom_type]['orb_name'].values)[0]
            for atom_type in atom_type_list
        ])
        species_str_1layer = atom_orb_name_list[np.where(atom_layer_list == 0)[0]] #species =  ['Se7.0-s3p2d1' 'W7.0-s3p2d2' 'Se7.0-s3p2d1']
        species_str_2layer = atom_orb_name_list[np.where(atom_layer_list == 1)[0]]
        species_1layer = {}
        species_2layer = {}
        for idx, item in enumerate(species_str_1layer, start=1):
            # 使用'-'分割字符串，得到原子部分和轨道部分
            try:
                atom_part, orb_part = item.split('-')
            except ValueError:
                print(f"无法分割的项: {item}")
                continue
            
            # 使用正则表达式提取原子符号（假设原子符号由字母组成）
            atom_match = re.match(r'([A-Za-z]+)', atom_part)
            if atom_match:
                atom = atom_match.group(1)
            else:
                print(f"无法提取原子符号: {atom_part}")
                continue
            
            # 使用正则表达式提取轨道类型和对应的数量
            orbitals = {}
            for orb, count in re.findall(r'([spdf])(\d+)', orb_part):
                orbitals[orb] = int(count)
            
            # 构建字典
            species_1layer[idx] = {
                "atom": atom,
                "orbitals": orbitals
            }
        
        for idx, item in enumerate(species_str_2layer, start=1):
            # 使用'-'分割字符串，得到原子部分和轨道部分
            try:
                atom_part, orb_part = item.split('-')
            except ValueError:
                print(f"无法分割的项: {item}")
                continue
            
            # 使用正则表达式提取原子符号（假设原子符号由字母组成）
            atom_match = re.match(r'([A-Za-z]+)', atom_part)
            if atom_match:
                atom = atom_match.group(1)
            else:
                print(f"无法提取原子符号: {atom_part}")
                continue
            
            # 使用正则表达式提取轨道类型和对应的数量
            orbitals = {}
            for orb, count in re.findall(r'([spdf])(\d+)', orb_part):
                orbitals[orb] = int(count)
            
            # 构建字典
            species_2layer[idx] = {
                "atom": atom,
                "orbitals": orbitals
            }
        
        print("species 1layer = ",species_1layer)
        print("species 2layer = ",species_2layer)
        C3_G_rep_1layer, C3_G_rep_2layer = C3_G_matrix(self.g_vec_list_K1,self.g_vec_list_K2,self.structure.reciprocal_Tmat,self.structure.twist_index,valley=self.valley)
        C3_MoTe2_rep = C3_MoTe2_all(C3_G_rep_1layer, C3_G_rep_2layer,atoms_species_1layer=species_1layer,atoms_species_2layer=species_2layer,spin=self.structure.spin)
        self.C3_matrix = scipy.sparse.csr_matrix(C3_MoTe2_rep)
        C3_matrix_2 = self.C3_matrix @ self.C3_matrix
        # if not np.allclose(self.C3_matrix.toarray().conj().T, C3_matrix_2.toarray()):
        #     raise ValueError("C3_matrix is not Hermitian")
        # if not np.all
        self.symm_matrix = [scipy.sparse.csr_matrix(np.eye(self.C3_matrix.shape[0])),self.C3_matrix,C3_matrix_2]
        self.symm_matrix_inv = [scipy.sparse.csr_matrix(np.eye(self.C3_matrix.shape[0])),self.C3_matrix.conj().T,C3_matrix_2.conj().T]
        
    
    def generate_g_symm_matrix(self):
        """Generate the g_symm_matrix."""
        # Implement the logic to generate g_symm_matrix
        g_symm_matrix = self.symm_matrix[1] @ self.g_matrix
        g_symm_matrix_2 = self.symm_matrix[2] @ self.g_matrix
        g_symm_matrix_inv = self.g_matrix.conj().T @ self.symm_matrix_inv[1]
        g_symm_matrix_2_inv = self.g_matrix.conj().T @ self.symm_matrix_inv[2]
        self.g_symm_matrix = [self.g_matrix,g_symm_matrix,g_symm_matrix_2]
        self.g_symm_matrix_inv = [self.g_matrix.conj().T,g_symm_matrix_inv,g_symm_matrix_2_inv]
    def rotate_vector(self, vec, angle):
        """Rotate a vector by a given angle."""
        theta = np.radians(angle)
        rot_matrix = np.array([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta), np.cos(theta)]
        ])
        if len(vec) == 2:
            return np.dot(rot_matrix, vec)
        elif len(vec) == 3:
            return np.dot(rot_matrix, vec[:2])

    def generate_all_parameters(self):
        """Generate all TAPW parameters."""
        self.generate_g_vec_list()
        self.generate_gr_matrix()
        if self.config.C3_H:
            self.generate_C3_matrix()
            self.generate_g_symm_matrix()
        print("g matrix shape = ",self.g_matrix.shape)
        print("TAPW parameters generated successfully!")

class BandStructureCalculator:
    """Band structure and Chern number calculator for twisted materials"""
    
    # Valley mapping
    VALLEY_MAP = {
        1: "K1",
        2: "K2", 
        11: "K1_120",
        12: "K1_240",
        5: "Gamma",
        31: "M1",
        32: "M2",
        33: "M3"
    }

    def __init__(self, hr_supercell, sr_supercell, structure:StructureProcessor, config:ComputeConfig, kpath_config:KPathGenerator=None):
        """Initialize the calculator
        
        Args:
            hr_supercell: Hamiltonian matrix in real space
            sr_supercell: Overlap matrix in real space
            structure: Processed structure information
            config: Computation configuration
            kpath_config: K-path configuration (optional, only needed for band structure calculation)
        """
        self.hr_supercell = hr_supercell
        self.sr_supercell = sr_supercell
        self.structure = structure
        self.config = config
        self.kpath_config = kpath_config
        
        self.result = {}
        
        # Validate valley configuration
        if not hasattr(self.config, 'valley') or self.config.valley not in self.VALLEY_MAP:
            raise ValueError(f"Invalid valley configuration. Must be one of {list(self.VALLEY_MAP.keys())}")
        
        self.valley_flag = self.VALLEY_MAP[self.config.valley]
        
        if self.config.TAPW:
            self.TAPW_parameters = TAPW_parameters(self.structure, self.config)
            self.TAPW_parameters.generate_all_parameters()
            
        if self.config.gpu:
            try:
                import cupy as cp
            except ImportError:
                raise ImportError("CuPy is not installed. Please install CuPy to use GPU acceleration.")

    def generate_kmesh(self, num_k):
        """Generate a uniform k-point mesh for Chern number calculation
        
        Args:
            num_k: Number of k-points in each direction
            
        Returns:
            numpy.ndarray: Array of k-points in the first Brillouin zone
        """
        # a1 = np.array([2/3, -1/3, 0])
        # a2 = np.array([-1/3, 2/3, 0])
        
        # Generate uniform mesh
        kx = np.linspace(-1, 1, num_k, endpoint=True)
        ky = np.linspace(-1, 1, num_k, endpoint=True)
        # kx, ky = np.meshgrid(kx, ky)
        
        # Convert to cartesian coordinates
        # kpoints = np.zeros((num_k * num_k, 3))
        # for i in range(num_k):
        #     for j in range(num_k):
        #         k = kx[i,j] * a1 + ky[i,j] * a2
        #         kpoints[i*num_k + j] = k
        K_mesh = np.meshgrid(kx, ky)
        kpoints = np.array(K_mesh).reshape(2, -1).T
        kpoints = np.hstack((kpoints, np.zeros((kpoints.shape[0], 1))))
        
        return kpoints

    def calculate_band_structure(self, path, kpoints=None):
        """Calculate band structure
        
        Args:
            path: Output path for results
            kpoints: Optional k-points array. If None, uses k-path for band structure.
                    For Chern number calculation, should provide mesh k-points.
        """
        # Create output directories
        os.makedirs(path, exist_ok=True)
        os.makedirs(os.path.join(path, "band_data"), exist_ok=True)
        
        # Use provided k-points or generate from k-path
        if kpoints is None:
            if self.kpath_config is None:
                raise ValueError("Must provide either kpoints or kpath_config")
            kpoints = self.kpath_config.kpoints
        
        # Save g-vectors if using TAPW
        if self.config.TAPW:
            np.save(os.path.join(path, f"g_vec_list_{self.config.n_g}_{self.valley_flag}_1layer"),
                   self.TAPW_parameters.g_vec_list_K1)
            np.save(os.path.join(path, f"g_vec_list_{self.config.n_g}_{self.valley_flag}_2layer"),
                   self.TAPW_parameters.g_vec_list_K2)
        
        # Calculate bands
        # self.result = 
        self.parallel_calculate_band_01(kpoints)
        
        # Save results
        # band_data = np.column_stack((kpoints, self.result['eig']))
        band_data = self.result['eig']
        suffix = "_2d" if getattr(self.config, "mode", None) == "chern" else ""
        # 在chern模式下，添加num_chern标识
        if getattr(self.config, "mode", None) == "chern" and hasattr(self.config, "num_chern"):
            suffix += f"_{self.config.num_chern}"
        # 保存主能带数据
        np.savetxt(os.path.join(path, f"band_data/band_data_{self.valley_flag}_valley{suffix}.txt"),
                  band_data, fmt='%15.11f')
        
        # Save eigenvectors if calculated
        if self.config.eig_vec_cal:
            np.save(os.path.join(path, f"vec_{self.valley_flag}_valley{suffix}"), self.result['vec'])
        
        # Save Hamiltonian if requested
        if self.config.hamk_save:
            np.save(os.path.join(path, f"hamk_{self.valley_flag}_valley{suffix}"), self.result['hamk'])

    def calculate_chern(self, path):
        """Calculate Chern number using uniform k-point mesh
        
        Args:
            path: Output path for results
        """
        # Generate uniform k-point mesh
        kpoints = self.generate_kmesh(self.config.num_chern)
        print("kpoints shape = ",kpoints.shape)
        print("kpoints = ",kpoints)
        
        # Calculate band structure on the mesh
        self.calculate_band_structure(path, kpoints)
        
        # TODO: Implement actual Chern number calculation using Berry curvature
        # This would involve calculating the Berry curvature at each k-point
        # and integrating over the Brillouin zone
        
        return NotImplemented

    def run_calculation(self, path):
        """Main calculation entry point
        
        Args:
            path: Output path for results
        """
        if self.config.mode == "band":
            self.calculate_band_structure(path)
        elif self.config.mode == "chern":
            self.calculate_chern(path)
        else:
            raise ValueError(f"Unknown calculation mode: {self.config.mode}")

    def format_atomic_species_data(self, data,max_key_length):
        formatted_data = ""
        for key, value in data.items():
            value_str = ', '.join([f"{k}: {v}" if not isinstance(v, dict) else f"{k}: {{{', '.join([f'{kk}: {vv}' for kk, vv in v.items()])}}}" for k, v in value.items()])
            formatted_data += f"{key}: {{{value_str}}} \n" + ' ' * (max_key_length + 5)
        return formatted_data[:-max_key_length-7]  # Remove the trailing comma and space

    def rotate_mat(self, axis, radian):
        return scipy.linalg.expm(np.cross(np.eye(3), axis / scipy.linalg.norm(axis) * radian))

    # @timing_decorator_factory(process_id=0)
    def rot(self,vec,theta):
        theta = theta/180*np.pi
        rot_mat = np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]])
        if len(vec) == 2:
            return np.dot(rot_mat,vec)
        elif len(vec) == 3:
            temp = np.zeros(3)
            temp[:2] = np.dot(rot_mat,vec[:2])
            temp[2] = vec[2]
            return temp
    
    def rot_gpu(self, vec, theta):
        theta = theta / 180 * cp.pi
        rot_mat = cp.array([[cp.cos(theta), -cp.sin(theta)], [cp.sin(theta), cp.cos(theta)]])
        if len(vec) == 2:
            return cp.dot(rot_mat, vec)
        elif len(vec) == 3:
            temp = cp.zeros(3)
            temp[:2] = cp.dot(rot_mat, vec[:2])
            temp[2] = vec[2]
            return temp

    @timing_decorator_factory(process_id=0)
    def check_hermitian(self, M, tol=1e-4):
        if not scipy.sparse.issparse(M):
            if not np.allclose(M, M.conj().T, atol=tol):
                raise ValueError("Matrix is not Hermitian")
        else:
            if not np.allclose(M.toarray(), M.toarray().conj().T, atol=tol):
                raise ValueError("Matrix is not Hermitian")

    # @timing_decorator_factory(process_id=0)
    def get_kvec(self, k):
    
        return np.dot(k, self.structure.reciprocal_Tmat)

    @timing_decorator_factory(process_id=0)
    def Getk_super_gauge_sparse(self, Hr, k):
        kvec = self.get_kvec(k)
        # num_wann = self.structure.sort_wann.shape[0]
        
        sorted_wann_x = self.structure.df['x'].values
        sorted_wann_y = self.structure.df['y'].values
        sorted_wann_z = self.structure.df['z'].values
        # print("shape of sorted_wann_x = ",sorted_wann_x.shape,sorted_wann_y.shape,mk.shape)
        sorted_wann = np.array([sorted_wann_x, sorted_wann_y, sorted_wann_z]).T
        sorted_wann = np.repeat(np.array([sorted_wann_x, sorted_wann_y, sorted_wann_z]).T,self.structure.df['orb_num'].values,axis=0)

        if self.structure.spin:
            # sorted_wann = np.repeat(sorted_wann,2,axis=0)
            sorted_wann = np.concatenate([sorted_wann,sorted_wann],axis=0)
        
        num_wann = len(sorted_wann)
        mk = scipy.sparse.csr_matrix((num_wann, num_wann), dtype=np.complex128)
        # print("shape of sorted_wann = ",sorted_wann.shape)
        for rvec, values_dic in Hr.items():
            row_index, col_index, val_index = values_dic["row"], values_dic["col"], values_dic["val"]
            # m_coor, n_coor = self.structure.sort_wann[row_index], self.structure.sort_wann[col_index]
            # print("row_index = ",np.max(row_index),np.min(row_index),len(row_index),len(sorted_wann))
            # print("col_index = ",np.max(col_index),np.min(col_index),len(col_index),len(sorted_wann))
            m_coor, n_coor = sorted_wann[row_index], sorted_wann[col_index]
            Rvec = np.dot(rvec, self.structure.Tmat)
            phase_factor = np.exp(-1j * np.dot(m_coor - n_coor, kvec)) * np.exp(1j * np.dot(kvec, Rvec))
            # phase_factor = np.exp(-1j * np.dot(m_coor - n_coor, kvec))
            mk += scipy.sparse.csr_matrix((val_index * phase_factor, (row_index, col_index)), shape=(num_wann, num_wann))

        return mk
    
    @timing_decorator_factory(process_id=0)
    def Getk_super_gauge_sparse_new(self, Hr, k):
        # 获取 k 向量
        kvec = self.get_kvec(k)  # 假设返回一个长度为3的 NumPy 数组

        # 提取并构建 sorted_wann 数组
        sorted_wann_x = self.structure.df['x'].values
        sorted_wann_y = self.structure.df['y'].values
        sorted_wann_z = self.structure.df['z'].values

        # 构建 (x, y, z) 坐标数组
        sorted_wann = np.vstack([sorted_wann_x, sorted_wann_y, sorted_wann_z]).T
        # 根据 orb_num 重复每个 Wannier 函数的坐标
        sorted_wann = np.repeat(sorted_wann, self.structure.df['orb_num'].values, axis=0)

        # 如果存在自旋，则将 sorted_wann 重复一次
        if self.structure.spin:
            sorted_wann = np.concatenate([sorted_wann, sorted_wann], axis=0)

        num_wann = sorted_wann.shape[0]

        # 初始化用于存储稀疏矩阵的行、列和数据
        rows = []
        cols = []
        data = []

        # 提取 Tmat，假设是一个 3x3 的 NumPy 数组
        Tmat = self.structure.Tmat

        # 迭代 Hr 的所有项，收集行、列和数据
        for rvec, values_dic in tqdm(Hr.items(), desc="Processing Hr items"):
            row_index = values_dic["row"]  # 假设为 NumPy 数组
            col_index = values_dic["col"]  # 假设为 NumPy 数组
            val_index = values_dic["val"]  # 假设为 NumPy 数组

            # 获取对应的坐标
            m_coor = sorted_wann[row_index]  # Shape: (N, 3)
            n_coor = sorted_wann[col_index]  # Shape: (N, 3)

            # 计算 Rvec = rvec · Tmat
            Rvec = np.dot(rvec, Tmat)  # Shape: (3,)

            # 计算 (m_coor - n_coor) · kvec
            delta = m_coor - n_coor  # Shape: (N, 3)
            dot_delta_k = np.dot(delta, kvec)  # Shape: (N,)
            dot_k_Rvec = np.dot(kvec, Rvec)  # Scalar

            # 计算相位因子
            phase_factor = np.exp(-1j * dot_delta_k) * np.exp(1j * dot_k_Rvec)  # Shape: (N,)

            # 计算数据值
            data_values = val_index * phase_factor  # Shape: (N,)

            # 收集行、列和数据
            rows.append(row_index)
            cols.append(col_index)
            data.append(data_values)

        # 合并所有行、列和数据
        if rows:
            rows = np.concatenate(rows)
            cols = np.concatenate(cols)
            data = np.concatenate(data)
        else:
            # 如果没有数据，则创建空数组
            rows = np.array([], dtype=np.int32)
            cols = np.array([], dtype=np.int32)
            data = np.array([], dtype=np.complex128)

        # 创建稀疏矩阵，使用 COO 格式，然后转换为 CSR 格式
        mk = scipy.sparse.csr_matrix((data, (rows, cols)), shape=(num_wann, num_wann))

        return mk


    @timing_decorator_factory(process_id=0)
    def Getk_super_gauge_sparse_C3(self, Hr, k):
        # if self.config.gpu:
        #     return self.Getk_super_gauge_sparse_C3_gpu(Hr, k)
        # else:
        #     return self.Getk_super_gauge_sparse_C3_cpu(Hr, k)
        return self.Getk_super_gauge_sparse_C3_cpu(Hr, k)
    
    @staticmethod
    def convert_cupy_to_scipy(cupy_csr):
        data = cp.asnumpy(cupy_csr.data)
        indices = cp.asnumpy(cupy_csr.indices)
        indptr = cp.asnumpy(cupy_csr.indptr)
        shape = cupy_csr.shape
        scipy_csr = scipy.sparse.csr_matrix((data, indices, indptr), shape=shape)
        return scipy_csr

    def Getk_super_gauge_sparse_C3_cpu(self, Hr, k):
        start_time = time.time()
        kvec = self.get_kvec(k)
        
        K1 = self.TAPW_parameters.K1
        K2 = self.TAPW_parameters.K2
        temp_k1 = np.zeros(3)
        temp_k2 = np.zeros(3)
        temp_k1[:2] = K1
        temp_k2[:2] = K2
        K1 = temp_k1
        K2 = temp_k2


        rotations = [0, -120, -240]
        kvec_K1 = [self.rot(kvec + K1, angle) - K1 for angle in rotations]
        kvec_K2 = [self.rot(kvec + K2, angle) - K2 for angle in rotations]

        num_wann = self.structure.sort_wann.shape[0]
        mk_list = [scipy.sparse.csr_matrix((num_wann, num_wann), dtype=np.complex128) for _ in range(3)]
        num_wann_per_layer = num_wann // 2

        # 使用列表预分配数据
        data_list = [[] for _ in range(3)]
        row_list = [[] for _ in range(3)]
        col_list = [[] for _ in range(3)]

        loop_start_time = time.time()
        phase_time = 0
        mk_list_time = 0
        time_time = 0

        for i in range(3):
            for rvec, values_dic in Hr.items():
                Rvec = np.dot(rvec, self.structure.Tmat)
                row_index, col_index, val = values_dic["row"], values_dic["col"], values_dic["val"]

                # index_row_1 = row_index < num_wann_per_layer
                # index_row_2 = ~index_row_1
                # index_col_1 = col_index < num_wann_per_layer
                # index_col_2 = ~index_col_1

                index_row_1, index_row_2 = row_index < num_wann_per_layer, row_index >= num_wann_per_layer
                index_col_1, index_col_2 = col_index < num_wann_per_layer, col_index >= num_wann_per_layer

                row_coords_1 = self.structure.sort_wann[row_index[index_row_1]]
                row_coords_2 = self.structure.sort_wann[row_index[index_row_2]]
                col_coords_1 = self.structure.sort_wann[col_index[index_col_1]]
                col_coords_2 = self.structure.sort_wann[col_index[index_col_2]]

            
                phase_start_time = time.time()
                
                exp_kvec_K1_Rvec = np.exp(1j * np.dot(kvec_K1[i], Rvec))
                exp_kvec_K2_Rvec = np.exp(1j * np.dot(kvec_K2[i], Rvec))

                phase_m_1 = np.exp(-1j * np.dot(row_coords_1, kvec_K1[i])) * exp_kvec_K1_Rvec
                phase_m_2 = np.exp(-1j * np.dot(row_coords_2, kvec_K2[i])) * exp_kvec_K2_Rvec
                phase_n_1 = np.exp(1j * np.dot(col_coords_1, kvec_K1[i]))
                phase_n_2 = np.exp(1j * np.dot(col_coords_2, kvec_K2[i]))

                phase_m = np.zeros(len(row_index), dtype=np.complex128)
                phase_n = np.zeros(len(col_index), dtype=np.complex128)
                phase_m[index_row_1], phase_m[index_row_2] = phase_m_1, phase_m_2
                phase_n[index_col_1], phase_n[index_col_2] = phase_n_1, phase_n_2

                phase_factors = phase_m * phase_n
                phase_end_time = time.time()
                phase_time += phase_end_time - phase_start_time

                mk_list_start_time = time.time()
                val *= phase_factors 
                time_time += time.time() - phase_end_time
                # data_list[i].extend(val)
                # row_list[i].extend(row_index)
                # col_list[i].extend(col_index)
                mk_list[i] += scipy.sparse.csr_matrix((val, (row_index, col_index)), shape=(num_wann, num_wann))
                mk_list_end_time = time.time()
                mk_list_time += mk_list_end_time - mk_list_start_time

        loop_end_time = time.time()

        # 构建稀疏矩阵
        # sparse_matrix_start_time = time.time()
        # mk_list = []
        # for i in range(3):
        #     # coo = coo_matrix(
        #     #     (data_list[i], (row_list[i], col_list[i])),
        #     #     shape=(num_wann, num_wann)
        #     # )
        #     # mk_matrix = coo.tocsr()
        #     mk_list[i] = scipy.sparse.csr_matrix((data_list[i], (row_list[i], col_list[i])), shape=(num_wann, num_wann))
        #     # mk_list.append(mk_matrix)
        # sparse_matrix_end_time = time.time()
        # sparse_matrix_time = sparse_matrix_end_time - sparse_matrix_start_time
        # mk_list_time = sparse_matrix_time 
        
        print(f"Loop time: {loop_end_time - loop_start_time:.6f} seconds")
        print(f"Phase calculation time: {phase_time:.6f} seconds")
        print(f"time calculation time: {time_time:.6f} seconds")
        print(f"mk_list update time: {mk_list_time:.6f} seconds")

        total_time = time.time()
        print(f"Total time: {total_time - start_time:.6f} seconds")
        
        return mk_list

    def Getk_super_gauge_sparse_C3_gpu(self, Hr, k):
        kvec = cp.asarray(self.get_kvec(k))

        K1 = cp.asarray(self.TAPW_parameters.K1)
        K2 = cp.asarray(self.TAPW_parameters.K2)
        sort_wann = cp.asarray(self.structure.sort_wann)
        Tmat = cp.asarray(self.structure.Tmat)
        temp_k1 = cp.zeros(3)
        temp_k2 = cp.zeros(3)
        temp_k1[:2] = K1
        temp_k2[:2] = K2
        K1 = temp_k1
        K2 = temp_k2

        rotations = [0, -120, -240]

        kvec_K1 = [self.rot_gpu(kvec + K1, angle) - K1 for angle in rotations]
        kvec_K2 = [self.rot_gpu(kvec + K2, angle) - K2 for angle in rotations]

        num_wann = sort_wann.shape[0]
        mk_list = [cp.sparse.csr_matrix((num_wann, num_wann), dtype=cp.complex128) for _ in range(3)]

        num_wann_per_layer = num_wann // 2

        for rvec, values_dic in Hr.items():
            rvec = cp.asarray(rvec)
            Rvec = cp.dot(rvec, Tmat)
            row_index, col_index, val = values_dic["row"], values_dic["col"], values_dic["val"]
            val = cp.asarray(val)
            row_index = cp.asarray(row_index)
            col_index = cp.asarray(col_index)
            index_row_1, index_row_2 = row_index < num_wann_per_layer, row_index >= num_wann_per_layer
            index_col_1, index_col_2 = col_index < num_wann_per_layer, col_index >= num_wann_per_layer

            for i in range(3):
                phase_m_1 = cp.exp(-1j * cp.dot(sort_wann[row_index[index_row_1]], kvec_K1[i])) * cp.exp(1j * cp.dot(kvec_K1[i], Rvec))
                phase_m_2 = cp.exp(-1j * cp.dot(sort_wann[row_index[index_row_2]], kvec_K2[i])) * cp.exp(1j * cp.dot(kvec_K2[i], Rvec))
                phase_n_1 = cp.exp(1j * cp.dot(sort_wann[col_index[index_col_1]], kvec_K1[i]))
                phase_n_2 = cp.exp(1j * cp.dot(sort_wann[col_index[index_col_2]], kvec_K2[i]))

                phase_m = cp.zeros(len(row_index), dtype=cp.complex128)
                phase_n = cp.zeros(len(col_index), dtype=cp.complex128)
                phase_m[index_row_1], phase_m[index_row_2] = phase_m_1, phase_m_2
                phase_n[index_col_1], phase_n[index_col_2] = phase_n_1, phase_n_2

                phase_factors = phase_m * phase_n
                mk_list[i] += cp.sparse.csr_matrix((val * phase_factors, (row_index, col_index)), shape=(num_wann, num_wann))
        for i in range(3):
            mk_list[i] = self.convert_cupy_to_scipy(mk_list[i])
        #     mk_list[i] = cp.asnumpy(mk_list[i])
        #     mk_list[i] = scipy.sparse.csr_matrix(mk_list[i])
        return mk_list
    
    def Getk_super_gauge_sparse_symm_old(self, Hr, symm_matrix, symm_matrix_inv, k):
        
        def print_sparse_matrix_info(matrix):
            """
            打印矩阵的形状、非零元素数、稀疏程度和估计内存占用。
            
            参数:
                matrix (scipy.sparse matrix 或 numpy.ndarray): 要分析的矩阵。
            """
            # 检查是否为稀疏矩阵
            if scipy.sparse.isspmatrix(matrix):
                nnz = matrix.nnz  # 非零元素数量
                shape = matrix.shape  # 矩阵形状
                total_elements = shape[0] * shape[1]  # 矩阵总元素数
                sparsity_degree = nnz / total_elements  # 稀疏程度（非零元素占比）
                sparsity_percentage = sparsity_degree * 100  # 稀疏程度（百分比）
                
                # 计算内存占用
                memory_usage = 0
                for attr in ['data', 'indices', 'indptr', 'row', 'col']:
                    if hasattr(matrix, attr):
                        array = getattr(matrix, attr)
                        memory_usage += array.nbytes  # NumPy 数组的字节数
                
            # 检查是否为密集矩阵（NumPy ndarray）
            elif isinstance(matrix, np.ndarray):
                nnz = np.count_nonzero(matrix)  # 非零元素数量
                shape = matrix.shape  # 矩阵形状
                total_elements = matrix.size  # 矩阵总元素数
                sparsity_degree = nnz / total_elements  # 稀疏程度（非零元素占比）
                sparsity_percentage = sparsity_degree * 100  # 稀疏程度（百分比）
                
                # 计算内存占用
                memory_usage = matrix.nbytes  # NumPy 数组的字节数
                
            else:
                raise ValueError("输入的必须是一个 SciPy 稀疏矩阵或 NumPy ndarray。")
            
            print(f"\n矩阵信息:")
            print(f"形状: {shape}")
            print(f"非零元素数: {nnz}")
            print(f"稀疏程度: {sparsity_degree:.6f} ({sparsity_percentage:.2f}%)")
            print(f"估计内存占用: {memory_usage / 1e9:.6f} GB")
        
        start_time = time.time()
        kvec = self.get_kvec(k)
        step1_time = time.time()
        print(f"Step 1 (get_kvec) time: {step1_time - start_time:.6f} seconds")
        
        K1 = self.TAPW_parameters.K1
        K2 = self.TAPW_parameters.K2
        temp_k1 = np.zeros(3)
        temp_k2 = np.zeros(3)
        temp_k1[:2] = K1
        temp_k2[:2] = K2
        K1 = temp_k1
        K2 = temp_k2
        step2_time = time.time()
        print(f"Step 2 (K1, K2 setup) time: {step2_time - step1_time:.6f} seconds")

        rotations = [0, -120, -240]
        kvec_K1 = [self.rot(kvec + K1, angle) - K1 for angle in rotations]
        kvec_K2 = [self.rot(kvec + K2, angle) - K2 for angle in rotations]
        step3_time = time.time()
        print(f"Step 3 (kvec_K1, kvec_K2 calculation) time: {step3_time - step2_time:.6f} seconds")

        num_wann = self.structure.sort_wann.shape[0]
        # mk = scipy.sparse.csr_matrix((num_wann, num_wann), dtype=np.complex128)
        # mk_list = [scipy.sparse.csr_matrix((num_wann, num_wann), dtype=np.complex128) for _ in range(3)]
        mk_list = [np.zeros((self.TAPW_parameters.g_matrix.shape[0],self.TAPW_parameters.g_matrix.shape[0]),dtype=np.complex128) for _ in range(3)]
        num_wann_per_layer = num_wann // 2
        step4_time = time.time()
        print(f"Step 4 (mk_list initialization) time: {step4_time - step3_time:.6f} seconds")

        loop_start_time = time.time()
        index_time = 0
        phase_time = 0
        mk_list_time = 0
        time_time = 0
        symm_time = 0
        mk = 0
        for i in range(3):
            for rvec, values_dic in Hr.items():
                Rvec = np.dot(rvec, self.structure.Tmat)
                index_start_time = time.time()
                row_index, col_index, val = values_dic["row"], values_dic["col"], values_dic["val"]

                
                # 创建布尔掩码
                mask_row_1 = row_index < num_wann_per_layer
                mask_row_2 = row_index >= num_wann_per_layer
                mask_col_1 = col_index < num_wann_per_layer
                mask_col_2 = col_index >= num_wann_per_layer

                # 使用布尔掩码进行索引
                row_coords_1 = self.structure.sort_wann[row_index[mask_row_1]]
                row_coords_2 = self.structure.sort_wann[row_index[mask_row_2]]
                col_coords_1 = self.structure.sort_wann[col_index[mask_col_1]]
                col_coords_2 = self.structure.sort_wann[col_index[mask_col_2]]

            
                phase_start_time = time.time()
                index_time += phase_start_time - index_start_time

                
                exp_kvec_K1_Rvec = np.exp(1j * np.dot(kvec_K1[i], Rvec))
                exp_kvec_K2_Rvec = np.exp(1j * np.dot(kvec_K2[i], Rvec))

                phase_m_1 = np.exp(-1j * np.dot(row_coords_1, kvec_K1[i])) * exp_kvec_K1_Rvec
                phase_m_2 = np.exp(-1j * np.dot(row_coords_2, kvec_K2[i])) * exp_kvec_K2_Rvec
                phase_n_1 = np.exp(1j * np.dot(col_coords_1, kvec_K1[i]))
                phase_n_2 = np.exp(1j * np.dot(col_coords_2, kvec_K2[i]))

                phase_m = np.zeros(len(row_index), dtype=np.complex128)
                phase_n = np.zeros(len(col_index), dtype=np.complex128)
                phase_m[mask_row_1], phase_m[mask_row_2] = phase_m_1, phase_m_2
                phase_n[mask_col_1], phase_n[mask_col_2] = phase_n_1, phase_n_2

                phase_factors = phase_m * phase_n
                phase_end_time = time.time()
                phase_time += phase_end_time - phase_start_time

                mk_list_start_time = time.time()
                # mk += symm_matrix[i]@self.cal_TAPW_hamiltonian_k(scipy.sparse.csr_matrix((val * phase_factors, (row_index, col_index)), shape=(num_wann, num_wann)))@symm_matrix_inv[i]
                # mk += self.TAPW_parameters.g_symm_matrix[i]@scipy.sparse.csr_matrix((val * phase_factors, (row_index, col_index)), shape=(num_wann, num_wann))@self.TAPW_parameters.g_symm_matrix_inv[i]
                # val *= phase_factors
                time_end_time = time.time()
                time_time += time_end_time - phase_end_time
                partial_mk = scipy.sparse.csr_matrix((val*phase_factors, (row_index, col_index)), shape=(num_wann, num_wann))
                # temp = symm_matrix[i] @ self.cal_TAPW_hamiltonian_k(partial_mk) @ symm_matrix_inv[i]
                symm_time += time.time() - time_end_time
                # print_sparse_matrix_info(partial_mk)
                # mk += temp
                # mk += symm_matrix[i] @ self.cal_TAPW_hamiltonian_k(partial_mk) @ symm_matrix_inv[i]
                temp = self.cal_TAPW_hamiltonian_k(partial_mk)
                # print_sparse_matrix_info(temp)
                # self.check_sparsity(temp)
                mk_list[i] += temp
                mk_list_end_time = time.time()
                mk_list_time += mk_list_end_time - mk_list_start_time

        loop_end_time = time.time()
        print(f"Loop time: {loop_end_time - loop_start_time:.6f} seconds")
        print(f"Index calculation time: {index_time:.6f} seconds")
        print(f"Phase calculation time: {phase_time:.6f} seconds")
        print(f"Time calculation time: {time_time:.6f} seconds")
        print(f"Symm calculation time: {symm_time:.6f} seconds")
        print(f"mk_list update time: {mk_list_time:.6f} seconds")
        # return mk.toarray()
        return mk_list

    @timing_decorator_factory(process_id=0)
    def Getk_super_gauge_sparse_symm(self, Hr, k):
        
        def print_sparse_matrix_info(matrix):
            """
            打印矩阵的形状、非零元素数、稀疏程度和估计内存占用。
            
            参数:
                matrix (scipy.sparse matrix 或 numpy.ndarray): 要分析的矩阵。
            """
            # 检查是否为稀疏矩阵
            if scipy.sparse.isspmatrix(matrix):
                nnz = matrix.nnz  # 非零元素数量
                shape = matrix.shape  # 矩阵形状
                total_elements = shape[0] * shape[1]  # 矩阵总元素数
                sparsity_degree = nnz / total_elements  # 稀疏程度（非零元素占比）
                sparsity_percentage = sparsity_degree * 100  # 稀疏程度（百分比）
                
                # 计算内存占用
                memory_usage = 0
                for attr in ['data', 'indices', 'indptr', 'row', 'col']:
                    if hasattr(matrix, attr):
                        array = getattr(matrix, attr)
                        memory_usage += array.nbytes  # NumPy 数组的字节数
                
            # 检查是否为密集矩阵（NumPy ndarray）
            elif isinstance(matrix, np.ndarray):
                nnz = np.count_nonzero(matrix)  # 非零元素数量
                shape = matrix.shape  # 矩阵形状
                total_elements = matrix.size  # 矩阵总元素数
                sparsity_degree = nnz / total_elements  # 稀疏程度（非零元素占比）
                sparsity_percentage = sparsity_degree * 100  # 稀疏程度（百分比）
                
                # 计算内存占用
                memory_usage = matrix.nbytes  # NumPy 数组的字节数
                
            else:
                raise ValueError("输入的必须是一个 SciPy 稀疏矩阵或 NumPy ndarray。")
            
            print(f"\n矩阵信息:")
            print(f"形状: {shape}")
            print(f"非零元素数: {nnz}")
            print(f"稀疏程度: {sparsity_degree:.6f} ({sparsity_percentage:.2f}%)")
            print(f"估计内存占用: {memory_usage / 1e9:.6f} GB")
        
        kvec = self.get_kvec(k)
        sorted_wann_x = self.structure.df['x'].values
        sorted_wann_y = self.structure.df['y'].values
        sorted_wann_z = self.structure.df['z'].values
        # print("shape of sorted_wann_x = ",sorted_wann_x.shape,sorted_wann_y.shape,mk.shape)
        # sorted_wann = np.array([sorted_wann_x, sorted_wann_y, sorted_wann_z]).T
        sorted_wann = np.repeat(np.array([sorted_wann_x, sorted_wann_y, sorted_wann_z]).T,self.structure.df['orb_num'].values,axis=0)
        sorted_layer_index = np.repeat(np.array(self.structure.df['layer']),self.structure.df['orb_num'].values,axis=0)
        if self.structure.spin:
            # sorted_wann = np.repeat(sorted_wann,2,axis=0)
            sorted_wann = np.concatenate([sorted_wann,sorted_wann],axis=0)
            sorted_layer_index = np.concatenate([sorted_layer_index,sorted_layer_index],axis=0)
        # print("sorted_layer_index = ",sorted_layer_index)
        num_wann = len(sorted_wann)
        
        start_time = time.time()
        kvec = self.get_kvec(k)
        step1_time = time.time()
        # print(f"Step 1 (get_kvec) time: {step1_time - start_time:.6f} seconds")
        
        K1 = self.TAPW_parameters.K1
        K2 = self.TAPW_parameters.K2
        temp_k1 = np.zeros(3)
        temp_k2 = np.zeros(3)
        temp_k1[:2] = K1
        temp_k2[:2] = K2
        K1 = temp_k1
        K2 = temp_k2
        step2_time = time.time()
        # print(f"Step 2 (K1, K2 setup) time: {step2_time - step1_time:.6f} seconds")

        rotations = [0, -120, -240]
        kvec_K1 = [self.rot(kvec + K1, angle) - K1 for angle in rotations]
        kvec_K2 = [self.rot(kvec + K2, angle) - K2 for angle in rotations]
        step3_time = time.time()
        if np.sum(np.abs(kvec)) < 1e-6:
            print("K1 = ",kvec_K1)
            print("K2 = ",kvec_K2)
            print("sum abs self.TAPW_parameters.g_matrix = ",np.sum(np.abs(self.TAPW_parameters.g_matrix)))
            # print("norm self.TAPW_parameters.g_matrix = ",np.linalg.norm(self.TAPW_parameters.g_matrix))
            # exit()
        # print(f"Step 3 (kvec_K1, kvec_K2 calculation) time: {step3_time - step2_time:.6f} seconds")

        # num_wann = self.structure.sort_wann.shape[0]
        # mk = scipy.sparse.csr_matrix((num_wann, num_wann), dtype=np.complex128)
        # mk_list = [scipy.sparse.csr_matrix((num_wann, num_wann), dtype=np.complex128) for _ in range(3)]
        mk_list = [np.zeros((self.TAPW_parameters.g_matrix.shape[0],self.TAPW_parameters.g_matrix.shape[0]),dtype=np.complex128) for _ in range(3)]
        num_wann_per_layer = num_wann // 2
        step4_time = time.time()
        # print(f"Step 4 (mk_list initialization) time: {step4_time - step3_time:.6f} seconds")

        loop_start_time = time.time()
        index_time = 0
        phase_time = 0
        mk_list_time = 0
        time_time = 0
        symm_time = 0
        mk = 0
        for i in range(3):
            partial_mk = 0
            row_list = []
            col_list = []
            val_list = []
            for rvec, values_dic in Hr.items():
                Rvec = np.dot(rvec, self.structure.Tmat)
                index_start_time = time.time()
                row_index, col_index, val = values_dic["row"], values_dic["col"], values_dic["val"]

                
                # 创建布尔掩码
                mask_row_1 = sorted_layer_index[row_index] == 0
                mask_row_2 = ~mask_row_1
                mask_col_1 = sorted_layer_index[col_index] == 0
                mask_col_2 = ~mask_col_1
                
                row_coords_1 = sorted_wann[row_index[mask_row_1]]
                row_coords_2 = sorted_wann[row_index[mask_row_2]]
                col_coords_1 = sorted_wann[col_index[mask_col_1]]
                col_coords_2 = sorted_wann[col_index[mask_col_2]]

            
                phase_start_time = time.time()
                index_time += phase_start_time - index_start_time

                
                exp_kvec_K1_Rvec = np.exp(1j * np.dot(kvec_K1[i], Rvec))
                exp_kvec_K2_Rvec = np.exp(1j * np.dot(kvec_K2[i], Rvec))

                phase_m_1 = np.exp(-1j * np.dot(row_coords_1, kvec_K1[i])) * exp_kvec_K1_Rvec
                phase_m_2 = np.exp(-1j * np.dot(row_coords_2, kvec_K2[i])) * exp_kvec_K2_Rvec
                phase_n_1 = np.exp(1j * np.dot(col_coords_1, kvec_K1[i]))
                phase_n_2 = np.exp(1j * np.dot(col_coords_2, kvec_K2[i]))

                phase_m = np.zeros(len(row_index), dtype=np.complex128)
                phase_n = np.zeros(len(col_index), dtype=np.complex128)
                phase_m[mask_row_1], phase_m[mask_row_2] = phase_m_1, phase_m_2
                phase_n[mask_col_1], phase_n[mask_col_2] = phase_n_1, phase_n_2

                phase_factors = phase_m * phase_n
                phase_end_time = time.time()
                phase_time += phase_end_time - phase_start_time

                # mk_list_start_time = time.time()
                # mk += symm_matrix[i]@self.cal_TAPW_hamiltonian_k(scipy.sparse.csr_matrix((val * phase_factors, (row_index, col_index)), shape=(num_wann, num_wann)))@symm_matrix_inv[i]
                # mk += self.TAPW_parameters.g_symm_matrix[i]@scipy.sparse.csr_matrix((val * phase_factors, (row_index, col_index)), shape=(num_wann, num_wann))@self.TAPW_parameters.g_symm_matrix_inv[i]
                # print("=====================================")
                # print("val = ",type(val),val)
                # print("phase_factors = ",type(phase_factors))
                data_values = val * phase_factors #*= phase_factors
                time_end_time = time.time()
                time_time += time_end_time - phase_end_time
                # row_list.extend(row_index)
                # col_list.extend(col_index)
                # val_list.extend(val*phase_factors)
                # print("=====================================")
                # print(f"i = {i} Rvec = {Rvec}")
                # print("len(row_index) = ",len(row_index))
                # print("len(col_index) = ",len(col_index))
                # print("len(val) = ",len(val))
                partial_mk += scipy.sparse.csr_matrix((data_values, (row_index, col_index)), shape=(num_wann, num_wann))
                # temp = symm_matrix[i] @ self.cal_TAPW_hamiltonian_k(partial_mk) @ symm_matrix_inv[i]
                symm_time += time.time() - time_end_time
                print(f"i = {i} Rvec = {Rvec} len(val) = {len(data_values)} time = {time.time() - time_end_time}")
                
                # mk += temp
                # mk += symm_matrix[i] @ self.cal_TAPW_hamiltonian_k(partial_mk) @ symm_matrix_inv[i]
            # time_end_time = time.time()
            # print("=====================================")
            # print("len(row_list) = ",len(row_list))
            # print("len(col_list) = ",len(col_list))
            # print("len(val_list) = ",len(val_list))
            # partial_mk = scipy.sparse.csr_matrix((val_list, (row_list, col_list)), shape=(num_wann, num_wann))
            # symm_time += time.time() - time_end_time
            mk_list_start_time = time.time()
            # print("time = ",time.time() - time_end_time)
            print_sparse_matrix_info(partial_mk) 
            temp = self.cal_TAPW_hamiltonian_k(partial_mk)

            mk_list[i] = temp
            mk_list_end_time = time.time()
            mk_list_time += mk_list_end_time - mk_list_start_time
            
        loop_end_time = time.time()
        print(f"Loop time: {loop_end_time - loop_start_time:.6f} seconds")
        print(f"Index calculation time: {index_time:.6f} seconds")
        print(f"Phase calculation time: {phase_time:.6f} seconds")
        print(f"Time calculation time: {time_time:.6f} seconds")
        print(f"Symm calculation time: {symm_time:.6f} seconds")
        print(f"mk_list update time: {mk_list_time:.6f} seconds")
        # return mk.toarray()
        return mk_list
    
    @timing_decorator_factory(process_id=0)
    def Getk_super_gauge_sparse_symm_new(self, Hr, k):
        def print_sparse_matrix_info(matrix):
            """
            打印矩阵的形状、非零元素数、稀疏程度和估计内存占用。
            
            参数:
                matrix (scipy.sparse matrix 或 numpy.ndarray): 要分析的矩阵。
            """
            if scipy.sparse.isspmatrix(matrix):
                nnz = matrix.nnz
                shape = matrix.shape
                total_elements = shape[0] * shape[1]
                sparsity_degree = nnz / total_elements
                sparsity_percentage = sparsity_degree * 100
                memory_usage = sum(getattr(matrix, attr).nbytes for attr in ['data', 'indices', 'indptr'])
            elif isinstance(matrix, np.ndarray):
                nnz = np.count_nonzero(matrix)
                shape = matrix.shape
                total_elements = matrix.size
                sparsity_degree = nnz / total_elements
                sparsity_percentage = sparsity_degree * 100
                memory_usage = matrix.nbytes
            else:
                raise ValueError("输入的必须是一个 SciPy 稀疏矩阵或 NumPy ndarray。")
            
            print(f"\n矩阵信息:")
            print(f"形状: {shape}")
            print(f"非零元素数: {nnz}")
            print(f"稀疏程度: {sparsity_degree:.6f} ({sparsity_percentage:.2f}%)")
            print(f"估计内存占用: {memory_usage / 1e9:.6f} GB")

        # 获取 k 向量
        kvec = self.get_kvec(k)

        # 提取并构建 sorted_wann 数组
        sorted_wann_x = self.structure.df['x'].values
        sorted_wann_y = self.structure.df['y'].values
        sorted_wann_z = self.structure.df['z'].values

        # 构建 (x, y, z) 坐标数组并重复
        sorted_wann = np.vstack([sorted_wann_x, sorted_wann_y, sorted_wann_z]).T
        sorted_wann = np.repeat(sorted_wann, self.structure.df['orb_num'].values, axis=0)

        # 构建 sorted_layer_index 数组并重复
        sorted_layer_index = np.repeat(self.structure.df['layer'].values, self.structure.df['orb_num'].values, axis=0)

        # 处理自旋
        if self.structure.spin:
            sorted_wann = np.concatenate([sorted_wann, sorted_wann], axis=0)
            sorted_layer_index = np.concatenate([sorted_layer_index, sorted_layer_index], axis=0)

        num_wann = sorted_wann.shape[0]

        # 预计算旋转后的 k 向量
        rotations = [0, -120, -240]
        kvec_K1 = []
        kvec_K2 = []
        K1 = self.TAPW_parameters.K1
        K2 = self.TAPW_parameters.K2
        temp_k1 = np.zeros(3)
        temp_k2 = np.zeros(3)
        temp_k1[:2] = K1
        temp_k2[:2] = K2
        K1 = temp_k1
        K2 = temp_k2
        for angle in rotations:
            rotated_k1 = self.rot(kvec + K1, angle) - K1
            rotated_k2 = self.rot(kvec + K2, angle) - K2
            kvec_K1.append(rotated_k1)
            kvec_K2.append(rotated_k2)

        # 初始化用于存储每个旋转的行、列和数据
        rows_list = [[] for _ in range(3)]
        cols_list = [[] for _ in range(3)]
        data_list = [[] for _ in range(3)]

        # 提取 TAPW 参数
        Tmat = self.structure.Tmat
        g_matrix_size = self.TAPW_parameters.g_matrix.shape[0]

        # 迭代 Hr 的所有项，收集行、列和数据
        for rvec, values_dic in Hr.items():#, desc="Processing Hr items"):
            row_index = values_dic["row"]  # 假设为 NumPy 数组
            col_index = values_dic["col"]  # 假设为 NumPy 数组
            val = values_dic["val"]        # 假设为 NumPy 数组

            # 计算 Rvec = rvec · Tmat
            Rvec = np.dot(rvec, Tmat)  # Shape: (3,)

            # 计算每个旋转的相位因子并收集数据
            for i in range(3):
                # 计算相位因子
                exp_kvec_K1_Rvec = np.exp(1j * np.dot(kvec_K1[i], Rvec))
                exp_kvec_K2_Rvec = np.exp(1j * np.dot(kvec_K2[i], Rvec))

                # 创建布尔掩码
                mask_row_1 = sorted_layer_index[row_index] == 0
                mask_row_2 = ~mask_row_1
                mask_col_1 = sorted_layer_index[col_index] == 0
                mask_col_2 = ~mask_col_1

                # 获取对应的坐标
                row_coords_1 = sorted_wann[row_index[mask_row_1]]
                row_coords_2 = sorted_wann[row_index[mask_row_2]]
                col_coords_1 = sorted_wann[col_index[mask_col_1]]
                col_coords_2 = sorted_wann[col_index[mask_col_2]]

                # 计算相位因子
                phase_m_1 = np.exp(-1j * np.dot(row_coords_1, kvec_K1[i])) * exp_kvec_K1_Rvec
                phase_m_2 = np.exp(-1j * np.dot(row_coords_2, kvec_K2[i])) * exp_kvec_K2_Rvec
                phase_n_1 = np.exp(1j * np.dot(col_coords_1, kvec_K1[i]))
                phase_n_2 = np.exp(1j * np.dot(col_coords_2, kvec_K2[i]))

                # 合并相位因子
                phase_m = np.empty_like(val, dtype=np.complex128)
                phase_n = np.empty_like(val, dtype=np.complex128)
                phase_m[mask_row_1] = phase_m_1
                phase_m[mask_row_2] = phase_m_2
                phase_n[mask_col_1] = phase_n_1
                phase_n[mask_col_2] = phase_n_2
                phase_factors = phase_m * phase_n

                # 计算数据值
                data_values = val * phase_factors

                # 收集行、列和数据
                rows_list[i].extend(row_index)
                cols_list[i].extend(col_index)
                data_list[i].extend(data_values)

        # 创建稀疏矩阵并应用 TAPW 哈密顿量转换
        mk_list = []
        for i in range(3):
            if rows_list[i]:
                mk_csr = scipy.sparse.csr_matrix((data_list[i], (rows_list[i], cols_list[i])), shape=(num_wann, num_wann))
                # mk_csr = mk_coo.tocsr()
                # 应用 TAPW 哈密顿量转换
                mk_transformed = self.cal_TAPW_hamiltonian_k(mk_csr)
                mk_list.append(mk_transformed)
            else:
                # 如果没有数据，则创建空的稀疏矩阵
                mk_transformed = scipy.sparse.csr_matrix((num_wann, num_wann), dtype=np.complex128)
                mk_list.append(mk_transformed)

        return mk_list
    
    @staticmethod
    def is_positive_definite(matrix, method='cholesky', tol=1e-10):
        """
        判断一个矩阵是否为正定矩阵。

        参数：
        - matrix (numpy.ndarray): 要检查的矩阵。
        - method (str): 使用的方法，可以是 'cholesky' 或 'eigen'。默认使用 'cholesky'。
        - tol (float): 容差，用于数值稳定性。默认值为1e-10。

        返回值：
        - bool: 如果矩阵是正定的，返回 True，否则返回 False。
        """
        # 检查输入是否为二维方阵
        if not isinstance(matrix, np.ndarray):
            raise TypeError("输入必须是一个NumPy数组。")
        if matrix.ndim != 2:
            raise ValueError("输入必须是一个二维矩阵。")
        rows, cols = matrix.shape
        if rows != cols:
            raise ValueError("输入矩阵必须是方阵。")
        
        # 检查矩阵是否对称
        if not np.allclose(matrix, matrix.T.conj(), atol=tol):
            print("矩阵不是对称的。")
            return False
        
        if method == 'cholesky':
            try:
                # 尝试进行Cholesky分解
                np.linalg.cholesky(matrix)
                return True
            except LinAlgError:
                print("Cholesky分解失败，矩阵不是正定的。")
                return False
        elif method == 'eigen':
            # 计算特征值
            eigenvalues = scipy.linalg.eigvalsh(matrix)
            if np.all(eigenvalues > tol):
                print(f"true 最小10个特征值: {np.sort(eigenvalues)[:10]}")
                return True
            else:
                print("存在非正特征值，矩阵不是正定的。")
                print(f"最小10个特征值: {np.sort(eigenvalues)[:200]}")
                return False
        else:
            raise ValueError("未知的方法。请选择 'cholesky' 或 'eigen'。")

    @timing_decorator_factory(process_id=0)
    def Getk_super_gauge_sparse_symm_final_HS(self, Hr,Sr,symm_matrix,symm_matrix_inv,k,mpi_index):
        Hk_list = self.Getk_super_gauge_sparse_symm(Hr, k)
        Sk_list = self.Getk_super_gauge_sparse_symm(Sr, k)

        if self.config.ge:
            HK_new = self.C3_symm(Hk_list[0], Hk_list[1], Hk_list[2], self.TAPW_parameters.C3_matrix)
            Sk_new = self.C3_symm(Sk_list[0], Sk_list[1], Sk_list[2], self.TAPW_parameters.C3_matrix)
            return HK_new, Sk_new
        else:
            Hk_new_list = [self.gen_H_new(Hk, Sk,mpi_index) for Hk, Sk in zip(Hk_list, Sk_list)]
            Hk_new = self.C3_symm(Hk_new_list[0], Hk_new_list[1], Hk_new_list[2], self.TAPW_parameters.C3_matrix)
            return Hk_new,0
    
    @timing_decorator_factory(process_id=0) 
    def Getk_super_gauge_sparse_final_HS(self, Hr, Sr, k, mpi_index):
        Hk = self.Getk_super_gauge_sparse(Hr, k)
        Sk = self.Getk_super_gauge_sparse(Sr, k)
        Hk = self.cal_TAPW_hamiltonian_k(Hk)
        Sk = self.cal_TAPW_hamiltonian_k(Sk)
        if not self.config.ge:
            Hk = self.gen_H_new(Hk, Sk, mpi_index)
            return Hk, 0
        else:
            return Hk, Sk


    @timing_decorator_factory(process_id=0)
    def C3_symm(self, hamk, hamk_C1, hamk_C2, C3_matrix):
        if not self.config.gpu:
            return self.C3_symm_cpu(hamk, hamk_C1, hamk_C2, C3_matrix)
        else:
            return self.C3_symm_gpu(hamk, hamk_C1, hamk_C2, C3_matrix)

    def C3_symm_cpu(self, hamk, hamk_C1, hamk_C2, C3_matrix):
        C3_matrix_2 = C3_matrix @ C3_matrix
        return (hamk + C3_matrix @ hamk_C1 @ C3_matrix.conj().T + C3_matrix_2 @ hamk_C2 @ C3_matrix_2.conj().T) / 3

    def C3_symm_gpu(self, hamk, hamk_C1, hamk_C2, C3_matrix):
        C3_matrix_2 = C3_matrix @ C3_matrix
        return (hamk + C3_matrix @ hamk_C1 @ C3_matrix.conj().T + C3_matrix_2 @ hamk_C2 @ C3_matrix_2.conj().T) / 3

    # @timing_decorator_factory(process_id=0)
    def gen_H_new(self, hamk, samk, gpu_index=0):
        if not self.config.gpu:
            return self.gen_H_new_cpu(hamk, samk)
        else:
            return self.gen_H_new_gpu(hamk, samk, gpu_index)
    
    @timing_decorator_factory(process_id=0)
    def gen_H_new_cpu(self, hamk, samk):
        S_eig, S_vec = scipy.linalg.eigh(samk)
        M_inv = np.diag(1 / np.sqrt(S_eig))
        UMinvUd = S_vec @ M_inv @ S_vec.conj().T
        # UMinvUd = (scipy.sparse.csr_matrix(S_vec) @ scipy.sparse.csr_matrix(M_inv)).toarray() @ S_vec.conj().T
        return UMinvUd @ hamk @ UMinvUd
    
    @timing_decorator_factory(process_id=0)
    def gen_H_new_gpu(self, hamk, samk, gpu_index=0):
        with cp.cuda.Device(gpu_index):
            
            samk_gpu = cp.asarray(samk)
            
            S_eig_gpu, S_vec_gpu = cp.linalg.eigh(samk_gpu)
            self.del_cupy_gpu(samk_gpu)
            
            M_inv_gpu = cp.diag(1 / cp.sqrt(S_eig_gpu))
            
            UMinvUd_gpu = S_vec_gpu @ M_inv_gpu @ S_vec_gpu.conj().T
            self.del_cupy_gpu(S_vec_gpu, S_eig_gpu, M_inv_gpu)

            hamk_gpu = cp.asarray(hamk)
            UH_gpu = UMinvUd_gpu @ hamk_gpu
            
            
            hamk_new_gpu = UH_gpu @ UMinvUd_gpu#.conj().T
            self.del_cupy_gpu(UMinvUd_gpu, UH_gpu)
            
            result = cp.asnumpy(hamk_new_gpu)
            self.del_cupy_gpu(hamk_gpu, hamk_new_gpu)

            return result

    @timing_decorator_factory(process_id=0)
    def gen_S_inv(self, samk, gpu_index=0):
        if not self.config.gpu:
            return self.gen_S_inv_cpu(samk)
        else:
            return self.gen_S_inv_gpu(samk, gpu_index)
        
    def gen_S_inv_cpu(self, samk):
        S_eig, S_vec = scipy.linalg.eigh(samk)
        M_inv = np.diag(1 / np.sqrt(S_eig))
        UMinvUd = (scipy.sparse.csr_matrix(S_vec) @ scipy.sparse.csr_matrix(M_inv)).toarray() @ S_vec.conj().T
        return UMinvUd
    
    def gen_S_inv_gpu(self, samk, gpu_index=0):
        with cp.cuda.Device(gpu_index):
            samk_gpu = cp.asarray(samk)
            S_eig_gpu, S_vec_gpu = cp.linalg.eigh(samk_gpu)

            self.del_cupy_gpu(samk_gpu)            
            M_inv_gpu = cp.diag(1 / cp.sqrt(S_eig_gpu))
            
            UMinvUd_gpu = S_vec_gpu @ M_inv_gpu @ S_vec_gpu.conj().T
            self.del_cupy_gpu(S_vec_gpu, M_inv_gpu)
            return UMinvUd_gpu
    
    def del_cupy_gpu(self, *args):
        for arg in args:
            del arg
        cp.get_default_memory_pool().free_all_blocks()
        cp.get_default_pinned_memory_pool().free_all_blocks()


    # @timing_decorator_factory(process_id=0)
    def check_sparsity(self,matrix):
        """
        Check the sparsity of a matrix.

        Parameters:
        matrix (numpy.ndarray or scipy.sparse matrix): The matrix to check.

        Returns:
        float: The sparsity ratio of the matrix.
        """
        if scipy.sparse.issparse(matrix):
            non_zero_elements = matrix.count_nonzero()
            total_elements = matrix.shape[0] * matrix.shape[1]
        else:
            non_zero_elements = np.count_nonzero(matrix)
            total_elements = matrix.size

        sparsity_ratio = non_zero_elements / total_elements
        sparsity_percentage = (1 - sparsity_ratio) * 100

        print(f"Matrix shape: {matrix.shape}")
        print(f"Non-zero elements: {non_zero_elements}")
        print(f"Total elements: {total_elements}")
        print(f"Sparsity ratio: {sparsity_ratio:.6f}")
        # print(f"Sparsity percentage: {sparsity_percentage:.2f}%")

        return sparsity_ratio
    
    @timing_decorator_factory(process_id=0)
    def cal_TAPW_hamiltonian_k(self, hamk):
        # if not self.config.gpu:
        #     return self.cal_TAPW_hamiltonian_k_cpu(hamk)
        # else:
        #     return self.cal_TAPW_hamiltonian_k_gpu(hamk)
        return self.cal_TAPW_hamiltonian_k_cpu(hamk)

    def cal_TAPW_hamiltonian_k_cpu(self, hamk):
        # result = self.TAPW_parameters.g_matrix.dot(hamk.dot(self.TAPW_parameters.g_matrix.conj().T))
        result = self.TAPW_parameters.g_matrix @ hamk @ self.TAPW_parameters.g_matrix_conj
        return result.toarray()

    def cal_TAPW_hamiltonian_k_cpu_new(self, hamk):
        # result = self.TAPW_parameters.g_matrix.dot(hamk.dot(self.TAPW_parameters.g_matrix.conj().T))
        if not self.structure.spin:
            print("spin true")
            M = self.TAPW_parameters.g_matrix.shape[0] // 2
            N = self.TAPW_parameters.g_matrix.shape[1] // 2
            gr_matrix = self.TAPW_parameters.g_matrix[:M, :N]
            gr_matrix_conj = self.TAPW_parameters.g_matrix_conj[:N, :M]
            hamk_11 = hamk[:N, :N]
            hamk_12 = hamk[:N, N:]
            # hamk_21 = hamk[N:, :N]
            hamk_22 = hamk[N:, N:]
            time1 = time.time()
            GHG_11 = gr_matrix @ hamk_11 @ gr_matrix_conj
            time2 = time.time()
            print("time1 = ",time2-time1)
            GHG_12 = gr_matrix @ hamk_12 @ gr_matrix_conj
            time3 = time.time()
            print("time2 = ",time3-time2)
            GHG_21 = GHG_12.conj().T
            time4 = time.time()
            print("time3 = ",time4-time3)
            GHG_22 = gr_matrix @ hamk_22 @ gr_matrix_conj
            time5 = time.time()
            print("time4 = ",time5-time4)
            result = scipy.sparse.bmat([[GHG_11, GHG_12], [GHG_21, GHG_22]]).toarray() 
            time6 = time.time()
            print("time5 = ",time6-time5)
                    
        else:
            result = (self.TAPW_parameters.g_matrix @ hamk).tocsc() @ self.TAPW_parameters.g_matrix_conj
            result = result.toarray()
        return result

    def cal_TAPW_hamiltonian_k_gpu(self, hamk):
        # Ensure hamk and g_matrix are Cupy arrays
        # if not isinstance(hamk, cp.ndarray):
        #     hamk_gpu = cp.asarray(hamk.toarray())
        #     print("hamk_gpu = ", hamk_gpu)
        # else:
        #     hamk_gpu = hamk
        hamk_gpu = hamk
        g_matrix_gpu = self.TAPW_parameters.g_matrix
        # if not isinstance(self.TAPW_parameters.g_matrix, cp.ndarray):
        #     g_matrix_gpu = cp.asarray(self.TAPW_parameters.g_matrix.toarray())
        # else:
        #     g_matrix_gpu = self.TAPW_parameters.g_matrix
        
        # print("type and shape of hamk_gpu = ", type(hamk_gpu), hamk_gpu.shape)
        # print("type and shape of g_matrix_gpu = ", type(g_matrix_gpu), g_matrix_gpu.shape)
        
        # Ensure g_matrix_gpu is a sparse matrix if needed
        # if isinstance(g_matrix_gpu, cp.sparse.csr_matrix):
        # print("type of hamk_gpu = ", type(hamk_gpu.conj().T))
        temp1 = hamk_gpu @ cp.sparse.csr_matrix(g_matrix_gpu.conj().T)
        # print("type and shape of temp1 = ", type(temp1), temp1.shape)
        temp1 = cp.sparse.csr_matrix(temp1)
        g_matrix_gpu = cp.sparse.csr_matrix(g_matrix_gpu)
        result_gpu = g_matrix_gpu @ temp1
        # else:
        #     result_gpu = cp.dot(g_matrix_gpu, cp.dot(hamk_gpu, g_matrix_gpu.conj().T))
        
        # Optionally convert back to scipy sparse matrix
        # result = scipy.sparse.csr_matrix(cp.asnumpy(result_gpu))
        
        return result_gpu

    @timing_decorator_factory(process_id=0)
    def calculate_band_01(self, kpoints,i):
        def print_time(stage):
            current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            print(f"{stage} Current Time: {current_time}")
            sys.stdout.flush()
        sys.stdout.flush() 
        if self.config.TAPW:
            if self.config.C3_H:
                # if not np.allclose((self.TAPW_parameters.C3_matrix@self.TAPW_parameters.C3_matrix).toarray(),self.TAPW_parameters.C3_matrix.conj().T.toarray()):
                #     raise ValueError("C3_matrix is not hermitian")
                sys.stdout.flush() 
                hamk,samk = self.Getk_super_gauge_sparse_symm_final_HS(self.hr_supercell, self.sr_supercell, self.TAPW_parameters.symm_matrix, self.TAPW_parameters.symm_matrix_inv, kpoints[:3],self.config.gpu_index[i%self.config.gpu_num])
                # hamk_list, samk_list = self.Getk_super_gauge_sparse_C3(self.hr_supercell, kpoints[:3]), self.Getk_super_gauge_sparse_C3(self.sr_supercell, kpoints[:3])
                # hamk_list = [self.cal_TAPW_hamiltonian_k(hamk_list[i]) for i in range(3)]
                # samk_list = [self.cal_TAPW_hamiltonian_k(samk_list[i]) for i in range(3)]
                # print("samk list type,shape = ",type(samk_list[0]),samk_list[0].shape)
                # if self.config.ge:
                #     hamk, samk = self.C3_symm(hamk_list[0], hamk_list[1], hamk_list[2], self.TAPW_parameters.C3_matrix), self.C3_symm(samk_list[0], samk_list[1], samk_list[2], self.TAPW_parameters.C3_matrix)
                # else:
                #     hamk = self.C3_symm(*[self.gen_H_new(hamk_list[i].toarray(), samk_list[i].toarray(),self.config.gpu_index[i%self.config.gpu_num]) for i in range(3)], self.TAPW_parameters.C3_matrix)
            else:
                sys.stdout.flush() 
                # hamk,samk = self.Getk_super_gauge_sparse(self.hr_supercell, kpoints[:3]), self.Getk_super_gauge_sparse(self.sr_supercell, kpoints[:3])
                # hamk, samk = self.cal_TAPW_hamiltonian_k(hamk), self.cal_TAPW_hamiltonian_k(samk)
                # if not self.config.ge:
                #     hamk = self.gen_H_new(hamk.toarray(),samk.toarray())
                hamk,samk = self.Getk_super_gauge_sparse_final_HS(self.hr_supercell, self.sr_supercell, kpoints[:3],self.config.gpu_index[i%self.config.gpu_num])
            
            if self.config.eigsh_cal:
                if self.config.ge:
                    # if self.config.gpu:
                    #     hamk_gpu = cp.asarray(hamk)
                    #     samk_gpu = cp.asarray(samk)
                    #     w = cp.linalg.eigh(hamk_gpu, k=self.config.num_bands_cal, M=samk_gpu, sigma=self.config.efermi, which='LM', return_eigenvectors=self.config.eig_vec_cal)
                    #     w = cp.asnumpy(w)
                    #     del hamk_gpu, samk_gpu
                    #     cp.get_default_memory_pool().free_all_blocks()
                    # else:
                    w = eigsh(hamk, k=self.config.num_bands_cal, M=samk, sigma=self.config.efermi, which='LM', return_eigenvectors=self.config.eig_vec_cal)
                else:
                    # if self.config.gpu:
                    #     hamk_gpu = cp.asarray(hamk)
                    #     w = cp.linalg.eigh(hamk_gpu, k=self.config.num_bands_cal, sigma=self.config.efermi, which='LM', return_eigenvectors=self.config.eig_vec_cal)
                    #     w = cp.asnumpy(w)
                    #     del hamk_gpu
                    #     cp.get_default_memory_pool().free_all_blocks()

                    # else:
                    w = eigsh(hamk, k=self.config.num_bands_cal, sigma=self.config.efermi, which='LM', return_eigenvectors=self.config.eig_vec_cal)
            else:
                w = lapack.zhegv(hamk, samk, itype=1, jobz='V' if self.config.eig_vec_cal else 'N')
        else:
            if self.config.eigsh_cal:
                if self.config.ge:
                    hamk = self.Getk_super_gauge_sparse(self.hr_supercell, kpoints[:3])
                    samk = self.Getk_super_gauge_sparse(self.sr_supercell, kpoints[:3])
                    w = eigsh(hamk, k=self.config.num_bands_cal, M=samk, sigma=self.config.efermi, which='LM', return_eigenvectors=self.config.eig_vec_cal)
                else:
                    # w = eigsh(self.hr_supercell, k=self.config.num_bands_cal, sigma=self.config.efermi, which='LM', return_eigenvectors=self.config.eig_vec_cal)
                    raise ValueError("Not implemented! Recommend to use generalized eigenvalue solver.")
        
        if self.config.eig_vec_cal:
            eig = np.sort(np.real(w[0]))
            vec = w[1][:, np.argsort(np.real(w[0]))]
            if not self.config.hamk_save:
                hamk,samk = 0,0
        else:
            eig = np.sort(np.real(w))
            vec = 0


        return eig, vec, hamk ,samk

    @timing_decorator_factory(process_id=0)
    def parallel_calculate_band_01(self, kpoints):
        

        start_time = time.time()
        current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        sys.stdout.flush() 
        print(f"Current Time: {current_time}")
        print(f"Running with num_processes = {self.config.num_processes}")
                
        def print_time(stage):
            current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            print(f"{stage} Current Time: {current_time}")
            sys.stdout.flush()

        def delayed_calculate_band_01(kpoint, delay):
            time.sleep(delay*self.config.delay_time)
            result = self.calculate_band_01(kpoint,delay)
            print(f"=============================     Kpoint {delay} {kpoint} finished    =============================")
            sys.stdout.flush() 
            return result

        delays = [i for i in range(len(kpoints))]
        # self.calculate_band_01(kpoints[0],0)
        # exit()
        tasks = [delayed(delayed_calculate_band_01)(kpoint, delay) for kpoint, delay in tqdm(zip(kpoints, delays))]
        result = Parallel(n_jobs=self.config.num_processes)(tasks)

        eig, vec, hamk, samk = zip(*result)# if self.config.eig_vec_cal else (result, None, None, None)

        self.result['eig'] = np.array(eig)
        self.result['vec'] = np.array(vec)
        self.result['hamk'] = hamk
        self.result['samk'] = samk

        end_time = time.time()

        print(f"Running time: {end_time - start_time:.2f} seconds")
        current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        print(f"Current Time: {current_time}")

     
    def write_wave_function_spin(self, path):
        """
        Write the wave function for all k-point and each spin to files.
        """


        # num_atom = [self.structure.atomic_species_data[atom]["atom_num"] for atom in self.structure.atomic_species_data.keys()]
        # atom_M = self.structure.M_orb

        num_Te = self.structure.X_orb
        num_Mo = self.structure.M_orb
        n_g = self.config.n_g
        valley_flag = self.config.valley_flag
        # num_Te = atoms_species["S"]["orb_num"]
        # num_Mo = atoms_species["W"]["orb_num"]
        g_vec_list_K_1layer = np.load(path+f"/g_vec_list_{n_g}_{valley_flag}_1layer.npy")
        g_vec_list_K_2layer = np.load(path+f"/g_vec_list_{n_g}_{valley_flag}_2layer.npy")

        orb_num = self.structure.num_orbs_per_unit_cell
        # band_wave = np.load(path + f'/band_{valley_flag}_shell_{n_g}_{self.config.eq_flag}_{self.config.symm_flag}_{self.config.solve_flag}_{self.config.num_bands_cal}_{self.config.end_flag}_vec.npy')
        band_wave = self.result['vec']
        print(np.shape(band_wave))
        num_kpoints = len(self.kpath_config.kpoints)
        valley_flag = self.config.valley_flag
        # if not os.path.exists(path+f'/{valley_flag}_valley'):
        #     os.mkdir(path+f'/{valley_flag}_valley')
        dir = os.path.join(path, f'{valley_flag}_valley')
        os.makedirs(dir, exist_ok=True)
        num_gn_all = len(g_vec_list_K_1layer)*2
        up_all_index, down_all_index = self.generate_indices(num_gn_all, num_Te, num_Mo, orb_num)
        print(up_all_index,down_all_index,band_wave.shape)
        np.save(os.path.join(dir, f'{valley_flag}_valley_up.npy'), band_wave[:,up_all_index])
        np.save(os.path.join(dir, f'{valley_flag}_valley_down.npy'), band_wave[:,down_all_index])
        # for K_index in range(num_kpoints):

        #     dir = os.path.join(path, f'{valley_flag}_valley', f'{valley_flag}_valley_kpoint_{K_index+1}')
        #     os.makedirs(dir, exist_ok=True)
        #     for i in range(self.config.num_bands_cal):
        #         band_index = i+1
        #         # output_wave_path = path+ f'/K1_valley/{valley_flag}_valley_kpoint_{K_index+1}/band_{band_index}_wave.txt'
        #         output_wave_path = os.path.join(dir, f'band_{band_index}_wave.txt')
        #         self.write_wave_2col(output_wave_path,band_wave[K_index,:,-band_index],g_vec_list_K_1layer,g_vec_list_K_2layer,orb_Mo=num_Mo,orb_Te=num_Te)
        
    
    def direct_sum(self, *matrices):
        """
        计算多个矩阵的直和
        
        参数：
        matrices: 一个矩阵列表
        
        返回值：
        直和矩阵
        """
        # 计算直和矩阵的形状
        shape_sum = np.sum([matrix.shape for matrix in matrices], axis=0)
        
        # 构造直和矩阵
        direct_sum_matrix = np.zeros(shape_sum,dtype=np.complex128)
        row_start = 0
        col_start = 0
        for matrix in matrices:
            rows, cols = matrix.shape
            direct_sum_matrix[row_start:row_start+rows, col_start:col_start+cols] = matrix
            row_start += rows
            col_start += cols

        return direct_sum_matrix

    def generate_indices(self, num_gn_all, num_Te, num_Mo, orbs_num):
        print(num_gn_all, num_Te, num_Mo, orbs_num)
        up_index = np.concatenate((np.arange(num_Te), np.arange(num_Te) + num_Te * 2, np.arange(num_Mo) + num_Te * 4))
        down_index = np.concatenate((np.arange(num_Te) + num_Te, np.arange(num_Te) + num_Te * 3, np.arange(num_Mo) + num_Te * 4 + num_Mo))

        up_all_index = np.concatenate([up_index + orbs_num * 2 * i for i in range(num_gn_all)])
        down_all_index = np.concatenate([down_index + orbs_num * 2 * i for i in range(num_gn_all)])

        return up_all_index.astype(int), down_all_index.astype(int)
 
    def write_hamk_spin(self, path):
        """
        Write the Hamiltonian matrix for all k-point and each spin to files.
        """
        hamk = self.result['hamk']
        dim_Hprime = np.shape(hamk[0])[0]

        num_Te = self.structure.X_orb
        num_Mo = self.structure.M_orb
        n_g = self.config.n_g
        valley_flag = self.config.valley_flag
        orb_num = self.structure.num_orbs_per_unit_cell
        g_vec_list_K_1layer = np.load(path+f"/g_vec_list_{n_g}_{valley_flag}_1layer.npy")
        num_gn_all = len(g_vec_list_K_1layer)*2 
        up_all_index, down_all_index = self.generate_indices(num_gn_all, num_Te, num_Mo, orb_num)
        num_kpoints = len(self.kpath_config.kpoints)
        H_spin_kpoints = np.zeros((2,num_kpoints,int(dim_Hprime/2),int(dim_Hprime/2)),dtype=np.complex64)
        for i in tqdm(range(num_kpoints)):
            K_index = i
            gamma_hamk = hamk[i]

            H_gamma_up = gamma_hamk[up_all_index][:, up_all_index]
            H_gamma_down = gamma_hamk[down_all_index][:, down_all_index]
            

            H_spin_kpoints[0,i] = H_gamma_up
            H_spin_kpoints[1,i] = H_gamma_down

            # if not os.path.exists(path+f"/symm_Hprime_wave"):
            #     os.mkdir(path+f"/symm_Hprime_wave")
            # if not os.path.exists(path+f"/symm_Hprime_wave/H_prime_kpoint_{K_index+1}"):
            #     os.mkdir(path+f"/symm_Hprime_wave/H_prime_kpoint_{K_index+1}")
            # # np.savetxt(path+f"/symm_Hprime_wave/H_prime_kpoint_{K_index+1}/Hprime_up_real.txt",H_gamma_up.real*hatree,fmt='%15.11f')
            # # np.savetxt(path+f"/symm_Hprime_wave/H_prime_kpoint_{K_index+1}/Hprime_up_imag.txt",H_gamma_up.imag*hatree,fmt='%15.11f')
            # np.savetxt(path+f"/symm_Hprime_wave/H_prime_kpoint_{K_index+1}/Hprime_down_real.txt",H_gamma_down.real*hatree,fmt='%15.11f')
            # np.savetxt(path+f"/symm_Hprime_wave/H_prime_kpoint_{K_index+1}/Hprime_down_imag.txt",H_gamma_down.imag*hatree,fmt='%15.11f')
            # np.savetxt(path+f"/symm_Hprime_wave/H_prime_kpoint_{K_index+1}/Hprime_full_real.txt",gamma_hamk.real*hatree,fmt='%15.11f')
            # np.savetxt(path+f"/symm_Hprime_wave/H_prime_kpoint_{K_index+1}/Hprime_full_imag.txt",gamma_hamk.imag*hatree,fmt='%15.11f')
        # if not os.path.exists(path+f"/symm_Hprime_wave_npy"):
        #     os.mkdir(path+f"/symm_Hprime_wave_npy")
        os.makedirs(os.path.join(path, 'symm_Hprime_wave_npy'), exist_ok=True)
        # np.save(path+f"/symm_Hprime_wave_npy/Hprime_up_down.npy",H_spin_kpoints)
        np.save(os.path.join(path, 'symm_Hprime_wave_npy', f'Hprime_up_down_{self.config.valley_flag}.npy'), H_spin_kpoints)
    
    def write_wave_2col(self,path,vec,g_vec_list_1layer,g_vec_list_2layer,orb_Te,orb_Mo):
        num_wann = len(vec)
        num_wann_perlayer = int(num_wann/2)
        f = open(path,"w")
        arr1 = np.arange(1, orb_Te+1)
        arr2 = np.arange(1, orb_Mo+1)
        

        orb_index_num = np.concatenate((arr1, arr1, arr1, arr1, arr2, arr2))
        orb_spin_index = np.concatenate((['up']*orb_Te,['down']*orb_Te,['up']*orb_Te,['down']*orb_Te,['up']*orb_Mo,['down']*orb_Mo))
        atoms_index = np.concatenate((['Te2']*orb_Te*2,['Te1']*orb_Te*2,['Mo']*orb_Mo*2))
        orb_all = orb_Te*4+orb_Mo*2
        f.write("#layer  gvec     gvec_x     gvec_y  atom  orb  spin     real       imag\n")
        for i in range(num_wann):
            if i < num_wann_perlayer:
                g_vec_index = i//orb_all
                g_vec = g_vec_list_1layer[g_vec_index]
                layer = 1
                f.write(f"{layer:>5d} {g_vec_index+1:>5d} {g_vec[0]:>12.6f} {g_vec[1]:>10.6f} {atoms_index[i%orb_all]:>4s} {orb_index_num[i%orb_all]:>4d} {orb_spin_index[i%orb_all]:>5s} {vec[i].real:>10.6f} {vec[i].imag:>10.6f}\n")
            else:
                g_vec_index = (i-int(num_wann_perlayer))//orb_all
                g_vec = g_vec_list_2layer[g_vec_index]
                layer = 2
                f.write(f"{layer:>5d} {g_vec_index+1:>5d} {g_vec[0]:>12.6f} {g_vec[1]:>10.6f} {atoms_index[i%orb_all]:>4s} {orb_index_num[i%orb_all]:>4d} {orb_spin_index[i%orb_all]:>5s} {vec[i].real:>10.6f} {vec[i].imag:>10.6f}\n")