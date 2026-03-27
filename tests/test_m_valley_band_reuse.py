from pathlib import Path
from types import SimpleNamespace

import numpy as np

from tapw.cal_ham_01 import BandStructureCalculator
from tapw.main import can_reuse_m_valley_c3_band_outputs, copy_reused_m_valley_band_outputs


def test_can_reuse_m_valley_c3_band_outputs_is_disabled_by_default():
    cfg = SimpleNamespace(
        mode="band",
        TAPW=True,
        C3_H=True,
        M_valley_D3_H=False,
        eig_vec_cal=False,
        hamk_save=False,
        valleys=[31, 32, 33],
        bravais="hex",
    )

    assert can_reuse_m_valley_c3_band_outputs(cfg) is False


def test_copy_reused_m_valley_band_outputs_renames_reference_files(tmp_path: Path):
    band_dir = tmp_path / "band"
    band_dir.mkdir()

    source_vbm = band_dir / "band_VBM_M1_valley.txt"
    source_cbm = band_dir / "band_CBM_M1_valley.txt"
    np.savetxt(source_vbm, np.array([[1.0, 2.0], [3.0, 4.0]]))
    np.savetxt(source_cbm, np.array([[5.0, 6.0], [7.0, 8.0]]))

    copy_reused_m_valley_band_outputs(tmp_path, "M1", "M3")

    copied_vbm = band_dir / "band_VBM_M3_valley.txt"
    copied_cbm = band_dir / "band_CBM_M3_valley.txt"
    assert copied_vbm.exists()
    assert copied_cbm.exists()
    assert np.allclose(np.loadtxt(copied_vbm), np.loadtxt(source_vbm))
    assert np.allclose(np.loadtxt(copied_cbm), np.loadtxt(source_cbm))


def test_save_band_static_metadata_writes_valley_specific_g_vectors(tmp_path: Path):
    calculator = BandStructureCalculator.__new__(BandStructureCalculator)
    calculator.config = SimpleNamespace(TAPW=True, n_g=4)
    calculator.valley_flag = "M2"
    calculator.use_C3_H = False
    calculator.TAPW_parameters = SimpleNamespace(
        g_vec_list_K1=np.array([[1.0, 2.0]], dtype=float),
        g_vec_list_K2=np.array([[3.0, 4.0]], dtype=float),
    )

    calculator.save_band_static_metadata(str(tmp_path))

    g1 = tmp_path / "g_vec_list_4_M2_1layer.npy"
    g2 = tmp_path / "g_vec_list_4_M2_2layer.npy"
    assert g1.exists()
    assert g2.exists()
    assert np.allclose(np.load(g1), calculator.TAPW_parameters.g_vec_list_K1)
    assert np.allclose(np.load(g2), calculator.TAPW_parameters.g_vec_list_K2)
