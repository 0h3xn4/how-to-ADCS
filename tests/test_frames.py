import numpy as np
from src.adcs_lab.frames import (julian_date, gmst_rad, eci_to_ecef_matrix, lvlh_from_eci_matrix)

JDMOD = 2451545.0

def test_j2000_epoch():
    assert abs(julian_date(2000, 1, 1, 12) - JDMOD) < 1e-9

def test_eci_ecef_is_a_rotation():
    R = eci_to_ecef_matrix(JDMOD + 100.0)

    # A rotation matrix multiplied by its transposed must be unit matrix and have det = +1
    
    assert np.allclose(R @ R.T, np.eye(3), atol=1e-9)
    assert abs(np.linalg.det(R) - 1.0) < 1e-12

def test_lvlh_axes_orthonormal():
    R = lvlh_from_eci_matrix([7000e3, 0, 0], [0, 7500.0, 0])
    assert np.allclose(R @ R.T, np.eye(3), atol=1e-9)
    assert abs(np.linalg.det(R) - 1.0) < 1e-9

