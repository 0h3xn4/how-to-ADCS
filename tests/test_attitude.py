import numpy as np
from scipy.spatial.transform import Rotation as Rot
from src.adcs_lab.attitude import (quat_to_dcm, dcm_to_quat, quat_mul, 
                                   quat_conjugate, axis_angle_to_quat, quat_normalize)

def test_quat_dcm_round_trip():
    rng = np.random.default_rng(0)

    for _ in range(200):
        q = quat_normalize(rng.standard_normal(4))
        if q[3] < 0: q = -q
        q2 = dcm_to_quat(quat_to_dcm(q))
        assert np.allclose(q, q2, atol=1e-9)

def test_known_90deg_about_z():
    # The body is rotated +90deg about z, relative to inertial
    # Under passive (inertial->body) convention, an intertial +x vector is therefore
    # expressed in body coordinates as pointing along -y: the body axes have turned +90deg
    # so the fixed intertial vectors appears to turn -90deg

    q = axis_angle_to_quat([0, 0, 1.0], np.pi/2)
    A_mat = quat_to_dcm(q)
    v_body = A_mat @ np.array([1.0, 0, 0])

    assert np.allclose(v_body, [0, -1, 0], atol=1e-12)

def test_matches_scipy():
    # scipy is also scalar-last, so no reordering needed
    # but its active by default, so our passive DCM equals scipys transpose

    q = axis_angle_to_quat([0, 1, 0], 0.7)      
    A_ours = quat_to_dcm(q)
    A_scipys = Rot.from_quat(q).as_matrix()
    assert np.allclose(A_ours, A_scipys.T, atol=1e-12)

def test_composition():
    # Our quat product matches matrix order, see page 47 of Fundamentals of attitude determination and control, by markley and crassidis
    qa = axis_angle_to_quat([0.3, 0.5, -0.2], 0.4)
    qb = axis_angle_to_quat([-0.1, 0.7, 0.3], 0.9)

    assert np.allclose(quat_to_dcm(quat_mul(qb, qa)), quat_to_dcm(qb) @ quat_to_dcm(qa), atol=1e-10)