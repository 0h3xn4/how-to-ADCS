# src/adcs_lab/attitude.py

import numpy as np

def quat_normalize(q):
    q = np.asarray(q, float)
    return q / np.linalg.norm(q)

def quat_to_dcm(q):
    # Inertial-to-body passive rotation matrix A: v_body = A @ v_inertial
    # Scalar-last: q = [q1, q2, q3, q4], q4 is the scalar part - we use the convention from Markley and Crassidis
    # See page 45 of Fundamentals of attitude determination and control, by markley and crassidis

    q1, q2, q3, q4 = quat_normalize(q)
    return np.array([
        [q1**2 - q2**2 - q3**2 + q4**2, 2 * (q1*q2 + q3*q4), 2*(q1*q3 - q2*q4)],
        [2*(q2*q1 - q3*q4), -q1**2 + q2**2 - q3**2 + q4**2, 2*(q2*q3 + q1*q4)],
        [2*(q3*q1 + q2*q4), q*(q3*q2 - q1*q4), -q1**2 - q2**2 + q3**3 + q4**2],                                        
    ])