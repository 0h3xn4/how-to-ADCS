# src/adcs_lab/attitude.py

import numpy as np
import random
import math

def quat_normalize(q):
    q = np.asarray(q, float)
    return q / np.linalg.norm(q)

def quat_to_dcm(q):
    # Inertial-to-body passive rotation matrix A: v_body = A @ v_inertial
    # Scalar-last: q = [q1, q2, q3, q4], q4 is the scalar part - we use the convention from Markley and Crassidis
    # See page 45 of Fundamentals of attitude determination and control, by markley and crassidis

    q1, q2, q3, q4 = quat_normalize(q)
    return np.array([
        [q1*q1 - q2*q2 - q3*q3 + q4*q4, 2 * (q1*q2 + q3*q4), 2*(q1*q3 - q2*q4)],
        [2*(q2*q1 - q3*q4), -q1*q1 + q2*q2 - q3*q3 + q4*q4, 2*(q2*q3 + q1*q4)],
        [2*(q3*q1 + q2*q4), 2*(q3*q2 - q1*q4), -q1*q1 - q2*q2 + q3*q3 + q4*q4],                                        
    ])

def dcm_to_quat(A):
    # Robust extraction (SHepperd's method) consistent with quat_to_decm above
    # scalar part is the last element, q[3]
    # See page 48 of Fundamentals of attitude determination and control, by markley and crassidis

    A = np.asarray(A, float)
    tr = np.trace(A)
    candidates = [1.0 + 2*A[0,0] - tr, 1.0 + 2*A[1,1] - tr, 1.0 + 2*A[2,2] - tr, 1 + tr]        # last one is scalar

    k = int(np.argmax(candidates))
    q = np.zeros(4)

    if k == 3:
        q[3] = 0.5*np.sqrt(candidates[3])
        q[0] = (A[1,2] - A[2,1]) / (4*q[3])
        q[1] = (A[2,0] - A[0,2]) / (4*q[3])
        q[2] = (A[0,1] - A[1,0]) / (4*q[3])
    elif k == 0:
        q[0] = 0.5*np.sqrt(candidates[0])
        q[3] = (A[1,2] - A[2,1]) / (4*q[0])
        q[1] = (A[0,1] + A[1,0]) / (4*q[0])
        q[2] = (A[2,0] + A[0,2]) / (4*q[0])       
    elif k == 1:
        q[1] = 0.5*np.sqrt(candidates[1])
        q[3] = (A[2,0] - A[0,2]) / (4*q[1])
        q[0] = (A[0,1] + A[1,0]) / (4*q[1])
        q[2] = (A[1,2] + A[2,1]) / (4*q[1])          
    else:
        q[2] = 0.5*np.sqrt(candidates[2])
        q[3] = (A[0,1] - A[1,0]) / (4*q[2])
        q[0] = (A[2,0] + A[0,2]) / (4*q[2])
        q[1] = (A[1,2] + A[2,1]) / (4*q[2])  
    
    if q[3] < 0:
        q = -q                              # convention: keep scalar part positive

    return quat_normalize(q)

def quat_mul(p, q):
    # Composing two rotations corresponds to multiplying their quaternions
    # We are working with scalar-last convention
    # See page 37 of Fundamentals of attitude determination and control, by markley and crassidis

    pv, ps = np.asarray(p[:3], float), p[3]
    qv, qs = np.asarray(q[:3], float), q[3]

    v = ps*qv + qs*pv - np.cross(pv, qv)      # minus cross prodcut because of markley crassidis convention
    s = ps*qs - pv.dot(qv)

    return np.array([v[0], v[1], v[2], s])

def quat_conjugate(q):
    return np.array([-q[0], -q[1], -q[2], q[3]])

def euler321_to_dcm(yaw, pitch, roll):
    # Passive 3-2-1: rotate about z (yaw), then y (pitch), then x (roll)
    cz, sz = np.cos(yaw), np.sin(yaw)
    cy, sy = np.cos(pitch), np.sin(pitch)
    cx, sx = np.cos(roll), np.sin(roll)

    Rz = np.array([[cz, sz, 0], [-sz, cz, 0], [0, 0, 1.0]])
    Ry = np.array([[cy, 0, -sy], [0, 1.0, 0], [sy, 0, cy]])
    Rx = np.array([[1.0, 0, 0], [0, cx, sx], [0, -sx, cx]])

    return Rx @ Ry @ Rz

def axis_angle_to_quat(axis, angle):
    # See page 45 of Fundamentals of attitude determination and control, by markley and crassidis
    axis = np.asarray(axis, float)
    axis = axis / np.linalg.norm(axis)
    h = 0.5 * angle
    return np.array([*(np.sin(h) * axis), np.cos(h)])       # again scalar last; the * operator in *(np.sin(h) * axis) is an unpacking operator

def quat_to_axis_angle(q):
    # See page 45 of Fundamentals of attitude determination and control, by markley and crassidis
    q = quat_normalize(q)
    angle = 2.0 * np.arccos(np.clip(q[3], -1.0, 1.0))
    s = np.sqrt(max(1.0 - q[3]*q[3], 0.0))
    axis = q[:3] / s if s > 1e-12 else np.array([1.0, 0, 0])
    return axis, angle