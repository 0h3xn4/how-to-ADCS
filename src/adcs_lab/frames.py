# src/adcs_lab/frames.py
import numpy as np

def julian_date(year, month, day, hour=0, minute=0, second=0.0):
    # Valid for years 1900-2100. Standard civil-date to JD conversion

    if month <= 2:
        year -= 1
        month += 12
    A = int(year / 100)
    B = 2 - A + int(A / 4)
    day_frac = ((((second/60.0) + minute) / 60.0) + hour) / 24.0
    jd = (int(365.25 * (year + 4716)) + int(30.6001 * (month + 1)) + day + B - 1524.5 + day_frac)      # This comes from Fundamentals of Astrodynamics and Application, by Vallado, see page 183
    return jd

def gmst_rad(jd):
    # Greenwich Mean Sidereal Time in radians (IAU 1982 approximation)

    T = (jd - 2451545.0) / 36525.0                                                          # Julian centuries since J2000
    gmst_sec = (67310.54841                                                                 # This comes from Fundamentals of Astrodynamics and Application, by Vallado, see page 188 (UT1 better for software)
                + (876600.0 * 3600.0 + 8640184.812866) * T                  
                + 0.093104 * T**2
                - 6.2e-6 * T**3)                                                            # seconds of time
    gmst = (gmst_sec % 86400.0) / 86400.0 * 2.0 * np.pi                                     # converts to from sec to RAD
    return gmst % (2.0 * np.pi)                                                             

def rot_z(angle_rad):
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    return np.array([[c, s, 0],
                     [-s, c, 0], 
                     [0, 0, 1]])

def eci_to_ecef_matrix(jd):
    # Rotation matrix R such that v_ecef = Rot @ v_eci

    return rot_z(gmst_rad(jd))

def lvlh_from_eci_matrix(r_eci, v_eci):
    # Returns R such that v_lvlh = Rot @ v_eci
    # r, v are intertial position and velocity
    # This comes from Fundamentals of attitude determination and control, by Markley and Crassidis, see page 36

    r = np.asarray(r_eci, float)
    v = np.asarray(v_eci, float)
    o3 = -r / np.linalg.norm(r)     # nadir
    h = np.cross(r,v)               
    o2 = -h / np.linalg.norm(h)     # negative orbit normal
    o1 = np.cross(o2,o3)            # roughly along-track, not exactly
    o1 = o1 / np.linalg.norm(o1)
    return np.vstack([o1, o2, o3])