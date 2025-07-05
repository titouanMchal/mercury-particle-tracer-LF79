import numpy as np

RM = 2.4397e6  # Mercury's radius (m)


def cartesian_to_spherical(x: float, y: float, z: float):
    """
    Converts : x, y, z --> r, fi, lat
    """
    r = np.sqrt(x**2 + y**2 + z**2)
    fi = np.arctan2(y, x)
    lat = np.arctan2(z, np.sqrt(x**2 + y**2))
    return r, fi, lat


def spherical_to_cartesian(r: float, fi: float, lat: float):
    """
    converts : r, fi, lat --> x, y, z
    """
    x = r * np.cos(fi) * np.cos(lat)
    y = r * np.sin(fi) * np.cos(lat)
    z = r * np.sin(lat)
    return x, y, z


def vec_ortho(B: np.ndarray):
    """
    Takes B vector and returns 2 orthogonal vectors u1, u2 (u1 in XZ plane)
    """
    Bx = B[0]
    By = B[1]
    Bz = B[2]
    Bxz = np.sqrt(B[0] ** 2 + B[2] ** 2)
    B = np.linalg.norm(B)
    u1 = np.array([Bz / Bxz, 0, -Bx / Bxz])
    u2 = np.array([By * u1[2] / B, Bxz / B, -By * u1[0] / B])
    return u1, u2


def neigbors(u1: np.ndarray, u2: np.ndarray, coord: np.ndarray, dist_vec: float = 5e3):
    """
    Computes two neighbor points at point coord = [x, y, z] (m)
    """
    return coord + dist_vec * u1, coord + dist_vec * u2


def cross(a, b):
    """
    Cross product - looks faster than np.cross
    """
    return np.array(
        [
            a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - b[0] * a[1],
        ]
    )
