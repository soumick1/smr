"""SE(3) and angle utilities.

Convention (fixed for the whole project, do not change silently):
  * Rotations are world-from-body: x_world = R @ x_body + t.
  * Euler angles are ZYX (yaw psi about z, pitch theta about y, roll phi
    about x):  R = Rz(psi) @ Ry(theta) @ Rx(phi).
  * Body angular velocity omega = (p, q, r) maps to Euler-angle rates by
    `euler_rates` (standard aerospace form; singular at pitch = +-90 deg,
    which the protocol excludes).
"""
from __future__ import annotations

import numpy as np

# --------------------------------------------------------------------- angles
def wrap_pi(a):
    """Wrap angle(s) to [-pi, pi)."""
    return (np.asarray(a) + np.pi) % (2 * np.pi) - np.pi


def wrap_2pi(a):
    """Wrap angle(s) to [0, 2*pi)."""
    return np.asarray(a) % (2 * np.pi)


def ang_dist(a, b):
    """Geodesic distance on the circle."""
    return np.abs(wrap_pi(np.asarray(a) - np.asarray(b)))


# ------------------------------------------------------------------ rotations
def Rx(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def Ry(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def Rz(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def euler_zyx_to_R(yaw, pitch, roll):
    """R = Rz(yaw) @ Ry(pitch) @ Rx(roll)."""
    return Rz(yaw) @ Ry(pitch) @ Rx(roll)


def R_to_euler_zyx(R):
    """Inverse of euler_zyx_to_R.  Returns (yaw, pitch, roll)."""
    pitch = np.arcsin(np.clip(-R[2, 0], -1.0, 1.0))
    if np.cos(pitch) > 1e-8:
        yaw = np.arctan2(R[1, 0], R[0, 0])
        roll = np.arctan2(R[2, 1], R[2, 2])
    else:  # gimbal lock: split arbitrarily, roll := 0
        yaw = np.arctan2(-R[0, 1], R[1, 1])
        roll = 0.0
    return yaw, pitch, roll


def euler_rates(omega_body, roll, pitch):
    """Euler-angle rates (yaw_dot, pitch_dot, roll_dot) from body rates.

    omega_body = (p, q, r).  Singular at |pitch| = pi/2.
    """
    p, q, r = omega_body
    sph, cph = np.sin(roll), np.cos(roll)
    cth, tth = np.cos(pitch), np.tan(pitch)
    roll_dot = p + (q * sph + r * cph) * tth
    pitch_dot = q * cph - r * sph
    yaw_dot = (q * sph + r * cph) / cth
    return yaw_dot, pitch_dot, roll_dot


# ---------------------------------------------------------------------- SE(3)
def make_T(R, t):
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = np.asarray(t, dtype=float)
    return T


def inv_T(T):
    R, t = T[:3, :3], T[:3, 3]
    Ti = np.eye(4)
    Ti[:3, :3] = R.T
    Ti[:3, 3] = -R.T @ t
    return Ti


def compose(Ta, Tb):
    return Ta @ Tb


def relative(T_from, T_to):
    """Delta such that T_to = T_from @ Delta (Delta in `from` body frame)."""
    return inv_T(T_from) @ T_to


def pose_errors(T_est, T_gt):
    """(rotation error in radians, translation error in world units)."""
    dR = T_est[:3, :3].T @ T_gt[:3, :3]
    cos = np.clip((np.trace(dR) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.arccos(cos)), float(np.linalg.norm(T_est[:3, 3] - T_gt[:3, 3]))
