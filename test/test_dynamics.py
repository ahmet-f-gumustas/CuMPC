"""exact-arc doğrulama: analitik referansa karşı (spec §12).

simulate_controls production __device__ diff_drive_step'i birebir çağırır.
"""
import numpy as np
import pytest

import cumpc_core as cc

DT = 0.05


def const_controls(v, omega, H):
    return np.tile(np.array([v, omega], dtype=np.float32), (H, 1))


def test_straight_line():
    H = 100
    s = cc.simulate_controls(np.zeros(3, np.float32), const_controls(1.0, 0.0, H), dt=DT)
    t = np.arange(1, H + 1) * DT
    assert np.allclose(s[:, 0], 1.0 * t, atol=1e-5)
    assert np.allclose(s[:, 1], 0.0, atol=1e-6)
    assert np.allclose(s[:, 2], 0.0, atol=1e-6)


def test_straight_line_with_heading():
    H = 50
    th0 = 0.7
    x0 = np.array([1.0, -2.0, th0], dtype=np.float32)
    s = cc.simulate_controls(x0, const_controls(0.8, 0.0, H), dt=DT)
    t = np.arange(1, H + 1) * DT
    assert np.allclose(s[:, 0], 1.0 + 0.8 * t * np.cos(th0), atol=1e-5)
    assert np.allclose(s[:, 1], -2.0 + 0.8 * t * np.sin(th0), atol=1e-5)


def test_constant_radius_circle():
    """v=1, omega=0.5 → R=2 çember; analitik: x=R sin(θ), y=R(1-cos(θ))."""
    H = 200
    v, w = 1.0, 0.5
    R = v / w
    s = cc.simulate_controls(np.zeros(3, np.float32), const_controls(v, w, H), dt=DT)
    th = w * np.arange(1, H + 1) * DT
    assert np.allclose(s[:, 0], R * np.sin(th), atol=1e-4)
    assert np.allclose(s[:, 1], R * (1.0 - np.cos(th)), atol=1e-4)
    assert np.allclose(s[:, 2], th, atol=1e-4)
    # EXACT ARC: çember üzerinde kalır (Euler dışa savrulurdu)
    r_err = np.abs(np.hypot(s[:, 0], s[:, 1] - R) - R)
    assert r_err.max() < 1e-4


def test_omega_to_zero_limit():
    """omega→0 yay kolu düz-çizgi koluna sürekli bağlanmalı."""
    H = 50
    s_zero = cc.simulate_controls(np.zeros(3, np.float32), const_controls(1.0, 0.0, H), dt=DT)
    s_eps = cc.simulate_controls(np.zeros(3, np.float32), const_controls(1.0, 2e-5, H), dt=DT)
    assert np.allclose(s_zero[:, :2], s_eps[:, :2], atol=1e-3)


def test_slip_identity_matches_no_slip():
    H = 80
    U = const_controls(1.2, 0.7, H)
    s1 = cc.simulate_controls(np.zeros(3, np.float32), U, dt=DT)
    s2 = cc.simulate_controls(np.zeros(3, np.float32), U,
                              kappa_v=1.0, kappa_w=1.0, v_y=0.0, dt=DT)
    assert np.array_equal(s1, s2)


def test_slip_active():
    """kappa_v < 1 → daha kısa yol; v_y > 0 → sola drift."""
    H = 100
    U = const_controls(1.0, 0.0, H)
    s_slip = cc.simulate_controls(np.zeros(3, np.float32), U, kappa_v=0.8, dt=DT)
    t_end = H * DT
    assert np.isclose(s_slip[-1, 0], 0.8 * t_end, atol=1e-4)

    s_drift = cc.simulate_controls(np.zeros(3, np.float32), U, v_y=0.2, dt=DT)
    # heading=0 iken +v_y sola (+y) taşır
    assert np.isclose(s_drift[-1, 1], 0.2 * t_end, atol=1e-4)

    s_kw = cc.simulate_controls(np.zeros(3, np.float32),
                                const_controls(1.0, 1.0, H), kappa_w=0.5, dt=DT)
    assert np.isclose(s_kw[-1, 2], 0.5 * H * DT, atol=1e-4)
