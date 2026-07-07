"""tek-adım step() smoke + boyut/sınır kontrolleri + warm-start (spec §12)."""
import numpy as np
import pytest

import cumpc_core as cc


def straight_ref(N=64, ds=0.05, x0=0.0):
    xs = x0 + np.arange(N, dtype=np.float32) * ds
    return np.stack([xs, np.zeros(N, np.float32), np.zeros(N, np.float32)], axis=1)


def test_step_shapes_and_limits(default_cfg):
    c = cc.MPPIController(default_cfg)
    u = c.step(np.zeros(3, np.float32), straight_ref())
    assert u.shape == (2,)
    assert np.all(np.isfinite(u))
    # control limit aşımı yok (örnekler clamp'li → ağırlıklı ortalama da sınır içinde)
    assert abs(u[0]) <= default_cfg.robot.v_max + 1e-5
    assert abs(u[1]) <= default_cfg.robot.omega_max + 1e-5
    assert c.nominal().shape == (default_cfg.H, 2)
    assert c.last_rollouts().shape == (default_cfg.K, default_cfg.H, 3)


def test_nominal_within_limits(default_cfg):
    c = cc.MPPIController(default_cfg)
    x = np.zeros(3, np.float32)
    for _ in range(20):
        c.step(x, straight_ref())
    nom = c.nominal()
    assert np.all(np.abs(nom[:, 0]) <= default_cfg.robot.v_max + 1e-5)
    assert np.all(np.abs(nom[:, 1]) <= default_cfg.robot.omega_max + 1e-5)
    assert np.all(np.isfinite(nom))


def test_converges_forward_on_straight_path(default_cfg):
    """İleri uzanan düz path → kapalı döngüde pozitif v'ye yakınsama (progress ödülü).

    lambda=50 ile ağırlıklar yayık → rampa kasıtlı olarak yumuşaktır; 40 adım
    (2 s) içinde belirgin ileri hız bekleriz.
    """
    c = cc.MPPIController(default_cfg)
    x = np.zeros(3, np.float32)
    vs = []
    for _ in range(40):
        u = c.step(x, straight_ref(x0=float(x[0])))
        vs.append(float(u[0]))
        # state'i rollout modeliyle ilerlet (kapalı döngü benzetimi)
        x = np.array([x[0] + u[0] * default_cfg.dt * np.cos(x[2]),
                      x[1] + u[0] * default_cfg.dt * np.sin(x[2]),
                      x[2]], dtype=np.float32)
    assert np.mean(vs[-10:]) > 0.1, f"ileri progress bekleniyordu, son v'ler={vs[-5:]}"


def test_warm_start_shift(default_cfg):
    """step sonrası nominal, warm-start shift'i yansıtmalı: son iki satır eşit."""
    c = cc.MPPIController(default_cfg)
    c.step(np.zeros(3, np.float32), straight_ref())
    nom = c.nominal()
    assert np.array_equal(nom[-1], nom[-2])


def test_reset_zeroes_nominal(default_cfg):
    c = cc.MPPIController(default_cfg)
    for _ in range(3):
        c.step(np.zeros(3, np.float32), straight_ref())
    c.reset()
    assert np.array_equal(c.nominal(), np.zeros((default_cfg.H, 2), np.float32))


def test_invalid_shapes_raise(default_cfg):
    c = cc.MPPIController(default_cfg)
    with pytest.raises(Exception):
        c.step(np.zeros(2, np.float32), straight_ref())
    with pytest.raises(Exception):
        c.step(np.zeros(3, np.float32), np.zeros((5, 2), np.float32))
