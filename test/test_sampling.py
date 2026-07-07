"""noise istatistiği (mean/std), clamp sınırları, sabit-seed determinizm (spec §12).

Örneklenen U doğrudan görünür olmadığından istatistik, tek-adım davranış ve
last_rollouts/nominal üzerinden; determinizm aynı seed'li iki controller'ın
birebir aynı çıktı üretmesiyle doğrulanır.
"""
import numpy as np
import pytest

import cumpc_core as cc


def straight_ref(N=64, ds=0.05):
    xs = np.arange(N, dtype=np.float32) * ds
    return np.stack([xs, np.zeros(N, np.float32), np.zeros(N, np.float32)], axis=1)


def make_ctrl(default_cfg, seed=None):
    if seed is not None:
        default_cfg.seed = seed
    return cc.MPPIController(default_cfg)


def test_deterministic_same_seed(default_cfg):
    ref = straight_ref()
    x = np.zeros(3, np.float32)
    c1 = make_ctrl(default_cfg, seed=999)
    c2 = make_ctrl(default_cfg, seed=999)
    for _ in range(5):
        u1 = c1.step(x, ref)
        u2 = c2.step(x, ref)
        assert np.array_equal(u1, u2)
    assert np.array_equal(c1.nominal(), c2.nominal())
    assert np.array_equal(c1.last_rollouts(), c2.last_rollouts())


def test_reset_restores_determinism(default_cfg):
    ref = straight_ref()
    x = np.zeros(3, np.float32)
    c = make_ctrl(default_cfg, seed=42)
    seq1 = [c.step(x, ref).copy() for _ in range(3)]
    c.reset()
    seq2 = [c.step(x, ref).copy() for _ in range(3)]
    for a, b in zip(seq1, seq2):
        assert np.array_equal(a, b)


def test_different_seed_differs(default_cfg):
    ref = straight_ref()
    x = np.zeros(3, np.float32)
    u1 = make_ctrl(default_cfg, seed=1).step(x, ref)
    u2 = make_ctrl(default_cfg, seed=2).step(x, ref)
    assert not np.array_equal(u1, u2)


def test_noise_statistics_via_rollouts(default_cfg):
    """U_nom=0'dan tek adım: rollout ilk-adım state'lerinden örnek istatistiği.

    px1 ≈ v0·dt (theta=0, tek adım) → v örneklerinin std'si ~ sigma_v,
    theta1 = omega0·dt → omega std ~ sigma_omega. Clamp etkisi küçük
    (sigma << limit) → %10 tolerans.
    """
    cfg = default_cfg
    cfg.seed = 1234
    c = cc.MPPIController(cfg)
    c.step(np.zeros(3, np.float32), straight_ref())
    roll = c.last_rollouts()           # (K,H,3)
    v0 = roll[:, 0, 0] / cfg.dt        # ilk adım: px1 = v·dt·cos(0)
    w0 = roll[:, 0, 2] / cfg.dt        # theta1 = omega·dt
    assert abs(np.mean(v0)) < 3 * cfg.sigma_v / np.sqrt(cfg.K)
    assert abs(np.std(v0) - cfg.sigma_v) < 0.1 * cfg.sigma_v
    assert abs(np.std(w0) - cfg.sigma_omega) < 0.1 * cfg.sigma_omega


def test_clamp_limits(default_cfg):
    """Limitlerin çok üstünde sigma → tüm örnekler [-max, max] içinde kalmalı."""
    cfg = default_cfg
    cfg.sigma_v = 50.0
    cfg.sigma_omega = 50.0
    c = cc.MPPIController(cfg)
    c.step(np.zeros(3, np.float32), straight_ref())
    roll = c.last_rollouts()
    v0 = roll[:, 0, 0] / cfg.dt
    w0 = roll[:, 0, 2] / cfg.dt
    assert np.all(np.abs(v0) <= cfg.robot.v_max + 1e-4)
    assert np.all(np.abs(w0) <= cfg.robot.omega_max + 1e-4)
    # clamp gerçekten devrede: kütle sınırlarda birikmeli
    assert np.mean(np.abs(v0) > 0.99 * cfg.robot.v_max) > 0.5
