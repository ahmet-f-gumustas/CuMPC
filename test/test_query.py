"""ESDF/terrain bilinear query doğruluğu — production-identical device yolu (spec §12)."""
import numpy as np
import pytest

import cumpc_core as cc


def make_ctrl(default_cfg):
    return cc.MPPIController(default_cfg)


def test_bilinear_known_grid(default_cfg):
    """Bilinear düzlemsel grid'i tam yeniden üretmeli: f(x,y)=2x+3y."""
    nx, ny, res = 16, 12, 0.5
    ox, oy = -1.0, 2.0
    xs = ox + np.arange(nx) * res
    ys = oy + np.arange(ny) * res
    X, Y = np.meshgrid(xs, ys)
    grid = (2.0 * X + 3.0 * Y).astype(np.float32)

    c = make_ctrl(default_cfg)
    c.set_esdf(grid, ox, oy, res)

    rng = np.random.default_rng(3)
    pts = np.stack([rng.uniform(ox, ox + (nx - 1) * res, 200),
                    rng.uniform(oy, oy + (ny - 1) * res, 200)], axis=1).astype(np.float32)
    got = c.query_esdf(pts)
    want = 2.0 * pts[:, 0] + 3.0 * pts[:, 1]
    assert np.allclose(got, want, atol=1e-4)


def test_grid_node_exact(default_cfg):
    rng = np.random.default_rng(5)
    grid = rng.normal(size=(8, 9)).astype(np.float32)
    ox, oy, res = 0.0, 0.0, 0.1
    c = make_ctrl(default_cfg)
    c.set_esdf(grid, ox, oy, res)
    pts = np.array([[i * res, j * res] for j in range(8) for i in range(9)],
                   dtype=np.float32)
    got = c.query_esdf(pts)
    assert np.allclose(got, grid.ravel(), atol=1e-5)


def test_boundary_clamp(default_cfg):
    grid = np.arange(12, dtype=np.float32).reshape(3, 4)
    c = make_ctrl(default_cfg)
    c.set_esdf(grid, 0.0, 0.0, 1.0)
    pts = np.array([[-5.0, -5.0], [50.0, 50.0], [-5.0, 50.0]], dtype=np.float32)
    got = c.query_esdf(pts)
    assert np.allclose(got, [grid[0, 0], grid[-1, -1], grid[-1, 0]], atol=1e-5)


def test_esdf_sign_correctness(default_cfg):
    """build_esdf: engel içinde negatif, dışında pozitif, mesafe doğru."""
    circles = np.array([[2.0, 0.0, 0.5]], dtype=np.float32)
    boxes = np.array([[5.0, 1.0, 0.4, 0.3]], dtype=np.float32)
    grid = cc.build_esdf(circles, boxes, 0.0, -2.0, 0.05, 160, 120)

    c = make_ctrl(default_cfg)
    c.set_esdf(grid, 0.0, -2.0, 0.05)

    pts = np.array([
        [2.0, 0.0],    # çember merkezi → -r
        [3.0, 0.0],    # çemberden 0.5 m
        [5.0, 1.0],    # kutu merkezi → içeride
        [5.0, 2.0],    # kutudan 0.7 m (üst kenar y=1.3)
    ], dtype=np.float32)
    got = c.query_esdf(pts)
    assert np.isclose(got[0], -0.5, atol=0.01)
    assert np.isclose(got[1], 0.5, atol=0.01)
    assert got[2] < 0.0
    assert np.isclose(got[3], 0.7, atol=0.01)


def test_terrain_query(default_cfg):
    """Eğimli düzlem: slope = atan(|∇h|) sabit; trav slope ile düşmeli."""
    nx, ny, res = 40, 40, 0.1
    xs = np.arange(nx) * res
    X, _ = np.meshgrid(xs, np.arange(ny) * res)
    height = (0.2 * X).astype(np.float32)  # dh/dx = 0.2
    feat = cc.build_elevation_features(height, res, slope_max=0.6, rough_max=0.05)
    assert feat.shape == (ny, nx, 3)

    c = make_ctrl(default_cfg)
    c.set_elevation(feat, 0.0, 0.0, res)
    pts = np.array([[2.0, 2.0], [1.0, 3.0]], dtype=np.float32)
    out = c.query_terrain(pts)
    want_slope = np.arctan(0.2)
    assert np.allclose(out[:, 0], want_slope, atol=1e-3)
    assert np.all(out[:, 2] < 1.0)   # eğim trav'ı düşürür
    assert np.all(out[:, 2] > 0.0)

    # düz zemin: slope=0, rough=0, trav=1
    flat = cc.build_elevation_features(np.zeros((20, 20), np.float32), res,
                                       slope_max=0.6, rough_max=0.05)
    c.set_elevation(flat, 0.0, 0.0, res)
    out = c.query_terrain(np.array([[0.5, 0.5]], dtype=np.float32))
    assert np.allclose(out[0], [0.0, 0.0, 1.0], atol=1e-6)
