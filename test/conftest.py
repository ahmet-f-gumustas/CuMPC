import pytest


@pytest.fixture
def default_cfg():
    """Bağımsız varsayılan konfigürasyon (harness'e bağımlılık yok).

    Değerler cumpc-mujoco reposundaki config/default.yaml ile eşleşir;
    kontratın tek kaynağı production'da kullanan kabuğun YAML'ıdır.
    """
    import cumpc_core as cc

    m = cc.MPPIConfig()
    m.K, m.H, m.dt = 2048, 40, 0.05
    m.lambda_ = 50.0
    m.sigma_v, m.sigma_omega = 0.15, 0.25
    m.seed = 12345

    r = m.robot
    r.wheel_radius, r.track_width = 0.10, 0.40
    r.v_max, r.omega_max = 2.0, 2.0
    r.a_max, r.alpha_max = 2.0, 3.0
    r.robot_radius = 0.32

    s = m.slip
    s.kappa_v, s.kappa_w, s.v_y = 1.0, 1.0, 0.0

    w = m.weights
    w.w_lat, w.w_head, w.w_prog, w.w_du = 50.0, 10.0, 50.0, 0.1
    w.w_coll_hard, w.w_coll_soft = 1.0e6, 200.0
    w.w_slope, w.w_rough, w.w_trav, w.w_rollover = 20.0, 10.0, 30.0, 1.0e5
    w.w_term = 1000.0

    c = m.cost
    c.k_accel, c.safe_hard, c.safe_soft = 0.1, 0.10, 0.30
    c.rollover_slope, c.term_slack = 0.45, 2.2

    return m
