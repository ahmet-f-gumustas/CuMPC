"""two-slot immutable config: prepare / commit / abort, and the epoch a step actually read.

The rules under test are not conveniences. A controller whose configuration can change underneath a
running step produces a tick that ran half on one config and half on another, and every number that
tick reports is then about a configuration nobody can name. So:

  * prepare() stages and does the expensive work (buffer growth) OFF the step path;
  * commit() only swaps, so it is cheap enough for a frame boundary;
  * a second prepare() while one is pending is REFUSED rather than silently overwriting;
  * "published" (active_epoch) and "applied" (observed_epoch) are two different facts.
"""
import numpy as np
import pytest

import cumpc_core as cc


def straight_ref(N=64, ds=0.05):
    xs = np.arange(N, dtype=np.float32) * ds
    return np.stack([xs, np.zeros(N, np.float32), np.zeros(N, np.float32)], axis=1)


def test_epoch_starts_published_but_unapplied(default_cfg):
    c = cc.MPPIController(default_cfg)
    assert c.active_epoch() >= 1
    # Nothing has run yet, so nothing has OBSERVED the config. Reporting the active epoch here would
    # claim a tick used a configuration no tick has seen.
    assert c.observed_epoch() == 0
    assert not c.has_prepared()


def test_a_step_observes_the_active_epoch(default_cfg):
    c = cc.MPPIController(default_cfg)
    c.step(np.zeros(3, np.float32), straight_ref())
    assert c.observed_epoch() == c.active_epoch()


def test_commit_without_prepare_changes_nothing(default_cfg):
    c = cc.MPPIController(default_cfg)
    before = c.active_epoch()
    assert c.commit() == before
    assert c.active_epoch() == before


def test_prepare_then_commit_advances_the_epoch(default_cfg):
    c = cc.MPPIController(default_cfg)
    before = c.active_epoch()

    assert c.prepare(default_cfg)
    assert c.has_prepared()
    # Preparing does NOT publish: until commit, the active epoch is untouched.
    assert c.active_epoch() == before

    after = c.commit()
    assert after == before + 1
    assert not c.has_prepared()


def test_a_second_prepare_while_one_is_pending_is_refused(default_cfg):
    c = cc.MPPIController(default_cfg)
    assert c.prepare(default_cfg)
    # Which of the two would commit()? The question has no good answer, so the second is refused
    # rather than silently overwriting the first.
    assert not c.prepare(default_cfg)
    assert c.has_prepared()


def test_abort_discards_the_prepared_config_without_publishing_it(default_cfg):
    c = cc.MPPIController(default_cfg)
    before = c.active_epoch()
    assert c.prepare(default_cfg)
    c.abort_prepared()
    assert not c.has_prepared()
    assert c.active_epoch() == before
    # And after an abort the slot is free again.
    assert c.prepare(default_cfg)


def test_published_and_applied_are_two_different_facts(default_cfg):
    c = cc.MPPIController(default_cfg)
    c.step(np.zeros(3, np.float32), straight_ref())
    assert c.prepare(default_cfg)
    published = c.commit()

    # The transition is PUBLISHED but no tick has read it yet. Treating these as one number is how a
    # transition that never reached the planner gets counted as applied.
    assert c.active_epoch() == published
    assert c.observed_epoch() == published - 1

    c.step(np.zeros(3, np.float32), straight_ref())
    assert c.observed_epoch() == published


def test_a_shape_change_goes_through_prepare_and_still_produces_a_control(default_cfg):
    """K and H are class (ii): changing them needs a re-init, which is what prepare() is for."""
    c = cc.MPPIController(default_cfg)
    c.step(np.zeros(3, np.float32), straight_ref())

    bigger = cc.MPPIConfig()
    for field in ("K", "H", "dt", "sigma_v", "sigma_omega", "seed"):
        setattr(bigger, field, getattr(default_cfg, field))
    bigger.robot = default_cfg.robot
    bigger.slip = default_cfg.slip
    bigger.weights = default_cfg.weights
    bigger.cost = default_cfg.cost
    bigger.K = default_cfg.K * 2
    bigger.H = default_cfg.H + 8

    assert c.prepare(bigger)
    assert c.commit() == 2

    u = c.step(np.zeros(3, np.float32), straight_ref())
    assert u.shape == (2,)
    assert np.all(np.isfinite(u))
    assert c.nominal().shape == (bigger.H, 2)
    assert c.last_rollouts().shape == (bigger.K, bigger.H, 3)
    assert c.observed_epoch() == 2
