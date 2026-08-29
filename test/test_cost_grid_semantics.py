"""ADR-014 Ç3 / K9(b): a map cell means a NORMALISED COST, and the kernel is told which.

The transport half of K9(b) landed first: set_cost_grid_device() adopts a grid that is already on
the GPU, without a copy. It moved WHERE the numbers come from and not WHAT THEY MEAN — the rollout
cost still read every cell as a signed distance in metres, subtracted the robot radius and compared
the result against safe_hard / safe_soft, which are metres.

That mismatch is invisible in the output, which is why it survived. A normalised [0,1] cost handed
to distance arithmetic reads as "under a metre of clearance everywhere", so the barrier fires in
EVERY cell — and a penalty applied to every rollout is a constant, which MPPI's softmax removes
exactly. The controller keeps producing plausible controls while ignoring the map completely.

So these tests are written the only way that can detect it: not "does a cost grid change the
control" against no map at all, but "do two grids that differ in WHICH HALF is expensive produce
different controls". A uniform grid cannot answer the question — see the last test, which asserts
that weakness on purpose.
"""
import numpy as np
import pytest

import cumpc_core as cc


def straight_ref(N=64, ds=0.05):
    xs = np.arange(N, dtype=np.float32) * ds
    return np.stack([xs, np.zeros(N, np.float32), np.zeros(N, np.float32)], axis=1)


# The grid spans 6.4 m either way at 0.05 m, centred on the robot's start.
NX = NY = 128
RES = 0.05
ORIGIN = -0.5 * NX * RES


def half_grid(expensive_is_positive_y, cheap=0.0, expensive=1.0):
    """A cost grid whose upper or lower half carries the expensive value."""
    grid = np.full((NY, NX), cheap, dtype=np.float32)
    upper = np.arange(NY)[:, None] >= NY // 2
    grid[np.broadcast_to(upper, grid.shape) == expensive_is_positive_y] = expensive
    return grid


def uniform_grid(value):
    return np.full((NY, NX), value, dtype=np.float32)


def plan_once(cfg, grid):
    """One step on a controller that has never planned.

    FRESHNESS IS THE METHOD, not tidiness: MPPI warm-starts, so u_prev and the nominal sequence
    carry from one step to the next and two consecutive steps differ for reasons the map had
    nothing to do with.
    """
    c = cc.MPPIController(cfg)
    c.set_cost_grid(grid, ORIGIN, ORIGIN, RES)
    return c.step(np.zeros(3, np.float32), straight_ref())


@pytest.fixture
def cost_cfg(default_cfg):
    # The lethal threshold is the normalised-space counterpart of safe_hard, which is metres and
    # unusable here. Without it the graded term still works but nothing is ever forbidden.
    default_cfg.cost.cost_lethal = 0.9
    return default_cfg


def test_cost_grid_changes_the_plan(cost_cfg):
    """The map is READ. Two grids that obstruct opposite halves must steer differently."""
    below = plan_once(cost_cfg, half_grid(True))
    above = plan_once(cost_cfg, half_grid(False))
    assert np.isfinite(below).all() and np.isfinite(above).all()
    assert abs(below[1] - above[1]) > 1e-3, (
        "two controllers identical except for which half of the world their cost grid makes "
        "expensive produced the same angular command: the map is not reaching the rollout cost"
    )


def test_the_planner_steers_away_from_the_expensive_half(cost_cfg):
    """And it is read with the right SIGN — a cost is avoided, not sought."""
    expensive_above = plan_once(cost_cfg, half_grid(True))
    expensive_below = plan_once(cost_cfg, half_grid(False))
    # Turning away from an expensive +y half means a more negative omega than turning away from an
    # expensive -y half. Only the ORDER is asserted; no magnitude here is a measurement.
    assert expensive_above[1] < expensive_below[1]


def test_the_map_is_read_as_a_cost_and_not_as_a_distance(cost_cfg):
    """The test that DISCRIMINATES between the two interpretations, and it had to be built.

    A 0.0-vs-1.0 grid differs under both readings — as metres, 0.0 is inside the barrier and 1.0 is
    well clear of it — so "the plan changed" proves only that the map is read, not HOW. The
    discriminator is the distance branch's SATURATION: any clearance at or beyond safe_soft
    contributes exactly zero, so under distance semantics 0.70 m and 1.00 m are indistinguishable
    (both clear robot_radius + safe_soft = 0.62 m). As a normalised cost they are not: 0.49 against
    1.00 after squaring. If these two plan identically, the cells are being read as metres.
    """
    assert cost_cfg.robot.robot_radius + cost_cfg.cost.safe_soft < 0.70, (
        "the fixture no longer saturates the distance branch at 0.70, so this test would no "
        "longer discriminate between the two readings"
    )
    mild = plan_once(cost_cfg, half_grid(True, cheap=0.70, expensive=1.00))
    swapped = plan_once(cost_cfg, half_grid(False, cheap=0.70, expensive=1.00))
    assert abs(mild[1] - swapped[1]) > 1e-3, (
        "two grids that a distance reading cannot tell apart planned identically, so the cells "
        "are being consumed as metres and not as the normalised cost ADR-014 Ç3 decided on"
    )


def test_a_uniform_cost_grid_carries_almost_no_information(cost_cfg):
    """The negative control, and the reason the tests above compare two PATTERNS.

    A constant added to every rollout's cost cancels in the softmax, so a uniform grid at 0.0 and
    one at 1.0 — the whole range — plan the same to within floating-point residue. A test that fed
    one uniform grid and expected "the map changed something" would be measuring nothing.
    """
    cheap = plan_once(cost_cfg, uniform_grid(0.0))
    expensive = plan_once(cost_cfg, uniform_grid(1.0))
    patterned = abs(plan_once(cost_cfg, half_grid(True))[1]
                    - plan_once(cost_cfg, half_grid(False))[1])
    uniform = abs(cheap[1] - expensive[1])
    assert patterned > uniform * 100.0, (
        f"a uniform grid moved the control by {uniform}, not decisively less than the "
        f"{patterned} a patterned one moves it, so the pattern tests cannot tell the map being "
        f"read from arithmetic noise"
    )


def test_an_undeclared_lethal_threshold_disables_the_barrier_rather_than_forbidding_everything(
    default_cfg,
):
    """cost_lethal is value-initialised to 0, and 0 must NOT mean "every cell is lethal".

    A caller written before this field existed leaves it at zero. Reading that as a threshold would
    make every cell of every cost grid lethal and paralyse the controller — a new field silently
    breaking old callers. Out of (0, 1] means "no lethal threshold declared"; the graded term still
    works, so the map is still read.
    """
    assert default_cfg.cost.cost_lethal == 0.0
    below = plan_once(default_cfg, half_grid(True))
    above = plan_once(default_cfg, half_grid(False))
    assert np.isfinite(below).all() and np.isfinite(above).all()
    assert abs(below[1] - above[1]) > 1e-3, "the graded cost term must work without a threshold"


def test_the_esdf_path_is_untouched(default_cfg):
    """Ç3 says the ESDF path STAYS upstream; only BudgetRT stops feeding it.

    set_esdf() still means metres. The value here is a clearance well inside safe_soft on one half
    and well outside it on the other, so the distance branch has to be running for the two to
    differ — under cost semantics both halves would be clamped to 1.0 and indistinguishable.
    """
    near = np.full((NY, NX), 5.0, dtype=np.float32)
    near[: NY // 2, :] = 0.05
    far = np.full((NY, NX), 5.0, dtype=np.float32)
    far[NY // 2 :, :] = 0.05

    def esdf_plan(grid):
        c = cc.MPPIController(default_cfg)
        c.set_esdf(grid, ORIGIN, ORIGIN, RES)
        return c.step(np.zeros(3, np.float32), straight_ref())

    assert abs(esdf_plan(near)[1] - esdf_plan(far)[1]) > 1e-3
