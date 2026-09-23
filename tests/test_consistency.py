"""Consistency tests: every analytical shortcut must agree with a full load flow.
Run with:  pytest -q
"""
import copy
import sys
import warnings
from pathlib import Path

import numpy as np
import pandapower as pp

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
warnings.filterwarnings("ignore")
from capacity import *          # noqa: E402,F403
from scenarios import *         # noqa: E402,F403

net = build_network()
units0 = conventional_units(net)
ptdf, lodf, rate = dc_sensitivities(net)
cont = credible_contingencies(net, ptdf)
tie = tie_lines(net)
P0 = nodal_injection(net, units0["p"], units0["bus"])
f0 = ptdf @ P0


def test_ptdf_matches_dc_load_flow():
    pp.rundcpp(net, numba=False)
    assert np.allclose(f0, net.res_line.p_from_mw.values, atol=1e-8)


def test_lodf_matches_outage_simulation():
    for k in cont:
        n = copy.deepcopy(net)
        n.line.loc[k, "in_service"] = False
        pp.rundcpp(n, numba=False)
        m = np.arange(len(net.line)) != k
        assert np.allclose((f0 + lodf[:, k] * f0[k])[m], n.res_line.p_from_mw.values[m], atol=1e-8)


def test_tie_line_zonal_ptdfs_sum_to_one():
    z = zonal_ptdf(ptdf, gsk(units0, ZONE_A, len(net.bus)), gsk(units0, ZONE_B, len(net.bus)))
    assert abs(np.sum(z[tie[0]] * tie[1]) - 1.0) < 1e-9


def test_vectorised_equals_reference():
    z = zonal_ptdf(ptdf, gsk(units0, ZONE_A, len(net.bus)), gsk(units0, ZONE_B, len(net.bus)))
    for sign in (1, -1):
        a = max_transfer(f0, sign * z, rate, lodf, cont)
        b = max_transfer_reference(f0, sign * z, rate, lodf, cont)
        assert abs(a[0] - b[0]) < 1e-9 and a[1] == b[1]


def test_ttc_is_exactly_the_n1_limit():
    """At TTC the worst N-1 loading is 100 %, 1 MW more gives a violation."""
    h = 12
    r = capacity_hour(net, units0, ptdf, lodf, rate, cont, tie, LOAD_PROFILE[h], WIND_DA[h], NP_SCHED[h])
    for extra, expect_ok in [(0.0, True), (1.0, False)]:
        de = r["de_export"] + extra
        u = balance_zone(balance_zone(r["units"], ZONE_A, de), ZONE_B, -de)
        base = set_dispatch(net, u, LOAD_PROFILE[h], LOAD_PROFILE[h], WIND_DA[h])
        worst = 0.0
        for k in [-1] + list(cont):
            n = copy.deepcopy(base)
            if k >= 0:
                n.line.loc[k, "in_service"] = False
            pp.rundcpp(n, numba=False)
            load = np.abs(n.res_line.p_from_mw.values) / rate
            if k >= 0:
                load[k] = 0
            worst = max(worst, load.max())
        assert (worst <= 1 + 1e-9) == expect_ok


def test_all_base_cases_are_n1_secure():
    for h in HOURS:
        for W in (WIND_DA, WIND_ID):
            r = capacity_hour(net, units0, ptdf, lodf, rate, cont, tie, LOAD_PROFILE[h], W[h], NP_SCHED[h])
            assert security_check(r["f0"], rate, lodf, cont)[1] <= 1 + 1e-9
