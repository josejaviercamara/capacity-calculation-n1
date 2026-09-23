"""
Day ahead (D-1) and intraday (ID) capacity calculation over 24 hours,
with a probabilistic Transmission Reliability Margin (TRM) from Monte Carlo.

Assumptions (all stated in the README):
  * Load in both zones follows the same daily profile (1.0 = case file load).
  * A 50 MW wind farm in zone NB. The intraday forecast is closer to the
    realised wind than the day ahead one (a weather front arrives earlier).
  * The scheduled net position of CH follows a simple daily pattern
    (import at night, export during the day), set by the market.
  * Each zone meets load + scheduled export with its own conventional units,
    shifted pro rata to their dispatch (same logic as the GSK).
  * Forecast errors are balanced by frequency containment reserves shared
    by all units of the synchronous area, pro rata to dispatch, so part of
    the imbalance flows through the other zone (unscheduled flows).
"""

import numpy as np
from capacity import (ZONE_A, ZONE_B, WIND_BUS, gsk, zonal_ptdf, max_transfer,
                      exchange)

HOURS = np.arange(24)
LOAD_PROFILE = np.array([0.72, 0.68, 0.66, 0.65, 0.66, 0.71, 0.80, 0.90,
                         0.96, 0.98, 0.99, 1.00, 0.99, 0.97, 0.95, 0.94,
                         0.95, 0.98, 1.02, 1.04, 1.00, 0.93, 0.84, 0.77])
WIND_CAP = 50.0
WIND_DA = WIND_CAP * np.array([0.30, 0.32, 0.33, 0.34, 0.34, 0.33, 0.31, 0.28,
                               0.25, 0.22, 0.20, 0.20, 0.22, 0.26, 0.32, 0.40,
                               0.48, 0.55, 0.60, 0.62, 0.62, 0.60, 0.57, 0.54])
# intraday update: the front arrives about three hours earlier than forecast
WIND_ID = WIND_CAP * np.array([0.30, 0.32, 0.33, 0.34, 0.34, 0.33, 0.30, 0.27,
                               0.25, 0.26, 0.32, 0.40, 0.48, 0.56, 0.62, 0.66,
                               0.68, 0.70, 0.70, 0.68, 0.66, 0.63, 0.60, 0.56])
NP_SCHED = np.array([-8, -8, -8, -8, -8, -6, -2, 4, 8, 10, 10, 10,
                     10, 10, 10, 8, 6, 4, 0, -2, -4, -6, -8, -8], dtype=float)

# forecast error standard deviations, as a share of installed wind / load
SIGMA = {"DA": {"wind": 0.12, "load": 0.03},
         "ID": {"wind": 0.05, "load": 0.015}}


def dispatch(net, units0, load_factor, wind, np_sched):
    """Conventional dispatch that meets zonal load and the scheduled exchange."""
    lz = net.bus.bz.values[net.load.bus.values.astype(int)]
    load_a = net.load.p_mw.values[lz == ZONE_A].sum() * load_factor
    load_b = net.load.p_mw.values[lz == ZONE_B].sum() * load_factor
    target = {ZONE_A: load_a + np_sched, ZONE_B: load_b - wind - np_sched}
    u = {k: v.copy() for k, v in units0.items()}
    for z, tot in target.items():
        m = u["zone"] == z
        u["p"][m] = units0["p"][m] / units0["p"][m].sum() * tot
    if np.any(u["p"] > u["pmax"] + 1e-6) or np.any(u["p"] < -1e-6):
        raise ValueError("dispatch outside unit limits")
    return u


def injection(net, units, load_factor, wind):
    inj = np.zeros(len(net.bus))
    np.add.at(inj, units["bus"], units["p"])
    np.add.at(inj, net.load.bus.values.astype(int), -net.load.p_mw.values * load_factor)
    inj[WIND_BUS] += wind
    return inj


def capacity_hour(net, units0, ptdf, lodf, rate, cont, tie, load_factor, wind, np_sched):
    """TTC in both directions for one hour and the binding constraints."""
    u = dispatch(net, units0, load_factor, wind, np_sched)
    f0 = ptdf @ injection(net, u, load_factor, wind)
    z = zonal_ptdf(ptdf, gsk(u, ZONE_A, len(net.bus)), gsk(u, ZONE_B, len(net.bus)))
    bce = exchange(f0, *tie)
    de_exp, arg_exp = max_transfer(f0, z, rate, lodf, cont)
    de_imp, arg_imp = max_transfer(f0, -z, rate, lodf, cont)
    return {"units": u, "f0": f0, "z": z, "bce": bce,
            "ttc_export": bce + de_exp, "bind_export": arg_exp, "de_export": de_exp,
            "ttc_import": -bce + de_imp, "bind_import": arg_imp, "de_import": de_imp}


def monte_carlo_trm(net, res, ptdf, lodf, rate, cont, load_factor, stage,
                    n_samples=1000, q=0.05, seed=0):
    """Probabilistic TRM: reduction of the export margin that is not exceeded
    with probability 1 - q, given the forecast errors of the stage."""
    rng = np.random.default_rng(seed)
    s = SIGMA[stage]
    u = res["units"]
    load_bus = net.load.bus.values.astype(int)
    load_mw = net.load.p_mw.values * load_factor
    share = u["p"] / u["p"].sum()                     # FCR sharing
    margins_exp, margins_imp = np.empty(n_samples), np.empty(n_samples)
    for i in range(n_samples):
        dp = np.zeros(len(net.bus))
        dp[WIND_BUS] += rng.normal(0, s["wind"] * WIND_CAP)
        lerr = rng.normal(0, s["load"], size=2)          # one error per zone
        lz = net.bus.bz.values[load_bus]
        np.add.at(dp, load_bus, -load_mw * np.where(lz == ZONE_A, lerr[0], lerr[1]))
        np.add.at(dp, u["bus"], -dp.sum() * share)       # balanced by FCR
        f = res["f0"] + ptdf @ dp
        margins_exp[i] = max_transfer(f, res["z"], rate, lodf, cont)[0]
        margins_imp[i] = max_transfer(f, -res["z"], rate, lodf, cont)[0]
    trm_exp = res["de_export"] - np.quantile(margins_exp, q)
    trm_imp = res["de_import"] - np.quantile(margins_imp, q)
    return max(trm_exp, 0.0), max(trm_imp, 0.0), margins_exp, margins_imp
