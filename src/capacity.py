"""
Cross zonal capacity calculation on the IEEE 30 bus test grid.

Method (standard DC based capacity calculation, as used by European TSOs):
  1. DC power flow sensitivities: nodal PTDF matrix and LODF matrix.
  2. Generation Shift Keys (GSK) turn nodal PTDFs into zonal PTDFs.
  3. For every monitored element in the N state and after every credible
     single outage (N-1), the remaining margin is divided by the zonal PTDF.
     The smallest positive value is the maximum additional exchange.
  4. TTC = base case exchange + maximum additional exchange.
     NTC = TTC - TRM, where TRM covers forecast uncertainty.

All flows are active power in MW. Line limits are the thermal ratings of the
case file (rateA, MVA), used as MW limits under the DC assumption.
"""

import copy
import logging
import numpy as np
import pandapower as pp
import pandapower.networks as pn
from pandapower.pypower.makePTDF import makePTDF
from pandapower.pypower.makeLODF import makeLODF

logging.getLogger("pandapower").setLevel(logging.ERROR)  # hide optional numba notice

ZONE_A = "CH"    # zone 1 of the case file
ZONE_B = "NB"    # zones 2 and 3 of the case file ("neighbour")
WIND_BUS = 9     # 50 MW wind farm in zone NB, at bus 10 (1-based numbering, as in the IEEE case)
CNEC_THRESHOLD = 0.05  # elements with |zonal PTDF| < 5 % do not limit capacity
LOAD_MIN_PF = 0.95     # loads compensated to cos(phi) >= 0.95 (only affects AC checks)
GEN_VM_PU = 1.05       # generator voltage set point (only affects AC checks)


# ---------------------------------------------------------------------------
# Network
# ---------------------------------------------------------------------------
def build_network():
    """IEEE 30 bus case split into two bidding zones.

    Reactive power only matters for the AC checks: the original case has loads
    down to cos(phi) = 0.71 and all voltage set points at 1.0 pu, which makes
    line 6-8 overloaded in AC even without any exchange. Loads are therefore
    assumed compensated to cos(phi) >= 0.95 and generators regulate 1.05 pu,
    both usual operating practice. DC results do not depend on this."""
    net = pn.case30()
    q_max = net.load.p_mw * np.tan(np.arccos(LOAD_MIN_PF))
    net.load["q_mvar"] = np.minimum(net.load.q_mvar, q_max)
    net.gen["vm_pu"] = GEN_VM_PU
    net.ext_grid["vm_pu"] = GEN_VM_PU
    zone = np.where(net.bus.zone.values == 1, ZONE_A, ZONE_B)
    net.bus["bz"] = zone
    return net


def conventional_units(net):
    """Dispatchable units as a table: bus, dispatch, zone (ext_grid first)."""
    pp.rundcpp(net, numba=False)
    buses = [int(net.ext_grid.bus.iloc[0])] + list(net.gen.bus.astype(int))
    p = [float(net.res_ext_grid.p_mw.iloc[0])] + list(net.gen.p_mw.astype(float))
    pmax = [float(net.ext_grid.max_p_mw.iloc[0])] + list(net.gen.max_p_mw.astype(float))
    zone = [net.bus.bz.iloc[b] for b in buses]
    return {"bus": np.array(buses), "p": np.array(p),
            "pmax": np.array(pmax), "zone": np.array(zone)}


def tie_lines(net):
    """Indices of lines connecting the two zones, oriented CH -> NB (+1/-1)."""
    idx, sign = [], []
    for i, r in net.line.iterrows():
        za, zb = net.bus.bz.iloc[int(r.from_bus)], net.bus.bz.iloc[int(r.to_bus)]
        if za != zb:
            idx.append(i)
            sign.append(1.0 if za == ZONE_A else -1.0)
    return np.array(idx), np.array(sign)


# ---------------------------------------------------------------------------
# DC sensitivities
# ---------------------------------------------------------------------------
def dc_sensitivities(net):
    """Nodal PTDF (lines x buses), LODF (lines x lines) and ratings (MW)."""
    pp.rundcpp(net, numba=False)
    ppc = net._ppc
    slack = int(net.ext_grid.bus.iloc[0])
    ptdf = makePTDF(ppc["baseMVA"], ppc["bus"], ppc["branch"], slack=slack)
    lodf = makeLODF(ppc["branch"], ptdf)
    rate = ppc["branch"][:, 5].real.copy()
    return ptdf, lodf, rate


def credible_contingencies(net, ptdf, tol=1e-6):
    """All single line outages that do not split the grid.
    An outage of line k islands the grid when its self sensitivity
    PTDF_k,from - PTDF_k,to equals 1 (the LODF denominator becomes zero)."""
    f = net.line.from_bus.values.astype(int)
    t = net.line.to_bus.values.astype(int)
    self_sens = ptdf[np.arange(len(f)), f] - ptdf[np.arange(len(f)), t]
    return np.where(np.abs(1.0 - self_sens) > tol)[0]


# ---------------------------------------------------------------------------
# Injections and zonal PTDFs
# ---------------------------------------------------------------------------
def nodal_injection(net, units_p, units_bus, load_scale_a=1.0, load_scale_b=1.0,
                    wind_mw=0.0):
    """Net nodal injection (MW) = generation - load."""
    inj = np.zeros(len(net.bus))
    np.add.at(inj, units_bus, units_p)
    lz = net.bus.bz.values[net.load.bus.values.astype(int)]
    scale = np.where(lz == ZONE_A, load_scale_a, load_scale_b)
    np.add.at(inj, net.load.bus.values.astype(int), -net.load.p_mw.values * scale)
    inj[WIND_BUS] += wind_mw
    return inj


def gsk(units, zone, n_bus):
    """Pro rata GSK: share of each unit's current dispatch within its zone."""
    g = np.zeros(n_bus)
    m = units["zone"] == zone
    w = units["p"][m] / units["p"][m].sum()
    np.add.at(g, units["bus"][m], w)
    return g


def zonal_ptdf(ptdf, gsk_a, gsk_b):
    """Flow change per MW of exchange CH -> NB (export from CH)."""
    return ptdf @ gsk_a - ptdf @ gsk_b


def balance_zone(units, zone, delta):
    """Change the dispatch of one zone by delta MW, pro rata to dispatch."""
    u = {k: v.copy() for k, v in units.items()}
    m = u["zone"] == zone
    u["p"][m] += delta * u["p"][m] / u["p"][m].sum()
    return u


# ---------------------------------------------------------------------------
# Capacity calculation
# ---------------------------------------------------------------------------
def max_transfer(f0, zptdf, rate, lodf, contingencies, threshold=CNEC_THRESHOLD):
    """Maximum additional exchange CH -> NB from base flows f0.

    For every state s (N and each credible outage k) and element l:
        post outage flow      F_l^k   = F_l + LODF_lk * F_k
        post outage zPTDF     z_l^k   = z_l + LODF_lk * z_k
        allowed shift         dE      = (+-Fmax_l - F_l^k) / z_l^k
    with the sign of Fmax equal to the sign of z_l^k (the flow moves towards
    that limit as the exchange grows). The binding constraint is the minimum
    over all elements and states. Elements with |z| below the threshold are
    not considered (CNEC selection). The outaged line itself is excluded.
    Returns the shift (MW) and the binding (element, outage) pair,
    outage = -1 meaning the N state.
    """
    k = np.asarray(contingencies, dtype=int)
    n = len(f0)
    # columns: N state followed by every outage
    F = np.column_stack([f0, f0[:, None] + lodf[:, k] * f0[k][None, :]])
    Z = np.column_stack([zptdf, zptdf[:, None] + lodf[:, k] * zptdf[k][None, :]])
    valid = np.abs(Z) >= threshold
    valid[k, np.arange(1, len(k) + 1)] = False
    limit = np.where(Z > 0, rate[:, None], -rate[:, None])
    with np.errstate(divide="ignore", invalid="ignore"):
        dE = np.where(valid, (limit - F) / Z, np.inf)
    l, c = np.unravel_index(np.argmin(dE), dE.shape)
    outage = -1 if c == 0 else int(k[c - 1])
    return float(dE[l, c]), (int(l), outage)


def max_transfer_reference(f0, zptdf, rate, lodf, contingencies, threshold=CNEC_THRESHOLD):
    """Plain loop version of max_transfer, kept to cross check the vectorised one."""
    best, arg = np.inf, None
    for k in [-1] + list(contingencies):
        if k == -1:
            f, z = f0, zptdf
        else:
            f = f0 + lodf[:, k] * f0[k]
            z = zptdf + lodf[:, k] * zptdf[k]
        for l in range(len(f)):
            if l == k or abs(z[l]) < threshold:
                continue
            limit = rate[l] if z[l] > 0 else -rate[l]
            de = (limit - f[l]) / z[l]
            if de < best:
                best, arg = de, (l, int(k))
    return best, arg


def post_contingency_flows(f0, lodf, contingencies):
    """Matrix (lines x outages) of post outage flows."""
    return f0[:, None] + lodf[:, contingencies] * f0[contingencies][None, :]


def security_check(f0, rate, lodf, contingencies):
    """Highest loading (share of rating) in the N state and over all N-1 states.
    A value above 1 means the base case itself is not N-1 secure."""
    F = post_contingency_flows(f0, lodf, contingencies)
    load = np.abs(F) / rate[:, None]
    load[contingencies, np.arange(len(contingencies))] = 0.0  # outaged line
    return float(np.max(np.abs(f0) / rate)), float(load.max())


def exchange(f, tie_idx, tie_sign):
    """Commercial exchange CH -> NB measured as the sum of tie line flows."""
    return float(np.sum(f[tie_idx] * tie_sign))


# ---------------------------------------------------------------------------
# Validation helpers against pandapower's own solver
# ---------------------------------------------------------------------------
def set_dispatch(net, units, load_scale_a=1.0, load_scale_b=1.0, wind_mw=0.0):
    """Write a dispatch into the pandapower model (for independent checks)."""
    n = copy.deepcopy(net)
    lz = n.bus.bz.values[n.load.bus.values.astype(int)]
    n.load["p_mw"] = n.load.p_mw.values * np.where(lz == ZONE_A, load_scale_a, load_scale_b)
    n.load["q_mvar"] = n.load.q_mvar.values * np.where(lz == ZONE_A, load_scale_a, load_scale_b)
    n.gen["p_mw"] = units["p"][1:]
    n.ext_grid["max_p_mw"] = 1e3
    if wind_mw:
        pp.create_sgen(n, WIND_BUS, p_mw=wind_mw, name="wind")
    return n


def line_label(net, l):
    """Line name with 1-based bus numbers, as in the IEEE case (e.g. '6-8')."""
    return f"{int(net.line.from_bus[l]) + 1}-{int(net.line.to_bus[l]) + 1}"
