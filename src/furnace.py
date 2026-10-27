# Mustaf bare



# furnace.py
# A virtual A400-style vertical furnace. One call to run_load() runs one
# batch and gives back two tables:
#
#   wafers         the true thickness of every wafer in the boat. On a real
#                  tool nobody gets to see this.
#   monitor_sites  9-site measurements on the monitor wafers only, with
#                  measurement noise. This is what I'd actually have to work with.
#
# Each wafer's thickness comes from deal_grove.py. This file just works out
# what conditions each slot really sees: its zone temperature, how much
# oxidant reaches it, and some random noise.

import math

import numpy as np
import pandas as pd

import config as cfg
from deal_grove import thickness


def zone_of(slot):
    return min((slot - 1) // cfg.SLOTS_PER_ZONE + 1, cfg.N_ZONES)


def slot_type(slot):
    if slot in cfg.DUMMY_SLOTS:
        return "dummy"
    if slot in cfg.MONITOR_SLOTS:
        return "monitor"
    return "product"


def slot_temperatures(setpoint_c, reactor="A", zone_trim_c=None, drift_c=0.0):
    if zone_trim_c is None:
        zone_trim_c = [0.0] * cfg.N_ZONES

    zone_temps = (setpoint_c
                  + np.array(cfg.TRUE_ZONE_OFFSETS_C[reactor])
                  + np.array(zone_trim_c)
                  + drift_c)

    zone_centers = [(z - 0.5) * cfg.SLOTS_PER_ZONE + 0.5 for z in range(1, cfg.N_ZONES + 1)]
    slots = np.arange(1, cfg.BOAT_SLOTS + 1)
    return np.interp(slots, zone_centers, zone_temps)


def oxidant_factor():
    slots = np.arange(1, cfg.BOAT_SLOTS + 1)
    if cfg.GAS_INLET == "bottom":
        distance = (cfg.BOAT_SLOTS - slots) / (cfg.BOAT_SLOTS - 1)  
    else:
        distance = (slots - 1) / (cfg.BOAT_SLOTS - 1)
    return 1.0 - cfg.DEPLETION * distance


def run_load(recipe=None, reactor="A", zone_trim_c=None, drift_c=0.0, seed=None):
    if recipe is None:
        recipe = cfg.BASE_RECIPE
    rng = np.random.default_rng(seed)

    slots = np.arange(1, cfg.BOAT_SLOTS + 1)

    run_noise = rng.normal(0, cfg.RUN_TEMP_NOISE_C)
    temps = slot_temperatures(recipe["temp_c"], reactor, zone_trim_c, drift_c) + run_noise

    effective_time = recipe["time_hr"] * oxidant_factor()
    x_nm = thickness(effective_time, temps, recipe["ambient"])

    x_nm = x_nm * (1 + rng.normal(0, cfg.WAFER_NOISE_FRAC, len(slots)))

    wafers = pd.DataFrame({
        "reactor": reactor,
        "slot": slots,
        "zone": [zone_of(s) for s in slots],
        "type": [slot_type(s) for s in slots],
        "temp_c": temps,
        "thickness_A": x_nm * 10,
    })

    monitor_sites = measure_monitors(wafers, rng)
    return wafers, monitor_sites


def measure_monitors(wafers, rng):
    radius_mm = cfg.WAFER_DIAMETER_MM / 2
    rows = []
    for w in wafers[wafers["type"] == "monitor"].itertuples():
        for site, (r_frac, angle) in enumerate(cfg.SITES, start=1):
            true_site = w.thickness_A * (1 + cfg.WIW_EDGE_FRAC * r_frac**2)
            measured = true_site + rng.normal(0, cfg.METROLOGY_NOISE_A)
            rows.append({
                "reactor": w.reactor,
                "slot": w.slot,
                "zone": w.zone,
                "site": site,
                "x_mm": r_frac * radius_mm * math.cos(math.radians(angle)),
                "y_mm": r_frac * radius_mm * math.sin(math.radians(angle)),
                "thickness_A": measured,
            })
    return pd.DataFrame(rows)


def monitor_summary(monitor_sites):
    g = monitor_sites.groupby(["reactor", "slot", "zone"])["thickness_A"]
    summary = g.agg(mean_A="mean", min_A="min", max_A="max").reset_index()
    summary["wiw_range_A"] = summary["max_A"] - summary["min_A"]
    summary["wiw_nu_pct"] = summary["wiw_range_A"] / (2 * summary["mean_A"]) * 100
    return summary[["reactor", "slot", "zone", "mean_A", "wiw_range_A", "wiw_nu_pct"]]


def uniformity(values):
    values = np.asarray(values)
    mean = values.mean()
    rng_ = values.max() - values.min()
    return {
        "mean_A": mean,
        "range_A": rng_,
        "sigma_A": values.std(ddof=1),
        "nu_pct": rng_ / (2 * mean) * 100,  
    }


def plot_boat_profile(loads, save_to):
    import matplotlib.pyplot as plt

    fig, (ax_t, ax_x) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

    for i, (wafers, sites) in enumerate(loads):
        color = f"C{i}"
        reactor = wafers["reactor"].iloc[0]
        real = wafers[wafers["type"] != "dummy"]
        mons = monitor_summary(sites)

        ax_t.plot(wafers["slot"], wafers["temp_c"], color=color, label=f"Reactor {reactor}")
        ax_x.plot(real["slot"], real["thickness_A"], ".", color=color, ms=3, alpha=0.5,
                  label=f"Reactor {reactor}, every wafer (true)")
        ax_x.plot(mons["slot"], mons["mean_A"], "o", color=color, mec="k", ms=8,
                  label=f"Reactor {reactor}, monitors (measured)")

    for ax in (ax_t, ax_x):
        ax.axvspan(0.5, 5.5, color="grey", alpha=0.15)
        ax.axvspan(105.5, 110.5, color="grey", alpha=0.15)
        for z in range(1, cfg.N_ZONES):
            ax.axvline(z * cfg.SLOTS_PER_ZONE + 0.5, color="k", ls=":", lw=0.8)
        ax.grid(alpha=0.3)

    for z in range(1, cfg.N_ZONES + 1):
        ax_t.text((z - 0.5) * cfg.SLOTS_PER_ZONE + 0.5, 1.02, f"Zone {z}",
                  transform=ax_t.get_xaxis_transform(), ha="center", fontsize=9)

    ax_x.axhline(cfg.TARGET_A, color="k", lw=1, label=f"Target {cfg.TARGET_A:g} Å")
    ax_t.set_ylabel("Actual temperature (°C)")
    ax_x.set_ylabel("Oxide thickness (Å)")
    ax_x.set_xlabel("Boat slot (1 = top, 110 = bottom, grey = dummies)")
    ax_t.legend(fontsize=8)
    ax_x.legend(fontsize=8)
    fig.suptitle("Boat profile, untuned furnace", y=0.995)
    fig.tight_layout()
    fig.savefig(save_to, dpi=150)
    print(f"Saved {save_to}")


if __name__ == "__main__":
    from pathlib import Path

    plots = Path(__file__).resolve().parents[1] / "plots"
    plots.mkdir(exist_ok=True)

    recipe = cfg.BASE_RECIPE
    print(f"Recipe: {recipe['ambient']}, {recipe['temp_c']:g} °C, {recipe['time_hr']} hr, "
          f"target {cfg.TARGET_A:g} Å\n")

    loads = []
    for reactor in cfg.REACTORS:
        wafers, sites = run_load(recipe, reactor=reactor, seed=42)
        loads.append((wafers, sites))

        print(f"Reactor {reactor} monitors (what the metrology tool reports)")
        print(monitor_summary(sites).to_string(index=False, float_format=lambda v: f"{v:8.1f}"))

        real = wafers[wafers["type"] != "dummy"]["thickness_A"]
        u = uniformity(real)
        print(f"  True wafer-to-wafer across all {len(real)} wafers: mean {u['mean_A']:.1f} Å, "
              f"range {u['range_A']:.1f} Å, nonuniformity {u['nu_pct']:.2f}%\n")

    plot_boat_profile(loads, plots / "boat_profile.png")