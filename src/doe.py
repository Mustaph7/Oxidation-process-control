# Mustaf Bare

"""
doe.py

Before I can fix the boat profile I need to know how much each heater
zone actually moves each monitor wafer, and that's what this experiment
finds. It's a 2^(5-1) fractional factorial on the 5 zone trims, where
every zone gets set to either -3 °C or +3 °C. I tried 2 °C first, but
run-to-run noise threw the answers off by up to 9%, so I went to 3 °C,
which stands out better while the response still stays a straight line.

Zones 1-4 go through every +/- combination for 16 runs, and zone 5 is
zones 1*2*3*4 multiplied together. That keeps every zone independent of
the others with half the runs of a full factorial (32). I also added 3
runs at the center with no trim to check that the response really is
linear, and the runs happen in random order so that if the tool drifts
during the experiment, it doesn't get blamed on one of the zones.

From there I fit a straight line for each monitor, thickness = baseline
+ S1*trim1 + S2*trim2 + ... + S5*trim5. The S numbers for all 5 monitors
make up the sensitivity matrix in Å per °C, with monitors as rows and
zones as columns. Since it's a simulator, I can also get the real answer
with all the noise turned off, so at the end I check how close the DOE got.
"""

from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

import config as cfg
import furnace
from deal_grove import thickness

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "data"
PLOTS = REPO / "plots"

STEP_C = 3.0      
N_CENTER = 3      
REACTOR = "A"
SEED = 7          


def build_design():
    rows = []
    for z1, z2, z3, z4 in product([-1, 1], repeat=4):
        z5 = z1 * z2 * z3 * z4
        rows.append([z1, z2, z3, z4, z5])

    for _ in range(N_CENTER):
        rows.append([0, 0, 0, 0, 0])

    return pd.DataFrame(rows, columns=["z1", "z2", "z3", "z4", "z5"])


def run_experiment(design, recipe, reactor, seed):
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(design))   

    log = []
    run_number = 0
    for row_index in order:
        run_number += 1
        coded = design.loc[row_index].to_numpy(dtype=float)
        trim = coded * STEP_C

        wafers, sites = furnace.run_load(recipe, reactor=reactor, zone_trim_c=trim,
                                         seed=seed * 1000 + run_number)
        monitors = furnace.monitor_summary(sites)

        entry = {"run": run_number, "design_row": int(row_index)}
        for z in range(cfg.N_ZONES):
            entry[f"trim_z{z + 1}"] = trim[z]
        for m in monitors.itertuples():
            entry[f"m{m.slot}"] = m.mean_A
        log.append(entry)

    return pd.DataFrame(log)


def fit_sensitivity(log):
    trim_cols = [f"trim_z{z}" for z in range(1, cfg.N_ZONES + 1)]
    monitor_cols = [f"m{s}" for s in cfg.MONITOR_SLOTS]

    trims = log[trim_cols].to_numpy()
    Y = log[monitor_cols].to_numpy()
    X = np.column_stack([np.ones(len(trims)), trims])

    coef = np.linalg.lstsq(X, Y, rcond=None)[0]
    baseline = coef[0]
    S = coef[1:].T   
    resid = Y - X @ coef
    ss_res = (resid ** 2).sum(axis=0)
    ss_tot = ((Y - Y.mean(axis=0)) ** 2).sum(axis=0)
    r2 = 1 - ss_res / ss_tot
    resid_sd = np.sqrt(ss_res / (len(Y) - X.shape[1]))

    return S, baseline, r2, resid_sd


def curvature_check(log):
    trim_cols = [f"trim_z{z}" for z in range(1, cfg.N_ZONES + 1)]
    monitor_cols = [f"m{s}" for s in cfg.MONITOR_SLOTS]

    is_center = (log[trim_cols] == 0).all(axis=1)
    center_avg = log.loc[is_center, monitor_cols].mean()
    corner_avg = log.loc[~is_center, monitor_cols].mean()
    return center_avg - corner_avg


def true_sensitivity(recipe, reactor, step=0.5):
    slots = np.array(cfg.MONITOR_SLOTS)
    ox = furnace.oxidant_factor()[slots - 1]
    site_factors = [1 + cfg.WIW_EDGE_FRAC * r ** 2 for r, angle in cfg.SITES]
    wiw = np.mean(site_factors)

    def monitor_thickness(trim):
        temps = furnace.slot_temperatures(recipe["temp_c"], reactor, trim)[slots - 1]
        x_nm = thickness(recipe["time_hr"] * ox, temps, recipe["ambient"])
        return x_nm * 10 * wiw

    S = np.zeros((len(slots), cfg.N_ZONES))
    for j in range(cfg.N_ZONES):
        nudge = np.zeros(cfg.N_ZONES)
        nudge[j] = step
        S[:, j] = (monitor_thickness(nudge) - monitor_thickness(-nudge)) / (2 * step)
    return S


def plot_sensitivity(S, save_to):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    im = ax.imshow(S, cmap="Blues", vmin=0)

    for i in range(S.shape[0]):
        for j in range(S.shape[1]):
            if S[i, j] > 0.6 * S.max():
                color = "white"
            else:
                color = "black"
            ax.text(j, i, f"{S[i, j]:.1f}", ha="center", va="center", color=color)

    zone_labels = [f"Zone {z}" for z in range(1, cfg.N_ZONES + 1)]
    monitor_labels = [f"Monitor slot {s}" for s in cfg.MONITOR_SLOTS]
    ax.set_xticks(range(cfg.N_ZONES), zone_labels)
    ax.set_yticks(range(len(cfg.MONITOR_SLOTS)), monitor_labels)
    ax.set_xlabel("Heater zone trimmed")
    ax.set_title("Zone sensitivity from DOE (Å per °C)")
    fig.colorbar(im, ax=ax, label="Å per °C")
    fig.tight_layout()
    fig.savefig(save_to, dpi=150)
    print(f"Saved {save_to}")


if __name__ == "__main__":
    recipe = cfg.BASE_RECIPE
    design = build_design()
    print(f"2^(5-1) fractional factorial + {N_CENTER} center runs = {len(design)} runs, "
          f"zones at ±{STEP_C:g} °C, Reactor {REACTOR}\n")

    log = run_experiment(design, recipe, REACTOR, SEED)
    S, baseline, r2, resid_sd = fit_sensitivity(log)
    S_true = true_sensitivity(recipe, REACTOR)

    names = [f"slot {s}" for s in cfg.MONITOR_SLOTS]
    zones = [f"zone {z}" for z in range(1, cfg.N_ZONES + 1)]

    print("Sensitivity matrix from the DOE (Å per °C)")
    table = pd.DataFrame(S, index=names, columns=zones)
    print(table.round(2).to_string())

    print("\nFit quality per monitor")
    for i in range(len(names)):
        print(f"  {names[i]:<9} R² = {r2[i]:.3f}   residual sd = {resid_sd[i]:.1f} Å")

    print("\nCurvature check (center runs minus corner runs, Å). Near 0 = linear is fine:")
    gaps = curvature_check(log)
    for i in range(len(names)):
        print(f"  {names[i]:<9} {gaps.iloc[i]:+.1f}")

    biggest_miss = np.abs(S - S_true).max()
    diag_error = np.abs(np.diag(S) / np.diag(S_true) - 1).max() * 100
    print(f"\nDOE vs true physics: biggest miss {biggest_miss:.2f} Å/°C, "
          f"own-zone values off by {diag_error:.1f}% at most")

    DATA.mkdir(exist_ok=True)
    PLOTS.mkdir(exist_ok=True)
    log.to_csv(DATA / "doe_runs.csv", index=False)

    matrix = pd.DataFrame(S, index=cfg.MONITOR_SLOTS, columns=range(1, cfg.N_ZONES + 1))
    matrix.to_csv(DATA / "sensitivity_matrix.csv", index_label="monitor_slot")

    print(f"\nSaved {DATA / 'doe_runs.csv'}")
    print(f"Saved {DATA / 'sensitivity_matrix.csv'}")
    plot_sensitivity(S, PLOTS / "sensitivity_matrix.png")