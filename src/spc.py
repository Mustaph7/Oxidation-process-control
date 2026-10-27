# Mustaf Bare

"""
spc.py

Case 2, part 1: catching drift. Zone tuning got the furnace flat and on target,
but a tuned furnace doesn't stay tuned. Heaters and thermocouples age, so the
real temperature slowly walks away from the setpoint and every run comes out a
little thicker than the one before. You can't see that on any single run, it
just looks like noise, so I need a way to catch it before wafers go out of spec.

I run the tuned furnace 100 times. The first 20 runs are stable and I use them
to set the control limits. After run 20 the heaters start drifting up by
0.025 °C every run, and at run 70 there's a sudden +1 °C step, like what you
might see after a maintenance event or a thermocouple swap. The drift and step
sizes are my own numbers, picked to act like a real slow drift.

Each run gets one number, the average of the 5 monitors, and that goes on an
individuals control chart. The center line and limits come from the 20 stable
runs, with sigma estimated from the average moving range divided by 1.128,
which is the normal way to do it when you only have one number per run.

Then every point gets checked against the four Western Electric rules. Rule 1
is one point past 3 sigma. Rule 2 is 2 out of 3 in a row past 2 sigma on the
same side. Rule 3 is 4 out of 5 in a row past 1 sigma on the same side, and
rule 4 is 8 in a row on the same side of the center line. Rule 1 only catches
big jumps. Rules 2-4 are the ones that catch a slow drift early, because they
look for patterns instead of single points.

The spec limits are 1000 Å ± 15 Å, which is also my pick. What I really want
to know is how many runs of warning SPC gives me before the first run goes out
of spec.
"""

from pathlib import Path

import numpy as np
import pandas as pd

import config as cfg
import furnace

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "data"
PLOTS = REPO / "plots"

REACTOR = "A"
N_RUNS = 100
BASELINE_RUNS = 20          # stable runs that set the control limits
DRIFT_START = 20            # heaters start drifting after this run
DRIFT_PER_RUN_C = 0.025     # how fast they drift up
STEP_RUN = 70               # the sudden shift happens on this run
STEP_C = 1.0                # how big the sudden shift is
LSL_A = 985.0               # lower spec limit
USL_A = 1015.0              # upper spec limit
SEED = 6000                 # picked so the 20 baseline runs match the real noise


def heater_drift(run):
    # how far off the real temperature is by this run (°C)
    drift = 0.0
    if run > DRIFT_START:
        drift = drift + (run - DRIFT_START) * DRIFT_PER_RUN_C
    if run >= STEP_RUN:
        drift = drift + STEP_C
    return drift


def load_zone_trims():
    # start from the tuned furnace that zone_tuning.py saved
    trims = pd.read_csv(DATA / "zone_trims.csv")
    return trims["trim_c"].to_numpy()


def run_one(run, trim, time_hr=None):
    # one furnace run with the drift added in, returns the average of the 5 monitors
    recipe = cfg.BASE_RECIPE.copy()
    if time_hr is not None:
        recipe["time_hr"] = time_hr

    wafers, sites = furnace.run_load(recipe, reactor=REACTOR, zone_trim_c=trim,
                                     drift_c=heater_drift(run), seed=SEED + run)
    monitors = furnace.monitor_summary(sites)
    return monitors["mean_A"].mean()


def control_limits(values):
    # sigma comes from the average moving range, not the plain standard deviation
    values = np.asarray(values)
    center = values.mean()
    moving_range = np.abs(np.diff(values))
    sigma = moving_range.mean() / 1.128
    return center, sigma


def western_electric(values, center, sigma):
    # gives back a list of (run, rule) for every point that breaks a rule
    z = (np.asarray(values) - center) / sigma   # how many sigmas off each point is
    flags = []

    for i in range(len(z)):
        run = i + 1

        # Rule 1: one point past 3 sigma
        if abs(z[i]) > 3:
            flags.append((run, 1))

        # Rule 2: 2 of the last 3 past 2 sigma, same side
        if i >= 2:
            last3 = z[i - 2:i + 1]
            if (last3 > 2).sum() >= 2 or (last3 < -2).sum() >= 2:
                flags.append((run, 2))

        # Rule 3: 4 of the last 5 past 1 sigma, same side
        if i >= 4:
            last5 = z[i - 4:i + 1]
            if (last5 > 1).sum() >= 4 or (last5 < -1).sum() >= 4:
                flags.append((run, 3))

        # Rule 4: the last 8 all on one side of the center line
        if i >= 7:
            last8 = z[i - 7:i + 1]
            if (last8 > 0).all() or (last8 < 0).all():
                flags.append((run, 4))

    return flags


def first_flag(flags, rule=None, after=0):
    # first run after a given point that broke any rule, or one specific rule
    for run, r in flags:
        if run <= after:
            continue
        if rule is None or r == rule:
            return run
    return None


def plot_chart(log, center, sigma, flags, save_to, title):
    import matplotlib.pyplot as plt

    fig, (ax, ax_drift) = plt.subplots(2, 1, figsize=(11, 7), sharex=True,
                                       gridspec_kw={"height_ratios": [3, 1]})

    runs = log["run"]
    ax.plot(runs, log["mean_A"], "o-", color="C0", ms=4, lw=1, label="Run mean (5 monitors)")

    # center line, then the 1, 2 and 3 sigma lines on both sides
    ax.axhline(center, color="k", lw=1, label="Center line")
    ax.axhline(center + sigma, color="grey", ls=":", lw=0.8)
    ax.axhline(center - sigma, color="grey", ls=":", lw=0.8)
    ax.axhline(center + 2 * sigma, color="grey", ls="--", lw=0.8)
    ax.axhline(center - 2 * sigma, color="grey", ls="--", lw=0.8)
    ax.axhline(center + 3 * sigma, color="grey", ls="-", lw=0.8)
    ax.axhline(center - 3 * sigma, color="grey", ls="-", lw=0.8)
    ax.text(N_RUNS + 1, center + 3 * sigma, "UCL", va="center", fontsize=8)
    ax.text(N_RUNS + 1, center - 3 * sigma, "LCL", va="center", fontsize=8)

    # spec limits
    ax.axhline(USL_A, color="C3", ls="--", lw=1.2, label="Spec limits")
    ax.axhline(LSL_A, color="C3", ls="--", lw=1.2)

    # A point can break more than one rule, so I only circle it with the
    # lowest-numbered one. Otherwise the circles just draw on top of each other.
    worst_rule = {}
    for run, rule in flags:
        if run not in worst_rule or rule < worst_rule[run]:
            worst_rule[run] = rule

    colors = {1: "C3", 2: "C1", 3: "C4", 4: "C2"}
    shown = []
    for run, rule in worst_rule.items():
        value = log.loc[log["run"] == run, "mean_A"].iloc[0]
        label = None
        if rule not in shown:     # only label each rule once in the legend
            label = f"Rule {rule}"
            shown.append(rule)
        ax.plot(run, value, "o", ms=9, mfc="none", mec=colors[rule], mew=1.5, label=label)

    ax.axvspan(0.5, BASELINE_RUNS + 0.5, color="grey", alpha=0.1)
    ax.text(BASELINE_RUNS / 2, LSL_A + 1, "baseline\n(sets limits)",
            ha="center", va="bottom", fontsize=8)
    ax.set_ylabel("Oxide thickness (Å)")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="lower right", ncol=2)

    # bottom: the drift that's actually happening, which you'd never see on a real tool
    drift_values = []
    for r in runs:
        drift_values.append(heater_drift(r))
    ax_drift.plot(runs, drift_values, color="C3")
    ax_drift.set_ylabel("Heater drift (°C)")
    ax_drift.set_xlabel("Run")
    ax_drift.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(save_to, dpi=150)
    print(f"Saved {save_to}")


if __name__ == "__main__":
    trim = load_zone_trims()
    print(f"Running the tuned furnace for {N_RUNS} runs. Drift starts after run {DRIFT_START} "
          f"(+{DRIFT_PER_RUN_C} °C/run), +{STEP_C:g} °C step at run {STEP_RUN}\n")

    means = []
    drifts = []
    for run in range(1, N_RUNS + 1):
        means.append(run_one(run, trim))
        drifts.append(heater_drift(run))

    log = pd.DataFrame({"run": range(1, N_RUNS + 1), "mean_A": means, "drift_c": drifts})

    # limits only come from the stable runs at the start
    center, sigma = control_limits(log["mean_A"].iloc[:BASELINE_RUNS])
    print(f"Control limits from runs 1-{BASELINE_RUNS}: center {center:.1f} Å, sigma {sigma:.2f} Å")
    print(f"  UCL {center + 3 * sigma:.1f} Å   LCL {center - 3 * sigma:.1f} Å   "
          f"spec {LSL_A:g}-{USL_A:g} Å\n")

    flags = western_electric(log["mean_A"], center, sigma)

    # write which rules each run broke into the log, like "R1 R2"
    log["flags"] = ""
    for run, rule in flags:
        i = log.index[log["run"] == run][0]
        log.loc[i, "flags"] = (log.loc[i, "flags"] + f" R{rule}").strip()

    print("First time each rule fired after the drift started:")
    for rule in [1, 2, 3, 4]:
        hit = first_flag(flags, rule, after=DRIFT_START)
        if hit is None:
            print(f"  Rule {rule}: never")
        else:
            print(f"  Rule {rule}: run {hit}")

    # anything flagged during the stable runs would be a false alarm
    false_alarms = 0
    for run, rule in flags:
        if run <= DRIFT_START:
            false_alarms += 1
    print(f"\nFlags during the stable baseline (false alarms): {false_alarms}")

    out_of_spec = log[(log["mean_A"] > USL_A) | (log["mean_A"] < LSL_A)]
    first_oos = None
    if len(out_of_spec) > 0:
        first_oos = int(out_of_spec["run"].iloc[0])
    first_warning = first_flag(flags, after=DRIFT_START)

    print(f"First SPC warning: run {first_warning}")
    print(f"First run out of spec: run {first_oos}")
    if first_warning is not None and first_oos is not None:
        print(f"SPC gave {first_oos - first_warning} runs of warning before anything went out of spec")
    print(f"Runs out of spec with no control: {len(out_of_spec)} of {N_RUNS}")

    DATA.mkdir(exist_ok=True)
    PLOTS.mkdir(exist_ok=True)
    log.to_csv(DATA / "spc_runs.csv", index=False)
    print(f"\nSaved {DATA / 'spc_runs.csv'}")
    plot_chart(log, center, sigma, flags, PLOTS / "spc_chart.png",
               "Individuals control chart, tuned furnace with heater drift (no control)")