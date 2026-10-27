#Mustaf Bare



"""
zone_tuning.py

Case 1: the top of the boat comes out thin. On the untuned furnace the zone 1
monitor grows about 984 Å while the middle of the boat sits around 1000-1006 Å,
so the load has a wafer-to-wafer spread I want to flatten out.

To fix it I use the sensitivity matrix from doe.py. It tells me how many Å each
monitor moves for every °C I change on each zone, so if I know how far off each
monitor is, I can solve for the zone trims that cancel all five errors at the
same time. Doing all five together matters because the zones bleed into each
other. If I turn up zone 1 to fix slot 8, slot 30 goes up a bit too, and the
matrix already knows that.

The matrix came out of a noisy DOE, so it isn't perfect. After every correction
I run the furnace again, measure the monitors, and correct again if I need to,
the same way you'd verify a tuning change on a real tool before releasing it.

To check if this is actually any better, I compare it to a manual way of doing
it where each run I only touch the zone whose monitor is furthest off, using
the rule of thumb that 1 °C is about 10 Å. That's just my stand-in for trial
and error, not how any particular fab does it.

I'm calling it "in spec" when all 5 monitors are within 10 Å of each other and
their average is within 5 Å of the 1000 Å target. I picked both of those numbers.
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
MAX_RUNS = 15
TRIM_LIMIT_C = 5.0      
SPEC_RANGE_A = 10.0     
SPEC_MEAN_A = 5.0       
RULE_OF_THUMB = 10.0   
SEED = 500


def load_sensitivity():
    # doe.py saved this, rows are monitors and columns are zones
    matrix = pd.read_csv(DATA / "sensitivity_matrix.csv", index_col="monitor_slot")
    return matrix.to_numpy()


def measure(trim, run_number, seed=SEED):
    wafers, sites = furnace.run_load(cfg.BASE_RECIPE, reactor=REACTOR,
                                     zone_trim_c=trim, seed=seed + run_number)
    monitors = furnace.monitor_summary(sites)["mean_A"].to_numpy()
    return wafers, monitors


def in_spec(monitors):
    spread = monitors.max() - monitors.min()
    off_target = abs(monitors.mean() - cfg.TARGET_A)
    return spread <= SPEC_RANGE_A and off_target <= SPEC_MEAN_A


def matrix_tuning(S, seed=SEED):
    trim = np.zeros(cfg.N_ZONES)
    history = []

    for run in range(1, MAX_RUNS + 1):
        wafers, monitors = measure(trim, run, seed)
        history.append(record("matrix", run, trim, monitors))
        if in_spec(monitors):
            break

        error = cfg.TARGET_A - monitors
        correction = np.linalg.solve(S, error)
        trim = trim + correction
        trim = np.clip(trim, -TRIM_LIMIT_C, TRIM_LIMIT_C)

    return pd.DataFrame(history), trim, wafers


def manual_tuning(seed=SEED):
    trim = np.zeros(cfg.N_ZONES)
    history = []

    for run in range(1, MAX_RUNS + 1):
        wafers, monitors = measure(trim, run, seed)
        history.append(record("manual", run, trim, monitors))
        if in_spec(monitors):
            break

        error = cfg.TARGET_A - monitors
        worst = np.argmax(np.abs(error))
        trim[worst] = trim[worst] + error[worst] / RULE_OF_THUMB
        trim = np.clip(trim, -TRIM_LIMIT_C, TRIM_LIMIT_C)

    return pd.DataFrame(history), trim, wafers


def record(method, run, trim, monitors):
    row = {"method": method, "run": run}
    for z in range(cfg.N_ZONES):
        row[f"trim_z{z + 1}"] = trim[z]
    for i in range(len(cfg.MONITOR_SLOTS)):
        row[f"m{cfg.MONITOR_SLOTS[i]}"] = monitors[i]
    row["monitor_range_A"] = monitors.max() - monitors.min()
    row["monitor_mean_A"] = monitors.mean()
    row["in_spec"] = in_spec(monitors)
    return row


def runs_needed(history):
    hits = history[history["in_spec"]]
    if len(hits) == 0:
        return None
    return int(hits["run"].iloc[0])


def compare_many(S, n_trials=200):
    results = {"matrix": [], "manual": []}
    for trial in range(n_trials):
        seed = 10000 + trial * 50
        matrix_hist = matrix_tuning(S, seed)[0]
        manual_hist = manual_tuning(seed)[0]
        results["matrix"].append(runs_needed(matrix_hist))
        results["manual"].append(runs_needed(manual_hist))

    print(f"\nRepeated {n_trials} times with different noise")
    for method in ["matrix", "manual"]:
        made_it = []
        for r in results[method]:
            if r is not None:
                made_it.append(r)
        made_it = np.array(made_it)

        failed = n_trials - len(made_it)
        in_two = (made_it <= 2).mean() * 100
        print(f"  {method:<7} avg {made_it.mean():.2f} runs to spec, done in 2 runs {in_two:.0f}% of the time, "
              f"worst {made_it.max()} runs, never made it {failed} times")


def true_uniformity(wafers):
    real = wafers[wafers["type"] != "dummy"]["thickness_A"]
    return furnace.uniformity(real)


def print_history(history):
    for row in history.itertuples():
        trims = ""
        for z in range(1, cfg.N_ZONES + 1):
            trims += f"{getattr(row, f'trim_z{z}'):+5.2f} "
        trims = trims.strip()

        if row.in_spec:
            status = "IN SPEC"
        else:
            status = ""

        print(f"  run {row.run:>2}  trims [{trims}]  "
              f"range {row.monitor_range_A:5.1f} Å  mean {row.monitor_mean_A:7.1f} Å  {status}")


def plot_results(before, after, matrix_hist, manual_hist, save_to):
    import matplotlib.pyplot as plt

    fig, (ax_boat, ax_conv) = plt.subplots(1, 2, figsize=(13, 4.8),
                                           gridspec_kw={"width_ratios": [1.6, 1]})

    before_real = before[before["type"] != "dummy"]
    after_real = after[after["type"] != "dummy"]
    ax_boat.plot(before_real["slot"], before_real["thickness_A"], ".", color="C3", ms=4, label="Before tuning")
    ax_boat.plot(after_real["slot"], after_real["thickness_A"], ".", color="C0", ms=4, label="After tuning")

    ax_boat.axhline(cfg.TARGET_A, color="k", lw=1, label="Target")
    ax_boat.axhspan(cfg.TARGET_A - SPEC_MEAN_A, cfg.TARGET_A + SPEC_MEAN_A,
                    color="k", alpha=0.07, label=f"Target ±{SPEC_MEAN_A:g} Å")
    for z in range(1, cfg.N_ZONES):
        ax_boat.axvline(z * cfg.SLOTS_PER_ZONE + 0.5, color="k", ls=":", lw=0.8)

    ax_boat.set_xlabel("Boat slot (1 = top)")
    ax_boat.set_ylabel("Oxide thickness (Å)")
    ax_boat.set_title("Every wafer in the boat, Reactor A")
    ax_boat.grid(alpha=0.3)
    ax_boat.legend(fontsize=8)

    ax_conv.plot(matrix_hist["run"], matrix_hist["monitor_range_A"], "o-", label="Sensitivity matrix")
    ax_conv.plot(manual_hist["run"], manual_hist["monitor_range_A"], "s-", label="Manual (worst zone first)")
    ax_conv.axhline(SPEC_RANGE_A, color="k", ls="--", lw=1, label="Spec")

    most_runs = max(len(matrix_hist), len(manual_hist))
    ax_conv.set_xticks(range(1, most_runs + 1))
    ax_conv.set_xlabel("Furnace run")
    ax_conv.set_ylabel("Monitor range (Å)")
    ax_conv.set_title("Runs needed to get in spec")
    ax_conv.grid(alpha=0.3)
    ax_conv.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(save_to, dpi=150)
    print(f"Saved {save_to}")


if __name__ == "__main__":
    S = load_sensitivity()
    print(f"Tuning Reactor {REACTOR} to {cfg.TARGET_A:g} Å. Spec: monitor range <= {SPEC_RANGE_A:g} Å, "
          f"mean within ±{SPEC_MEAN_A:g} Å\n")

    matrix_hist, final_trim, after = matrix_tuning(S)
    print("Sensitivity matrix method")
    print_history(matrix_hist)

    manual_hist, manual_trim, manual_after = manual_tuning()
    print("\nManual method (worst zone first, 10 Å per °C)")
    print_history(manual_hist)

    before, before_monitors = measure(np.zeros(cfg.N_ZONES), 1)
    u_before = true_uniformity(before)
    u_after = true_uniformity(after)
    reduction = (1 - u_after["range_A"] / u_before["range_A"]) * 100

    print("\nResult for this session")
    print(f"  Matrix method: {runs_needed(matrix_hist)} runs   Manual method: {runs_needed(manual_hist)} runs")
    print(f"  Final zone trims (°C): {np.round(final_trim, 2)}")
    print(f"  All 100 wafers before: range {u_before['range_A']:.1f} Å, nonuniformity {u_before['nu_pct']:.2f}%, "
          f"mean {u_before['mean_A']:.1f} Å")
    print(f"  All 100 wafers after:  range {u_after['range_A']:.1f} Å, nonuniformity {u_after['nu_pct']:.2f}%, "
          f"mean {u_after['mean_A']:.1f} Å")
    print(f"  Range reduced by {reduction:.0f}%")

    compare_many(S)

    DATA.mkdir(exist_ok=True)
    PLOTS.mkdir(exist_ok=True)

    log = pd.concat([matrix_hist, manual_hist])
    log.to_csv(DATA / "zone_tuning_log.csv", index=False)

    trims = pd.DataFrame({"zone": range(1, cfg.N_ZONES + 1), "trim_c": final_trim})
    trims.to_csv(DATA / "zone_trims.csv", index=False)

    print(f"\nSaved {DATA / 'zone_tuning_log.csv'}")
    print(f"Saved {DATA / 'zone_trims.csv'}")
    plot_results(before, after, matrix_hist, manual_hist, PLOTS / "zone_tuning.png")