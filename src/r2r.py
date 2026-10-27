# Mustaf Bare

"""
r2r.py

Case 2, part 2: fixing the drift. SPC told me something was wrong around run
40, but by itself it doesn't fix anything, and left alone 34 of the 100 runs
went out of spec. Here I add a run-to-run (R2R) controller that adjusts the
oxidation time after every run so the drift never gets the chance to push
wafers out of spec.

The controller uses an EWMA, an exponentially weighted moving average. After
each run I take how far the monitors came out from target, after removing
whatever change I made to the time on purpose, and blend it with my running
estimate: new estimate = lambda * latest + (1 - lambda) * old estimate. Then I
set the next run's time to cancel that estimate. Lambda decides how much I
trust the newest run. A high lambda reacts fast but chases noise, a low lambda
is smooth but slow to catch up.

To turn a thickness error into a time change I need to know how many Å one
more hour of oxidation adds. I get that from deal_grove.py at the recipe
conditions (dry, 1000 °C, around 3.2 hr). It's an approximation, since the
real furnace has zone trims and gas depletion on top, but the EWMA keeps
correcting so it doesn't need to be exact.

I run the exact same 100 runs as spc.py, same drift and same noise, once with
no control and once for a few different lambdas, so any difference comes from
the controller and not from luck.

To score it I use Ppk, which is (distance from the mean to the closest spec
limit) / (3 x standard deviation) over all 100 runs. It's the same formula as
Cpk, but Cpk uses short-term sigma and Ppk uses the overall sigma, and with
drift in the data the overall one is the honest choice. 1.33 or higher is the
usual "capable" bar.

One thing to watch: once the controller is on, the thickness chart stays flat
even though the heaters keep drifting. The drift doesn't go away, it just
shows up in the time adjustments instead. So on a real tool you'd chart the
controller's adjustments too, otherwise you'd never know the heaters need work.
"""

from pathlib import Path

import pandas as pd

import config as cfg
import spc
from deal_grove import thickness

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "data"
PLOTS = REPO / "plots"

LAMBDAS = [0.1, 0.3, 0.6]    
MAIN_LAMBDA = 0.3            
TIME_LIMIT = 0.15            


def thickness_per_hour():
    r = cfg.BASE_RECIPE
    t = r["time_hr"]
    step = 0.01
    up = thickness(t + step, r["temp_c"], r["ambient"])
    down = thickness(t - step, r["temp_c"], r["ambient"])
    return float((up - down) / (2 * step)) * 10   # nm -> Å


def run_with_control(trim, lam, gain):
    base_time = cfg.BASE_RECIPE["time_hr"]
    estimate = 0.0          
    time_hr = base_time
    rows = []

    for run in range(1, spc.N_RUNS + 1):
        measured = spc.run_one(run, trim, time_hr=time_hr)

        my_change = gain * (time_hr - base_time)
        offset = measured - my_change - cfg.TARGET_A

        estimate = lam * offset + (1 - lam) * estimate

        rows.append({"run": run, "time_hr": time_hr, "mean_A": measured,
                     "estimate_A": estimate, "drift_c": spc.heater_drift(run)})

        time_hr = base_time - estimate / gain
        lowest = base_time * (1 - TIME_LIMIT)
        highest = base_time * (1 + TIME_LIMIT)
        time_hr = min(max(time_hr, lowest), highest)

    return pd.DataFrame(rows)


def run_without_control(trim):
    rows = []
    for run in range(1, spc.N_RUNS + 1):
        measured = spc.run_one(run, trim)
        rows.append({"run": run, "time_hr": cfg.BASE_RECIPE["time_hr"], "mean_A": measured,
                     "estimate_A": 0.0, "drift_c": spc.heater_drift(run)})
    return pd.DataFrame(rows)


def score(log):
    values = log["mean_A"].to_numpy()
    mean = values.mean()
    sd = values.std(ddof=1)
    closest_limit = min(spc.USL_A - mean, mean - spc.LSL_A)
    ppk = closest_limit / (3 * sd)

    out_of_spec = 0
    for v in values:
        if v > spc.USL_A or v < spc.LSL_A:
            out_of_spec += 1

    return {"mean_A": mean, "sd_A": sd, "ppk": ppk, "out_of_spec": out_of_spec}


def plot_results(no_control, controlled, lam, save_to):
    import matplotlib.pyplot as plt

    fig, (ax, ax_time, ax_drift) = plt.subplots(3, 1, figsize=(11, 8.5), sharex=True,
                                                gridspec_kw={"height_ratios": [3, 1.3, 1]})

    ax.plot(no_control["run"], no_control["mean_A"], "o-", color="C3", ms=3, lw=1,
            alpha=0.8, label="No control")
    ax.plot(controlled["run"], controlled["mean_A"], "o-", color="C0", ms=3, lw=1,
            label=f"EWMA R2R control (λ = {lam})")
    ax.axhline(cfg.TARGET_A, color="k", lw=1, label="Target")
    ax.axhline(spc.USL_A, color="C3", ls="--", lw=1.2, label="Spec limits")
    ax.axhline(spc.LSL_A, color="C3", ls="--", lw=1.2)
    ax.axvline(spc.STEP_RUN, color="grey", ls=":", lw=1)
    ax.text(spc.STEP_RUN + 0.5, spc.LSL_A + 1, "+1 °C step", fontsize=8, va="bottom")
    ax.set_ylabel("Oxide thickness (Å)")
    ax.set_title("Same 100 runs, same drift and noise, with and without R2R control")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="upper left")

    ax_time.plot(controlled["run"], controlled["time_hr"] * 60, color="C0")
    ax_time.axhline(cfg.BASE_RECIPE["time_hr"] * 60, color="k", lw=1, ls="--")
    ax_time.set_ylabel("Oxidation\ntime (min)")
    ax_time.grid(alpha=0.3)

    # bottom: the real heater drift
    ax_drift.plot(controlled["run"], controlled["drift_c"], color="C3")
    ax_drift.set_ylabel("Heater\ndrift (°C)")
    ax_drift.set_xlabel("Run")
    ax_drift.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(save_to, dpi=150)
    print(f"Saved {save_to}")


if __name__ == "__main__":
    trim = spc.load_zone_trims()
    gain = thickness_per_hour()
    print(f"Controller gain from Deal-Grove: {gain:.0f} Å per hour ({gain / 60:.2f} Å per minute)")
    print(f"Spec {spc.LSL_A:g}-{spc.USL_A:g} Å, same drift and noise as spc.py\n")

    results = []
    no_control = run_without_control(trim)
    s = score(no_control)
    s["method"] = "No control"
    results.append(s)

    logs = {}
    for lam in LAMBDAS:
        log = run_with_control(trim, lam, gain)
        logs[lam] = log
        s = score(log)
        s["method"] = f"EWMA λ={lam}"
        results.append(s)

    print(f"{'method':<14}{'mean Å':>9}{'sd Å':>8}{'Ppk':>7}{'out of spec':>14}")
    for s in results:
        print(f"{s['method']:<14}{s['mean_A']:>9.1f}{s['sd_A']:>8.2f}{s['ppk']:>7.2f}"
              f"{s['out_of_spec']:>10} / {spc.N_RUNS}")

    main = logs[MAIN_LAMBDA]
    center, sigma = spc.control_limits(no_control["mean_A"].iloc[:spc.BASELINE_RUNS])

    def count_flagged_runs(values):
        flagged = []
        for run, rule in spc.western_electric(values, center, sigma):
            if run not in flagged:
                flagged.append(run)
        return len(flagged)

    flagged_controlled = count_flagged_runs(main["mean_A"])
    flagged_uncontrolled = count_flagged_runs(no_control["mean_A"])
    print(f"\nRuns flagged by SPC on the thickness chart: {flagged_uncontrolled} with no control, "
          f"{flagged_controlled} with λ={MAIN_LAMBDA}")

    time_change = (main["time_hr"].iloc[-1] - cfg.BASE_RECIPE["time_hr"]) * 60
    print(f"By run {spc.N_RUNS} the controller has cut {abs(time_change):.1f} min off the recipe "
          f"to make up for the drift.")
    print("That's the warning sign to watch once R2R is on: the thickness looks fine,")
    print("but the knob keeps moving one way because the heaters are still drifting.")

    DATA.mkdir(exist_ok=True)
    PLOTS.mkdir(exist_ok=True)
    main.to_csv(DATA / "r2r_runs.csv", index=False)
    summary = pd.DataFrame(results)
    summary = summary[["method", "mean_A", "sd_A", "ppk", "out_of_spec"]]
    summary.to_csv(DATA / "r2r_summary.csv", index=False)
    print(f"\nSaved {DATA / 'r2r_runs.csv'}")
    print(f"Saved {DATA / 'r2r_summary.csv'}")
    plot_results(no_control, main, MAIN_LAMBDA, PLOTS / "r2r_control.png")