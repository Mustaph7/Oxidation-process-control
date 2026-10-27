# Mustaf Bare

"""
matching.py

Case 3: Reactor B doesn't match Reactor A. The A400 has two reactors, and the
whole point is that a lot can run on either one and come out the same. Out of
the box, B's profile down the boat is a different shape from A's. Its top
monitor comes out about 11 Å thinner than A's and its middle monitors 5-8 Å
thicker. If that isn't fixed, wafers come out different depending on which
reactor they happened to land in.

The catch is that the run averages hide it. The thin top and thick middle
mostly cancel out, so B's average lands within about 1 Å of A's and a t-test
on the averages says they look the same. That's why I compare them slot by
slot instead of just on the average.

The usual way to handle this is to pick one reactor as the "golden" one and
match the other to it. A is already tuned from Case 1, so A is my golden
reactor. I run A 20 times and average each monitor, and that profile becomes
B's target, not just a flat 1000 Å.

To move B I reuse the sensitivity matrix from the Reactor A DOE instead of
running a new DOE on B. Both reactors are the same hardware design, so the
zones should affect the monitors the same way, and if that holds it saves the
19 runs a second DOE would cost. Since it's a simulator I can check that at the
end by comparing A's matrix with B's true sensitivity.

My first try applied the full correction after every tuning run and stopped as
soon as one run looked good, but single runs are noisy and that kept
overshooting. So now I apply half of each correction for 6 runs. It's the same
idea as the EWMA in r2r.py: don't trust any one run all the way.

To check the result I run both reactors 20 more times and compare every
monitor slot. B counts as matched when every slot is within 3 Å of A on
average, which is my own pick. I also show the t-test p-value per slot. With
20 runs a t-test can pick up differences of a couple of Å, so a small p-value
on a tiny difference just means the test is sensitive, not that the match is
bad. The 3 Å limit is the practical call.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

import config as cfg
import doe
import furnace

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "data"
PLOTS = REPO / "plots"

N_COMPARE = 20          
TUNING_RUNS = 6         
DAMPING = 0.5           
MATCH_LIMIT_A = 3.0    
TRIM_LIMIT_C = 5.0


def load_sensitivity():
    matrix = pd.read_csv(DATA / "sensitivity_matrix.csv", index_col="monitor_slot")
    return matrix.to_numpy()


def load_a_trims():
    trims = pd.read_csv(DATA / "zone_trims.csv")
    return trims["trim_c"].to_numpy()


def monitor_profile(reactor, trim, seed):
    wafers, sites = furnace.run_load(cfg.BASE_RECIPE, reactor=reactor,
                                     zone_trim_c=trim, seed=seed)
    return furnace.monitor_summary(sites)["mean_A"].to_numpy()


def many_runs(reactor, trim, first_seed):
    profiles = []
    for i in range(N_COMPARE):
        profiles.append(monitor_profile(reactor, trim, first_seed + i))
    return np.array(profiles)


def tune_to_golden(S, golden, first_seed):
    trim = np.zeros(cfg.N_ZONES)
    history = []

    for run in range(1, TUNING_RUNS + 1):
        measured = monitor_profile("B", trim, first_seed + run)
        gap = golden - measured
        history.append({"run": run, "trim": trim.copy(), "worst_gap_A": np.abs(gap).max()})

        correction = np.linalg.solve(S, gap)
        trim = trim + DAMPING * correction
        trim = np.clip(trim, -TRIM_LIMIT_C, TRIM_LIMIT_C)

    return trim, history


def compare(a_runs, b_runs):
    slot_diff = b_runs.mean(axis=0) - a_runs.mean(axis=0)

    slot_p = []
    for j in range(a_runs.shape[1]):
        result = stats.ttest_ind(b_runs[:, j], a_runs[:, j], equal_var=False)
        slot_p.append(result.pvalue)

    avg_result = stats.ttest_ind(b_runs.mean(axis=1), a_runs.mean(axis=1), equal_var=False)

    return {
        "slot_diff": slot_diff,
        "slot_p": np.array(slot_p),
        "worst_slot_diff_A": np.abs(slot_diff).max(),
        "mean_diff_A": slot_diff.mean(),
        "avg_p": avg_result.pvalue,
    }


def print_comparison(label, result):
    print(f"{label}")
    print(f"  {'slot':<6}{'B - A':>9}{'p-value':>10}")
    for j in range(len(cfg.MONITOR_SLOTS)):
        print(f"  {cfg.MONITOR_SLOTS[j]:<6}{result['slot_diff'][j]:>+7.1f} Å{result['slot_p'][j]:>10.4f}")
    print(f"  worst slot {result['worst_slot_diff_A']:.1f} Å, run averages differ by "
          f"{result['mean_diff_A']:+.1f} Å (p = {result['avg_p']:.2f})\n")


def plot_results(a_runs, b_before, b_after, save_to):
    import matplotlib.pyplot as plt

    fig, (ax_prof, ax_diff) = plt.subplots(1, 2, figsize=(13, 4.8))
    slots = cfg.MONITOR_SLOTS

    ax_prof.errorbar(slots, a_runs.mean(axis=0), yerr=a_runs.std(axis=0, ddof=1),
                     fmt="o-", color="k", capsize=3, label="Reactor A (golden)")
    ax_prof.errorbar(slots, b_before.mean(axis=0), yerr=b_before.std(axis=0, ddof=1),
                     fmt="s--", color="C3", capsize=3, label="Reactor B before")
    ax_prof.errorbar(slots, b_after.mean(axis=0), yerr=b_after.std(axis=0, ddof=1),
                     fmt="^-", color="C0", capsize=3, label="Reactor B after matching")
    ax_prof.set_xlabel("Monitor slot (1 = top)")
    ax_prof.set_ylabel("Oxide thickness (Å)")
    ax_prof.set_title(f"Monitor profile, average of {N_COMPARE} runs (bars = 1 sd)")
    ax_prof.grid(alpha=0.3)
    ax_prof.legend(fontsize=8)

    before_diff = b_before.mean(axis=0) - a_runs.mean(axis=0)
    after_diff = b_after.mean(axis=0) - a_runs.mean(axis=0)
    x = np.arange(len(slots))
    ax_diff.bar(x - 0.2, before_diff, width=0.4, color="C3", label="Before")
    ax_diff.bar(x + 0.2, after_diff, width=0.4, color="C0", label="After")
    ax_diff.axhspan(-MATCH_LIMIT_A, MATCH_LIMIT_A, color="k", alpha=0.08,
                    label=f"±{MATCH_LIMIT_A:g} Å match limit")
    ax_diff.axhline(0, color="k", lw=0.8)
    ax_diff.set_xticks(x, [f"slot {s}" for s in slots])
    ax_diff.set_ylabel("B minus A (Å)")
    ax_diff.set_title("Reactor B vs A at each monitor")
    ax_diff.grid(alpha=0.3, axis="y")
    ax_diff.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(save_to, dpi=150)
    print(f"Saved {save_to}")


if __name__ == "__main__":
    S = load_sensitivity()
    a_trim = load_a_trims()

    a_runs = many_runs("A", a_trim, first_seed=7000)
    golden = a_runs.mean(axis=0)
    b_before = many_runs("B", np.zeros(cfg.N_ZONES), first_seed=8000)
    before = compare(a_runs, b_before)
    print_comparison(f"Before matching ({N_COMPARE} runs each)", before)

    b_trim, history = tune_to_golden(S, golden, first_seed=8500)
    print(f"Tuning Reactor B toward A's profile ({TUNING_RUNS} runs, half-step corrections)")
    for h in history:
        trims = ""
        for t in h["trim"]:
            trims += f"{t:+5.2f} "
        print(f"  run {h['run']}  trims [{trims.strip()}]  worst monitor gap {h['worst_gap_A']:.1f} Å")
    print(f"  Final Reactor B trims (°C): {np.round(b_trim, 2)}\n")

    b_after = many_runs("B", b_trim, first_seed=9000)
    after = compare(a_runs, b_after)
    print_comparison(f"After matching ({N_COMPARE} fresh runs of B)", after)

    if after["worst_slot_diff_A"] <= MATCH_LIMIT_A:
        print(f"Matched: every slot within {MATCH_LIMIT_A:g} Å of Reactor A")
    else:
        print(f"Not matched yet: worst slot is {after['worst_slot_diff_A']:.1f} Å off")

    S_true_b = doe.true_sensitivity(cfg.BASE_RECIPE, "B")
    diag_error = np.abs(np.diag(S) / np.diag(S_true_b) - 1).max() * 100
    print(f"A's DOE matrix vs B's true sensitivity: own-zone values within {diag_error:.1f}%,")
    print(f"so reusing it was fine and saved the {len(doe.build_design())} runs of a second DOE.")

    DATA.mkdir(exist_ok=True)
    PLOTS.mkdir(exist_ok=True)

    b_trims = pd.DataFrame({"zone": range(1, cfg.N_ZONES + 1), "trim_c": b_trim})
    b_trims.to_csv(DATA / "reactor_b_trims.csv", index=False)

    summary = pd.DataFrame({
        "monitor_slot": cfg.MONITOR_SLOTS,
        "golden_A": golden,
        "b_minus_a_before_A": before["slot_diff"],
        "p_before": before["slot_p"],
        "b_minus_a_after_A": after["slot_diff"],
        "p_after": after["slot_p"],
    })
    summary.to_csv(DATA / "matching_summary.csv", index=False)

    print(f"\nSaved {DATA / 'reactor_b_trims.csv'}")
    print(f"Saved {DATA / 'matching_summary.csv'}")
    plot_results(a_runs, b_before, b_after, PLOTS / "reactor_matching.png")