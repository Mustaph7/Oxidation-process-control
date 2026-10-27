# Mustaf Bare

"""
app.py

A small web front end for the furnace simulator so people can play with it
without running any Python. It uses the same code as the rest of the project
(furnace.py, deal_grove.py, the SPC rules and so on), so what you see here is
the real simulator, not a copy of it.

Three tabs, one per case. The Furnace tab lets you change the recipe and the
zone trims and watch the boat profile move. The Drift tab runs 100 runs with
heater drift, with R2R control on or off. The Matching tab tunes Reactor B
toward Reactor A and lets you see what the damping setting does.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO / "src"))  

import config as cfg
import furnace
import spc
from deal_grove import thickness

DATA = REPO / "data"
GITHUB = "https://github.com/Mustaph7/Oxidation-process-control"

st.set_page_config(page_title="Furnace Uniformity & Drift Control", layout="wide")



def load_trims(name):
    path = DATA / name
    if not path.exists():
        return np.zeros(cfg.N_ZONES)
    return pd.read_csv(path)["trim_c"].to_numpy()


def load_sensitivity():
    matrix = pd.read_csv(DATA / "sensitivity_matrix.csv", index_col="monitor_slot")
    return matrix.to_numpy()


def thickness_per_hour(recipe):
    t = recipe["time_hr"]
    step = 0.01
    up = thickness(t + step, recipe["temp_c"], recipe["ambient"])
    down = thickness(t - step, recipe["temp_c"], recipe["ambient"])
    return float((up - down) / (2 * step)) * 10


def ppk(values, lsl, usl):
    mean = np.mean(values)
    sd = np.std(values, ddof=1)
    return min(usl - mean, mean - lsl) / (3 * sd)


def drift_at(run, start, rate, step_run, step_c):
    drift = 0.0
    if run > start:
        drift = drift + (run - start) * rate
    if run >= step_run:
        drift = drift + step_c
    return drift


@st.cache_data
def run_drift(rate, step_run, step_c, lam, seed, n_runs=100, start=20):
    trim = load_trims("zone_trims.csv")
    base = cfg.BASE_RECIPE.copy()
    gain = thickness_per_hour(base)

    plain = []
    controlled = []
    times = []
    drifts = []
    estimate = 0.0
    time_hr = base["time_hr"]

    for run in range(1, n_runs + 1):
        drift = drift_at(run, start, rate, step_run, step_c)
        drifts.append(drift)

        w, sites = furnace.run_load(base, "A", trim, drift_c=drift, seed=seed + run)
        plain.append(furnace.monitor_summary(sites)["mean_A"].mean())

        recipe = base.copy()
        recipe["time_hr"] = time_hr
        w, sites = furnace.run_load(recipe, "A", trim, drift_c=drift, seed=seed + run)
        measured = furnace.monitor_summary(sites)["mean_A"].mean()
        controlled.append(measured)
        times.append(time_hr)

        offset = measured - gain * (time_hr - base["time_hr"]) - cfg.TARGET_A
        estimate = lam * offset + (1 - lam) * estimate
        time_hr = base["time_hr"] - estimate / gain
        time_hr = min(max(time_hr, base["time_hr"] * 0.85), base["time_hr"] * 1.15)

    return np.array(plain), np.array(controlled), np.array(times), np.array(drifts)


@st.cache_data
def run_matching(damping, tuning_runs, n_compare=20):
    S = load_sensitivity()
    a_trim = load_trims("zone_trims.csv")

    def profile(reactor, trim, seed):
        w, sites = furnace.run_load(cfg.BASE_RECIPE, reactor, trim, seed=seed)
        return furnace.monitor_summary(sites)["mean_A"].to_numpy()

    a_runs = np.array([profile("A", a_trim, 7000 + i) for i in range(n_compare)])
    golden = a_runs.mean(axis=0)
    b_before = np.array([profile("B", np.zeros(cfg.N_ZONES), 8000 + i) for i in range(n_compare)])

    trim = np.zeros(cfg.N_ZONES)
    for run in range(1, tuning_runs + 1):
        gap = golden - profile("B", trim, 8500 + run)
        trim = np.clip(trim + damping * np.linalg.solve(S, gap), -5, 5)

    b_after = np.array([profile("B", trim, 9000 + i) for i in range(n_compare)])
    return a_runs, b_before, b_after, trim


# ---------------------------------------------------------------- page

st.title("Batch Furnace Uniformity & Drift Control")
st.write(
    "A physics-based simulator of a dual-reactor vertical oxidation furnace, used to work "
    "through three process engineering problems: boat uniformity, heater drift and reactor "
    f"matching. Everything is simulated, and the code and write-up are on [GitHub]({GITHUB})."
)

tab_furnace, tab_drift, tab_match = st.tabs(["Furnace", "Drift & R2R control", "Reactor matching"])


# ---------------------------------------------------------------- tab 1

with tab_furnace:
    st.subheader("Run one load and see the boat profile")
    st.write("Change the recipe or the zone trims and watch how thickness changes down the boat. "
             "Only the 5 monitor wafers get measured, the same as on a real tool.")

    left, right = st.columns([1, 2.4])

    with left:
        reactor = st.radio("Reactor", cfg.REACTORS, horizontal=True)
        ambient = st.radio("Ambient", ["dry", "wet"], horizontal=True)
        temp_c = st.slider("Temperature (°C)", 900, 1100, 1000, 5)
        time_min = st.slider("Time (min)", 10, 400, 192, 1)
        seed = st.number_input("Noise seed", 0, 10_000, 42, 1)

        tuned_file = "zone_trims.csv" if reactor == "A" else "reactor_b_trims.csv"
        use_tuned = st.checkbox("Start from the tuned trims (Case 1 / Case 3)", value=False)
        start = load_trims(tuned_file) if use_tuned else np.zeros(cfg.N_ZONES)

        st.caption("Zone trims (°C), zone 1 is the top of the boat")
        trims = []
        for z in range(cfg.N_ZONES):
            trims.append(st.slider(f"Zone {z + 1}", -5.0, 5.0, float(round(start[z], 1)), 0.1,
                                   key=f"trim_{reactor}_{use_tuned}_{z}"))

    recipe = {"ambient": ambient, "temp_c": float(temp_c), "time_hr": time_min / 60}
    wafers, sites = furnace.run_load(recipe, reactor, np.array(trims), seed=int(seed))
    monitors = furnace.monitor_summary(sites)
    real = wafers[wafers["type"] != "dummy"]
    u = furnace.uniformity(real["thickness_A"])

    with right:
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Mean (all wafers)", f"{u['mean_A']:.1f} Å")
        m2.metric("Wafer-to-wafer range", f"{u['range_A']:.1f} Å")
        m3.metric("Nonuniformity", f"{u['nu_pct']:.2f} %")
        m4.metric("Monitor range", f"{monitors['mean_A'].max() - monitors['mean_A'].min():.1f} Å")

        fig, (ax_t, ax_x) = plt.subplots(2, 1, figsize=(9, 6), sharex=True,
                                         gridspec_kw={"height_ratios": [1, 2]})
        ax_t.plot(wafers["slot"], wafers["temp_c"], color="C3")
        ax_t.set_ylabel("Temperature (°C)")
        ax_x.plot(real["slot"], real["thickness_A"], ".", color="C0", ms=4, label="Every wafer (true)")
        ax_x.plot(monitors["slot"], monitors["mean_A"], "o", color="k", ms=8, label="Monitors (measured)")
        ax_x.set_ylabel("Oxide thickness (Å)")
        ax_x.set_xlabel("Boat slot (1 = top)")
        for ax in (ax_t, ax_x):
            for z in range(1, cfg.N_ZONES):
                ax.axvline(z * cfg.SLOTS_PER_ZONE + 0.5, color="k", ls=":", lw=0.8)
            ax.axvspan(0.5, 5.5, color="grey", alpha=0.15)
            ax.axvspan(105.5, 110.5, color="grey", alpha=0.15)
            ax.grid(alpha=0.3)
        ax_x.legend(fontsize=8)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        st.dataframe(monitors[["slot", "zone", "mean_A", "wiw_range_A"]].round(1),
                     hide_index=True, width="stretch")



with tab_drift:
    st.subheader("100 runs with heater drift, with and without R2R control")
    st.write("Runs 1-20 are stable and set the control limits. After that the heaters drift up, "
             "with a sudden step partway through. Both lines use the exact same noise, so the "
             "only difference is the controller.")

    c1, c2, c3, c4 = st.columns(4)
    rate = c1.slider("Drift per run (°C)", 0.0, 0.06, 0.025, 0.005, format="%.3f")
    step_c = c2.slider("Step size (°C)", 0.0, 3.0, 1.0, 0.1)
    step_run = c3.slider("Step happens at run", 25, 100, 70, 1)
    lam = c4.slider("EWMA λ", 0.05, 0.9, 0.3, 0.05)

    plain, controlled, times, drifts = run_drift(rate, step_run, step_c, lam, spc.SEED)
    runs = np.arange(1, len(plain) + 1)
    lsl, usl = spc.LSL_A, spc.USL_A

    center, sigma = spc.control_limits(plain[:20])
    flags = spc.western_electric(plain, center, sigma)
    first_warning = None
    for run, rule in flags:
        if run > 20:
            first_warning = run
            break

    oos_plain = int(((plain > usl) | (plain < lsl)).sum())
    oos_ctrl = int(((controlled > usl) | (controlled < lsl)).sum())

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Out of spec, no control", f"{oos_plain} / 100")
    m2.metric("Out of spec, R2R", f"{oos_ctrl} / 100")
    m3.metric("Ppk, no control → R2R", f"{ppk(plain, lsl, usl):.2f} → {ppk(controlled, lsl, usl):.2f}")
    m4.metric("First SPC warning", f"run {first_warning}" if first_warning else "none")

    fig, (ax, ax_time, ax_drift) = plt.subplots(3, 1, figsize=(11, 7.5), sharex=True,
                                                gridspec_kw={"height_ratios": [3, 1.2, 1]})
    ax.plot(runs, plain, "o-", color="C3", ms=3, lw=1, label="No control")
    ax.plot(runs, controlled, "o-", color="C0", ms=3, lw=1, label=f"EWMA R2R (λ = {lam})")
    ax.axhline(cfg.TARGET_A, color="k", lw=1)
    ax.axhline(usl, color="C3", ls="--", lw=1.2, label="Spec limits")
    ax.axhline(lsl, color="C3", ls="--", lw=1.2)
    ax.axhline(center + 3 * sigma, color="grey", lw=0.8, label="Control limits")
    ax.axhline(center - 3 * sigma, color="grey", lw=0.8)
    ax.axvspan(0.5, 20.5, color="grey", alpha=0.1)
    ax.set_ylabel("Oxide thickness (Å)")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(alpha=0.3)

    ax_time.plot(runs, times * 60, color="C0")
    ax_time.axhline(cfg.BASE_RECIPE["time_hr"] * 60, color="k", ls="--", lw=1)
    ax_time.set_ylabel("R2R time\n(min)")
    ax_time.grid(alpha=0.3)

    ax_drift.plot(runs, drifts, color="C3")
    ax_drift.set_ylabel("Heater\ndrift (°C)")
    ax_drift.set_xlabel("Run")
    ax_drift.grid(alpha=0.3)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.info("With R2R on, the thickness stays flat but the heaters are still drifting. "
            "The drift shows up in the controller's time adjustments (middle panel) instead, "
            "which is why you'd chart those too on a real tool.")



with tab_match:
    st.subheader("Match Reactor B to Reactor A")
    st.write("Reactor A is the tuned \"golden\" reactor. B gets tuned toward A's average monitor "
             "profile using A's sensitivity matrix. Try full-strength corrections (damping 1.0) "
             "against half steps to see why the damping is there.")

    c1, c2 = st.columns(2)
    damping = c1.slider("Damping (fraction of each correction applied)", 0.1, 1.0, 0.5, 0.1)
    tuning_runs = c2.slider("Tuning runs", 1, 10, 6, 1)

    a_runs, b_before, b_after, b_trim = run_matching(damping, tuning_runs)
    before_diff = b_before.mean(axis=0) - a_runs.mean(axis=0)
    after_diff = b_after.mean(axis=0) - a_runs.mean(axis=0)

    m1, m2, m3 = st.columns(3)
    m1.metric("Worst slot gap, before", f"{np.abs(before_diff).max():.1f} Å")
    m2.metric("Worst slot gap, after", f"{np.abs(after_diff).max():.1f} Å")
    m3.metric("Run-average gap, before", f"{before_diff.mean():+.1f} Å",
              help="Small even before matching, because B's thin top and thick middle cancel out")

    fig, (ax_p, ax_d) = plt.subplots(1, 2, figsize=(12, 4.5))
    slots = cfg.MONITOR_SLOTS
    ax_p.plot(slots, a_runs.mean(axis=0), "o-", color="k", label="Reactor A (golden)")
    ax_p.plot(slots, b_before.mean(axis=0), "s--", color="C3", label="Reactor B before")
    ax_p.plot(slots, b_after.mean(axis=0), "^-", color="C0", label="Reactor B after")
    ax_p.set_xlabel("Monitor slot (1 = top)")
    ax_p.set_ylabel("Oxide thickness (Å), 20-run average")
    ax_p.grid(alpha=0.3)
    ax_p.legend(fontsize=8)

    x = np.arange(len(slots))
    ax_d.bar(x - 0.2, before_diff, width=0.4, color="C3", label="Before")
    ax_d.bar(x + 0.2, after_diff, width=0.4, color="C0", label="After")
    ax_d.axhspan(-3, 3, color="k", alpha=0.08, label="±3 Å match limit")
    ax_d.axhline(0, color="k", lw=0.8)
    ax_d.set_xticks(x, [f"slot {s}" for s in slots])
    ax_d.set_ylabel("B minus A (Å)")
    ax_d.grid(alpha=0.3, axis="y")
    ax_d.legend(fontsize=8)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.caption(f"Reactor B zone trims after matching (°C): {np.round(b_trim, 2).tolist()}")