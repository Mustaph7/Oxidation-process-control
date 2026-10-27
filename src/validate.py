# Mustaf Bare
# This file compares the model we made earlier against an outside source and tells you how far off we are.

# validate.py
#
# Checks my Deal-Grove model against outside numbers before I build
# anything else on top of it. If the model is off here, everything
# downstream (furnace, zone tuning, R2R) is off too.
#
# Right now the reference is the BYU cleanroom oxide calculator. Later
# I'll add real measured data digitized from the Deal & Grove (1965) paper.
#
# Any CSV in data/ with these columns gets picked up:
#   source, ambient, orientation, temp_c, time_hr, thickness_A
# Thickness is in angstroms because that's what BYU gives back.
#
# Run from the repo folder:  python src\validate.py

from pathlib import Path

import numpy as np
import pandas as pd

from deal_grove import thickness

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "data"
PLOTS = REPO / "plots"

COLUMNS = ["source", "ambient", "orientation", "temp_c", "time_hr", "thickness_A"]


def load_reference():
    tables = []
    for path in sorted(DATA.glob("*.csv")):
        df = pd.read_csv(path)
        # skip anything that isn't a reference file (simulation output later, etc.)
        if not all(col in df.columns for col in COLUMNS):
            continue
        tables.append(df)

    if not tables:
        raise FileNotFoundError("No reference CSVs found in data/")

    ref = pd.concat(tables, ignore_index=True)

    # rows I haven't filled in yet are blank, so drop them
    ref = ref.dropna(subset=["thickness_A"])

    # clean up so "Dry " and "dry" count as the same thing
    ref["ambient"] = ref["ambient"].str.strip().str.lower()
    ref["orientation"] = ref["orientation"].astype(str).str.strip()

    # BYU gives angstroms, my model works in nm
    ref["ref_nm"] = ref["thickness_A"] / 10
    return ref


def compare(ref):
    model_values = []
    for row in ref.itertuples():
        x = thickness(row.time_hr, row.temp_c, row.ambient, row.orientation)
        model_values.append(float(x))

    ref = ref.copy()
    ref["model_nm"] = model_values
    # positive error = my model grows thicker than the reference
    ref["error_pct"] = (ref["model_nm"] - ref["ref_nm"]) / ref["ref_nm"] * 100
    return ref


def print_results(res):
    print("\nModel vs. reference")
    print(f"{'ambient':<8}{'temp':>6}{'time':>7}{'ref nm':>10}{'model nm':>10}{'error %':>9}")
    for row in res.itertuples():
        print(f"{row.ambient:<8}{row.temp_c:>6.0f}{row.time_hr:>7.1f}"
              f"{row.ref_nm:>10.1f}{row.model_nm:>10.1f}{row.error_pct:>9.1f}")

    print("\nAverage error by ambient (absolute %)")
    for ambient in ["dry", "wet"]:
        errs = res.loc[res["ambient"] == ambient, "error_pct"].abs()
        if len(errs) > 0:
            print(f"  {ambient}: avg {errs.mean():.1f}%, worst {errs.max():.1f}%  ({len(errs)} points)")

    print(f"\nOverall average error: {res['error_pct'].abs().mean():.1f}%")


def parity_plot(res, save_to):
    # If the model were perfect, every point would sit on the diagonal.
    # Dotted lines show +/-10%, which is about what I'm aiming for.
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.scatter(res.loc[res["ambient"] == "dry", "ref_nm"],
               res.loc[res["ambient"] == "dry", "model_nm"], marker="o", label="dry")
    ax.scatter(res.loc[res["ambient"] == "wet", "ref_nm"],
               res.loc[res["ambient"] == "wet", "model_nm"], marker="s", label="wet")

    lo = 0.8 * min(res["ref_nm"].min(), res["model_nm"].min())
    hi = 1.2 * max(res["ref_nm"].max(), res["model_nm"].max())
    line = np.array([lo, hi])
    ax.plot(line, line, "k-", lw=1, label="perfect match")
    ax.plot(line, line * 1.1, "k:", lw=1, label="±10%")
    ax.plot(line, line * 0.9, "k:", lw=1)

    # log scale since thicknesses run from ~10 nm to ~1000 nm
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Reference thickness (nm)")
    ax.set_ylabel("Model thickness (nm)")
    ax.set_title("Deal-Grove model vs. reference")
    ax.grid(alpha=0.3, which="both")
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_to, dpi=150)
    print(f"Saved {save_to}")


def curves_plot(res, save_to):
    # Model growth curves with the reference points dotted on top,
    # dry on the left and wet on the right.
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, ambient in zip(axes, ["dry", "wet"]):
        pts = res[res["ambient"] == ambient]
        if pts.empty:
            ax.set_visible(False)
            continue

        times = np.linspace(0, pts["time_hr"].max() * 1.1, 200)
        temps = sorted(pts["temp_c"].unique())
        for i, temp in enumerate(temps):
            color = f"C{i}"  # same color for a temperature's curve and its dots
            ax.plot(times, thickness(times, temp, ambient), color=color, label=f"{temp:g} °C")
            these = pts[pts["temp_c"] == temp]
            ax.scatter(these["time_hr"], these["ref_nm"], color=color, edgecolor="k", zorder=3)

        ax.set_title(f"{ambient.capitalize()} oxidation, <100> Si (dots = reference)")
        ax.set_xlabel("Time (hr)")
        ax.set_ylabel("Oxide thickness (nm)")
        ax.grid(alpha=0.3)
        ax.legend()

    fig.tight_layout()
    fig.savefig(save_to, dpi=150)
    print(f"Saved {save_to}")


if __name__ == "__main__":
    ref = load_reference()

    if ref.empty:
        print("No reference numbers filled in yet. Add them to data/byu_reference.csv")
    else:
        res = compare(ref)
        print_results(res)

        PLOTS.mkdir(exist_ok=True)
        parity_plot(res, PLOTS / "validation_parity.png")
        curves_plot(res, PLOTS / "validation_curves.png")