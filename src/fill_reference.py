
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
DEFAULT_CSV = REPO / "data" / "byu_reference.csv"


def as_hms(hours):
    # BYU wants time as hrs:mins:secs, so 0.5 -> 0:30:00
    total_sec = round(hours * 3600)
    h, rest = divmod(total_sec, 3600)
    m, s = divmod(rest, 60)
    return f"{h}:{m:02d}:{s:02d}"


def main():
    csv_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_CSV
    df = pd.read_csv(csv_path)

    blank = df.index[df["thickness_A"].isna()].tolist()
    if not blank:
        print(f"Every row in {csv_path.name} is already filled in.")
        return

    print(f"{len(blank)} rows to fill in {csv_path.name}")
    print("For each one, enter these in the BYU calculator with initial thickness = 0.\n")

    for count, i in enumerate(blank, start=1):
        row = df.loc[i]
        print(f"[{count}/{len(blank)}]  {row['ambient'].capitalize()}, "
              f"{row['temp_c']:g} °C, <{row['orientation']}>, time {as_hms(row['time_hr'])}")

        while True:
            answer = input("    BYU thickness (Å): ").strip()
            if answer == "" or answer.lower() == "q":
                break
            try:
                value = float(answer)
            except ValueError:
                print("    That's not a number, try again.")
                continue

            if value <= 0:
                print("    Thickness has to be positive, try again.")
                continue
            df.loc[i, "thickness_A"] = value
            df.to_csv(csv_path, index=False)
            break

        if answer.lower() == "q":
            break

    left = int(df["thickness_A"].isna().sum())
    print(f"\nSaved. {left} rows still blank.")
    if left == 0:
        print("All done. Run: python src\\validate.py")


if __name__ == "__main__":
    main()
