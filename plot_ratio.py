import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def prepare_srim_data(df):
    df = df.copy()
    df = df.rename(columns={df.columns[0]: "Energy_MeV"})
    df["Energy_MeV"] = pd.to_numeric(df["Energy_MeV"], errors="coerce")
    df = df.dropna(subset=["Energy_MeV"])
    return df

def prepare_astar_data(df):
    df = df.copy()
    df = df.rename(columns={df.columns[0]: "Energy_MeV"})
    df["Energy_MeV"] = pd.to_numeric(df["Energy_MeV"], errors="coerce")
    df = df.dropna(subset=["Energy_MeV"])
    return df

def plot_ratio(srim, astar, ion, target):
    if astar is None:
        print("No ASTAR data available to compute ratio.")
        return

    astar_interp = np.interp(
        srim["Energy_MeV"],
        astar["Energy_MeV"],
        astar["Elec_Stop"]
    )
    ratio = astar_interp / srim["Total_Stop"]

    # Plot only the ratio
    plt.figure(figsize=(8,6))
    plt.plot(srim["Energy_MeV"], ratio, color="green", linewidth=1.5)
    plt.title(f"{ion} in {target} - ASTAR / SRIM Ratio")
    plt.xlabel("Energy (MeV)")
    plt.ylabel("Ratio")
    plt.grid(True)
    plt.show()

def main():
    ion = input("Enter ion (e.g., he): ").strip().capitalize()
    target = input("Enter target (e.g., ar, al, w, bi, cd): ").strip().capitalize()

    srim_file = os.path.join("csv_files", f"{ion.lower()}_in_{target.lower()}_srim.csv")
    astar_file = os.path.join("csv_files", f"{ion.lower()}_in_{target.lower()}_astar.csv")

    if not os.path.exists(srim_file):
        print(f"Missing SRIM file: {srim_file}")
        return

    srim = pd.read_csv(srim_file)
    srim = prepare_srim_data(srim)

    astar = None
    if os.path.exists(astar_file):
        astar = pd.read_csv(astar_file)
        astar = prepare_astar_data(astar)

    plot_ratio(srim, astar, ion, target)

if __name__ == "__main__":
    main()