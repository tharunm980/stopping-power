import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

def load_csv_data(ion, target, source):
    filename = f"{ion}_in_{target}_{source}.csv"
    path = os.path.join("csv_files", filename)
    if not os.path.exists(path):
        print(f"Missing {source.upper()} CSV file: {path}")
        return None
    df = pd.read_csv(path)
    if df.empty:
        print(f"{source.upper()} data for {ion} in {target} is empty.")
        return None
    return df

def prepare_data(df):
    if "Energy_keV" in df.columns:
        df = df[df["Energy_keV"].notnull()]
        df["Energy_MeV"] = df["Energy_keV"] / 1000.0
    elif "Energy_MeV" in df.columns:
        df = df[df["Energy_MeV"].notnull()]
    else:
        raise KeyError("Data missing 'Energy_keV' and 'Energy_MeV' columns.")
    df = df.sort_values("Energy_MeV").reset_index(drop=True)
    return df

def plot_data(srim, astar, ion, target):
    srim_interp = astar.copy()

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharex=True, sharey=True)

    axes[0].plot(srim_interp["Energy_MeV"], srim_interp["Elec_Stop"],
                 label="SRIM", color="red", linestyle="-", marker="o", markersize=2,
                 markerfacecolor='none', markeredgewidth=0.8, linewidth=0.7)
    axes[0].plot(astar["Energy_MeV"], astar["Elec_Stop"],
                 label="ASTAR", color="blue", linestyle=":", linewidth=4)
    axes[0].set_title("Electronic Stopping Power")
    axes[0].set_ylabel("Stopping (MeV cm²/g)")
    axes[0].set_xscale('log')
    axes[0].set_yscale('log')
    axes[0].legend()
    axes[0].grid(True, which="both", linestyle=":")

    axes[1].plot(srim_interp["Energy_MeV"], srim_interp["Nuc_Stop"],
                 label="SRIM", color="red", linestyle="-", marker="o", markersize=2,
                 markerfacecolor='none', markeredgewidth=0.8, linewidth=0.7)
    axes[1].plot(astar["Energy_MeV"], astar["Nuc_Stop"],
                 label="ASTAR", color="blue", linestyle=":", linewidth=4)
    axes[1].set_title("Nuclear Stopping Power")
    axes[1].set_xscale('log')
    axes[1].set_yscale('log')
    axes[1].legend()
    axes[1].grid(True, which="both", linestyle=":")

    axes[2].plot(srim_interp["Energy_MeV"], srim_interp["Total_Stop"],
                 label="SRIM", color="red", linestyle="-", marker="o", markersize=2,
                 markerfacecolor='none', markeredgewidth=0.8, linewidth=0.7)
    axes[2].plot(astar["Energy_MeV"], astar["Total_Stop"],
                 label="ASTAR", color="blue", linestyle=":", linewidth=4)
    axes[2].set_title("Total Stopping Power")
    axes[2].set_xlabel("Energy (MeV)")
    axes[2].set_xscale('log')
    axes[2].set_yscale('log')
    axes[2].legend()
    axes[2].grid(True, which="both", linestyle=":")

    fig.suptitle(f"Stopping Power Comparison: {ion.upper()} in {target.upper()}", fontsize=18)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

def main():
    ion = input("Enter ion (e.g. He): ").strip().lower()
    target = input("Enter target element (e.g. Ar): ").strip().lower()

    srim = load_csv_data(ion, target, "srim")
    astar = load_csv_data(ion, target, "astar")

    if srim is None or astar is None:
        print(f"Missing data for {ion} in {target}.")
        return

    try:
        srim = prepare_data(srim)
    except KeyError as e:
        print(f"Error in SRIM data: {e}")
        return

    try:
        astar = prepare_data(astar)
    except KeyError as e:
        print(f"Error in ASTAR data: {e}")
        return

    if srim.empty or astar.empty:
        print(f"One or both datasets are empty after processing.")
        return

    plot_data(srim, astar, ion, target)

if __name__ == "__main__":
    main()
