import os
import pandas as pd
import re

def convert_all_dat_to_csv(dat_folder='dat_files', csv_folder='csv_files'):
    os.makedirs(csv_folder, exist_ok=True)

    for filename in os.listdir(dat_folder):
        if not filename.endswith('.dat'):
            continue

        element = filename.replace('.dat', '').lower()
        input_path = os.path.join(dat_folder, filename)
        output_path = os.path.join(csv_folder, f"{element}.csv")

        rows = []
        with open(input_path, 'r') as f:
            for line in f:
                if line.strip().startswith('#'):
                    continue

                parts = re.findall(r'[-+]?\d*\.\d+|\d+E[+-]\d+', line)
                if len(parts) < 6:
                    continue

                try:
                    energy_kev = float(parts[0])
                    energy_mev = energy_kev / 1000.0
                    elec_stop = float(parts[1])
                    nuc_stop = float(parts[2])
                    total_stop = elec_stop + nuc_stop
                    range_a = float(parts[3])
                    strag_long = float(parts[4])
                    strag_lat = float(parts[5])
                except ValueError:
                    continue

                rows.append([
                    energy_mev, elec_stop, nuc_stop, total_stop,
                    range_a, strag_long, strag_lat
                ])

        if rows:
            df = pd.DataFrame(rows, columns=[
                'Energy_MeV', 'Elec_Stop', 'Nuc_Stop', 'Total_Stop',
                'Range_A', 'Straggling_Long_A', 'Straggling_Lat_A'
            ])
            df.to_csv(output_path, index=False)
            print(f"✅ Converted: {filename} → {output_path}")
        else:
            print(f"⚠️  Skipped: {filename} (no valid data)")

if __name__ == "__main__":
    convert_all_dat_to_csv()
