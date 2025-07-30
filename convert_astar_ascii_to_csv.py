import os
import pandas as pd
import re

def convert_astar_ascii_to_csv(input_folder='astar_raw', output_folder='csv_files'):
    os.makedirs(output_folder, exist_ok=True)
    for fname in os.listdir(input_folder):
        if not fname.lower().endswith('.txt') and not fname.lower().endswith('.dat'):
            continue
        path = os.path.join(input_folder, fname)
        data = []
        with open(path, 'r') as f:
            for line in f:
                parts = re.findall(r'[-+]?\d*\.\d+(?:E[-+]?\d+)?', line)
                if len(parts) >= 4:
                    try:
                        data.append({
                            'Energy_MeV': float(parts[0]),
                            'Elec_Stop': float(parts[1]),
                            'Nuc_Stop': float(parts[2]),
                            'Total_Stop': float(parts[3])
                        })
                    except:
                        continue
        if not data:
            print(f"No valid rows in {fname}")
            continue
        df = pd.DataFrame(data)
        base = os.path.splitext(fname)[0].lower()
        csvname = f"{base}.csv"
        csvpath = os.path.join(output_folder, csvname)
        df.to_csv(csvpath, index=False)
        print(f"Created ASTAR CSV: {csvpath}")

if __name__ == '__main__':
    convert_astar_ascii_to_csv()
