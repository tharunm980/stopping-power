import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests
from io import StringIO
import ssl
import os
import re
from striprtf.striprtf import rtf_to_text

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

def convert_rtf_to_plaintext(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8') as file:
            rtf_content = file.read()
        plain_text = rtf_to_text(rtf_content)
        return plain_text
    except Exception as e:
        print(f"Error converting RTF file {filepath} to plain text: {e}")
        return None

def fetch_srim_data(ion='He', target='Ar'):
    file_name = f"SRIM_{ion}_in_{target}.rtf"
    print(f"Attempting to read SRIM data from local RTF file: {file_name}")

    if os.path.exists(file_name):
        plain_text_data = convert_rtf_to_plaintext(file_name)
        if plain_text_data:
            print("\n--- SRIM Plain Text Data (Inspect this for issues) ---")
            print(plain_text_data[:1000])
            print("----------------------------------------------------\n")

            try:
                lines = plain_text_data.strip().splitlines()
                data_rows = []
                
                data_start_index = -1
                for i, line in enumerate(lines):
                    stripped_line = line.strip()
                    if re.match(r'^#?\s*[-]+\s*[-]+', stripped_line):
                        data_start_index = i + 1
                        break
                
                if data_start_index == -1:
                    raise ValueError("Could not find SRIM data start marker (dashes) in plain text. Please check the format of SRIM_He_in_Ar.rtf's plaintext.")

                for line in lines[data_start_index:]:
                    stripped_line = line.strip()
                    if stripped_line and not stripped_line.startswith('#'):
                        processed_line = re.sub(r'(\d(?:\.\d+)?)\s+(keV|MeV|A|um)\b', r'\1\2', stripped_line)
                        
                        parts = [p.strip() for p in re.split(r'\s+', processed_line) if p.strip()]

                        if len(parts) == 6:
                            data_rows.append(parts)
                        elif len(parts) > 0:
                            print(f"Skipping malformed SRIM data row (wrong column count after pre-processing): '{stripped_line}' -> '{processed_line}' ({len(parts)} parts)")

                if not data_rows:
                    raise ValueError("No valid SRIM data rows found after parsing. The file might be empty or formatted unexpectedly.")

                srim_raw_col_names = ['Energy_Raw', 'Elec_Stop_Raw', 'Nuc_Stop_Raw', 'Range_Raw', 'Long_Straggling_Raw', 'Lat_Straggling_Raw']
                df = pd.DataFrame(data_rows, columns=srim_raw_col_names)

                df['Energy_MeV'] = df['Energy_Raw'].apply(lambda x: float(x.replace('keV', '').replace('MeV', '')) / (1000.0 if 'keV' in x else 1.0))

                df['Elec_Stop'] = pd.to_numeric(df['Elec_Stop_Raw'], errors='coerce')
                df['Nuc_Stop'] = pd.to_numeric(df['Nuc_Stop_Raw'], errors='coerce')
                df['Elec_Stop_g'] = df['Elec_Stop'] * 1000.0
                df['Nuc_Stop_g'] = df['Nuc_Stop'] * 1000.0
                df['Total_Stop_g'] = df['Elec_Stop_g'] + df['Nuc_Stop_g']

                def parse_length_col(val):
                    try:
                        val_str = str(val).strip()
                        if 'A' in val_str:
                            return float(val_str.replace('A', '')) * 1e-8
                        elif 'um' in val_str:
                            return float(val_str.replace('um', '')) * 1e-4
                        else:
                            return float(val_str)
                    except ValueError:
                        return np.nan

                df['Range_cm'] = df['Range_Raw'].apply(parse_length_col)
                df['Long_Straggling_cm'] = df['Long_Straggling_Raw'].apply(parse_length_col)
                df['Lat_Straggling_cm'] = df['Lat_Straggling_Raw'].apply(parse_length_col)

                if df[['Energy_MeV', 'Total_Stop_g', 'Elec_Stop_g', 'Nuc_Stop_g']].isnull().any().any():
                    print(f"Warning: Critical NaN values found after parsing SRIM data from {file_name}. Review data quality.")
                    df.dropna(subset=['Energy_MeV', 'Total_Stop_g', 'Elec_Stop_g', 'Nuc_Stop_g'], inplace=True)
                    if df.empty:
                        print("SRIM dataframe became empty after dropping NaNs. Returning None.")
                        return None
                
                print(f"SRIM data loaded and processed successfully from local RTF file: {file_name}")
                return df[['Energy_MeV', 'Elec_Stop_g', 'Nuc_Stop_g', 'Total_Stop_g', 'Range_cm', 'Long_Straggling_cm', 'Lat_Straggling_cm']]

            except Exception as e:
                print(f"Error parsing plain text from SRIM RTF file {file_name}: {e}")
                return None
        else:
            print(f"RTF conversion to plain text failed for {file_name}.")
            return None
    else:
        print(f"Local SRIM RTF file '{file_name}' not found. Attempting to fetch from URL (likely to fail due to 404).")
        url = f"https://raw.githubusercontent.com/swesterd/NeucBot/master/neucbot/data/stopping_power/SRIM_{ion}_in_{target}.dat"
        print(f"Fetching SRIM data from: {url}")
        try:
            col_names = ['Energy_MeV', 'Elec_Stop', 'Nuc_Stop', 'Total_Stop', 'Range', 'Straggling']
            df = pd.read_csv(url, comment='#', sep='\s+', names=col_names)
            for col in ['Energy_MeV', 'Elec_Stop', 'Nuc_Stop', 'Total_Stop', 'Range', 'Straggling']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            df['Total_Stop_g'] = df['Total_Stop'] * 1000.0
            print("SRIM data fetched and processed successfully from URL.")
            return df
        except requests.exceptions.RequestException as e:
            print(f"Network or request error fetching SRIM data from URL: {e}")
            return None
        except Exception as e:
            print(f"Error parsing SRIM data from URL: {e}")
            return None


def fetch_astar_data(ion='He', target='Ar'):
    file_name = f"ASTAR_{ion}_in_{target}.rtf"
    print(f"Attempting to read ASTAR data from local RTF file: {file_name}")

    if os.path.exists(file_name):
        plain_text_data = convert_rtf_to_plaintext(file_name)
        if plain_text_data:
            print("\n--- ASTAR Plain Text Data (Inspect this for issues) ---")
            print(plain_text_data[:1000])
            print("----------------------------------------------------\n")

            try:
                col_names = ['Energy_MeV', 'Elec_Stop', 'Nuc_Stop', 'Total_Stop', 'CSDA_Range', 'Proj_Range', 'Detour']
                
                lines = plain_text_data.strip().splitlines()
                data_start_row = 0
                for i, line in enumerate(lines):
                    if line.strip().startswith('T ') and 'STOP(e)' in line:
                        data_start_row = i + 1
                        break
                
                if data_start_row == 0:
                    raise ValueError("Could not find ASTAR data header in plain text. Please check the format of ASTAR_He_in_Ar.rtf's plaintext.")

                clean_data_lines = []
                for line_idx in range(data_start_row, len(lines)):
                    line = lines[line_idx].strip()
                    if line and not line.startswith('#'):
                        parts = [p.strip() for p in re.split(r'\s+', line) if p.strip()]
                        if len(parts) == len(col_names):
                            clean_data_lines.append(" ".join(parts))
                        elif len(parts) > 0:
                            print(f"Skipping malformed ASTAR data row (wrong column count): '{line}'")

                if not clean_data_lines:
                    raise ValueError("No valid ASTAR data rows found after parsing. The file might be empty or formatted unexpectedly.")

                df = pd.read_csv(StringIO("\n".join(clean_data_lines)), 
                                 sep='\s+', 
                                 names=col_names,
                                 engine='python'
                                 )

                for col in col_names:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                
                if df[['Energy_MeV', 'Elec_Stop', 'Nuc_Stop', 'Total_Stop']].isnull().any().any():
                    print(f"Warning: Critical NaN values found after parsing ASTAR data from {file_name}. Review data quality.")
                    df.dropna(subset=['Energy_MeV', 'Elec_Stop', 'Nuc_Stop', 'Total_Stop'], inplace=True)
                    if df.empty:
                        print("ASTAR dataframe became empty after dropping NaNs. Returning None.")
                        return None

                print(f"ASTAR data loaded from local RTF file: {file_name}")
                return df

            except Exception as e:
                print(f"Error parsing plain text from ASTAR RTF file {file_name}: {e}")
                return None
        else:
            print(f"RTF conversion to plain text failed for {file_name}.")
            return None
    else:
        print(f"Local ASTAR RTF file '{file_name}' not found. Attempting to fetch from URL (known to give 404).")
        url = f"https://physics.nist.gov/PhysRefData/Star/Text/ASTAR.php?ion={ion}&mat={target}"
        print(f"Fetching ASTAR data from: {url}")
        try:
            response = requests.get(url)
            response.raise_for_status()

            data_string = response.text
            col_names = ['Energy_MeV', 'Elec_Stop', 'Nuc_Stop', 'Total_Stop', 'CSDA_Range', 'Proj_Range', 'Detour']
            df = pd.read_csv(StringIO(data_string), skiprows=8, sep='\s+', names=col_names)
            for col in col_names:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            print("ASTAR data fetched and processed successfully from URL.")
            return df
        except requests.exceptions.HTTPError as http_err:
            print(f"HTTP error fetching ASTAR data from URL: {http_err} - URL might be incorrect or resource moved.")
            return None
        except requests.exceptions.ConnectionError as conn_err:
            print(f"Connection error fetching ASTAR data from URL: {conn_err} - Check your internet connection or URL.")
            return None
        except Exception as e:
            print(f"Error parsing ASTAR data from URL: {e}")
            return None

def plot_comparison(srim_df, astar_df, ion, target):
    if srim_df is None and astar_df is None:
        print("Both data sources could not be loaded. Aborting plot.")
        return

    plt.style.use('seaborn-v0_8-whitegrid')

    # Plot 1: Total Stopping Power
    fig1, ax1 = plt.subplots(figsize=(10, 7))
    if srim_df is not None:
        if not srim_df['Total_Stop_g'].isnull().all() and not srim_df['Energy_MeV'].isnull().all():
            ax1.plot(srim_df['Energy_MeV'], srim_df['Total_Stop_g'],
                     label='SRIM (from SRIM RTF)', color='blue', linestyle='-', marker='o', markersize=3)
        else:
            print("SRIM 'Total_Stop_g' or 'Energy_MeV' column contains only NaNs after processing, cannot plot on total stopping power graph.")
    else:
        print("SRIM data not available for plotting total stopping power.")

    if astar_df is not None:
        if not astar_df['Total_Stop'].isnull().all() and not astar_df['Energy_MeV'].isnull().all():
            ax1.plot(astar_df['Energy_MeV'], astar_df['Total_Stop'],
                     label='ASTAR (from NIST RTF)', color='red', linestyle='--', marker=None)
        else:
            print("ASTAR 'Total_Stop' or 'Energy_MeV' column contains only NaNs after processing, cannot plot on total stopping power graph.")
    else:
        print("ASTAR data not available for plotting total stopping power.")
    
    if (srim_df is not None and not srim_df['Total_Stop_g'].isnull().all() and not srim_df['Energy_MeV'].isnull().all()) or \
       (astar_df is not None and not astar_df['Total_Stop'].isnull().all() and not astar_df['Energy_MeV'].isnull().all()):
        ax1.set_xscale('log')
        ax1.set_yscale('log')
        ax1.set_xlabel('Ion Energy ($MeV$)', fontsize=12)
        ax1.set_ylabel('Total Stopping Power ($MeV \\cdot cm^2 / g$)', fontsize=12)
        ax1.set_title(f'Total Stopping Power for {ion} Ions in {target}', fontsize=14, fontweight='bold')
        ax1.grid(True, which='both', linestyle=':')
        ax1.legend()
    else:
        print("No valid data for total stopping power plot.")

    # Plot 2: Electronic Stopping Power
    fig2, ax2 = plt.subplots(figsize=(10, 7))
    if srim_df is not None:
        if not srim_df['Elec_Stop_g'].isnull().all() and not srim_df['Energy_MeV'].isnull().all():
            ax2.plot(srim_df['Energy_MeV'], srim_df['Elec_Stop_g'],
                     label='SRIM (from SRIM RTF)', color='blue', linestyle='-', marker='o', markersize=3)
        else:
            print("SRIM 'Elec_Stop_g' or 'Energy_MeV' column contains only NaNs after processing, cannot plot on electronic stopping power graph.")
    else:
        print("SRIM data not available for plotting electronic stopping power.")

    if astar_df is not None:
        if not astar_df['Elec_Stop'].isnull().all() and not astar_df['Energy_MeV'].isnull().all():
            ax2.plot(astar_df['Energy_MeV'], astar_df['Elec_Stop'],
                     label='ASTAR (from NIST RTF)', color='red', linestyle='--', marker=None)
        else:
            print("ASTAR 'Elec_Stop' or 'Energy_MeV' column contains only NaNs after processing, cannot plot on electronic stopping power graph.")
    else:
        print("ASTAR data not available for plotting electronic stopping power.")

    if (srim_df is not None and not srim_df['Elec_Stop_g'].isnull().all() and not srim_df['Energy_MeV'].isnull().all()) or \
       (astar_df is not None and not astar_df['Elec_Stop'].isnull().all() and not astar_df['Energy_MeV'].isnull().all()):
        ax2.set_xscale('log')
        ax2.set_yscale('log')
        ax2.set_xlabel('Ion Energy ($MeV$)', fontsize=12)
        ax2.set_ylabel('Electronic Stopping Power ($MeV \\cdot cm^2 / g$)', fontsize=12)
        ax2.set_title(f'Electronic Stopping Power for {ion} Ions in {target}', fontsize=14, fontweight='bold')
        ax2.grid(True, which='both', linestyle=':')
        ax2.legend()
    else:
        print("No valid data for electronic stopping power plot.")

    # Plot 3: Nuclear Stopping Power
    fig3, ax3 = plt.subplots(figsize=(10, 7))
    if srim_df is not None:
        if not srim_df['Nuc_Stop_g'].isnull().all() and not srim_df['Energy_MeV'].isnull().all():
            ax3.plot(srim_df['Energy_MeV'], srim_df['Nuc_Stop_g'],
                     label='SRIM (from SRIM RTF)', color='blue', linestyle='-', marker='o', markersize=3)
        else:
            print("SRIM 'Nuc_Stop_g' or 'Energy_MeV' column contains only NaNs after processing, cannot plot on nuclear stopping power graph.")
    else:
        print("SRIM data not available for plotting nuclear stopping power.")

    if astar_df is not None:
        if not astar_df['Nuc_Stop'].isnull().all() and not astar_df['Energy_MeV'].isnull().all():
            ax3.plot(astar_df['Energy_MeV'], astar_df['Nuc_Stop'],
                     label='ASTAR (from NIST RTF)', color='red', linestyle='--', marker=None)
        else:
            print("ASTAR 'Nuc_Stop' or 'Energy_MeV' column contains only NaNs after processing, cannot plot on nuclear stopping power graph.")
    else:
        print("ASTAR data not available for plotting nuclear stopping power.")

    if (srim_df is not None and not srim_df['Nuc_Stop_g'].isnull().all() and not srim_df['Energy_MeV'].isnull().all()) or \
       (astar_df is not None and not astar_df['Nuc_Stop'].isnull().all() and not astar_df['Energy_MeV'].isnull().all()):
        ax3.set_xscale('log')
        ax3.set_yscale('log')
        ax3.set_xlabel('Ion Energy ($MeV$)', fontsize=12)
        ax3.set_ylabel('Nuclear Stopping Power ($MeV \\cdot cm^2 / g$)', fontsize=12)
        ax3.set_title(f'Nuclear Stopping Power for {ion} Ions in {target}', fontsize=14, fontweight='bold')
        ax3.grid(True, which='both', linestyle=':')
        ax3.legend()
    else:
        print("No valid data for nuclear stopping power plot.")

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    ION_SPECIES = 'He'
    TARGET_MATERIAL = 'Ar'

    srim_data = fetch_srim_data(ion=ION_SPECIES, target=TARGET_MATERIAL)
    astar_data = fetch_astar_data(ion=ION_SPECIES, target=TARGET_MATERIAL)

    plot_comparison(srim_data, astar_data, ion=ION_SPECIES, target=TARGET_MATERIAL)