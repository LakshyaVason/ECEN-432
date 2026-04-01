import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
import os

# ==========================================
# 1. Load and Clean the Data
# ==========================================
script_dir = os.path.dirname(os.path.abspath(__file__))
scope_path = os.path.join(script_dir, 'q2_input_scope_analyzer_CSV.csv')
logic_path = os.path.join(script_dir, 'q2_logic analyzer_CSV.csv')

df_scope = pd.read_csv(scope_path, skiprows=27, encoding='latin1')
df_logic = pd.read_csv(logic_path, skiprows=7, encoding='latin1')

df_scope.columns = df_scope.columns.str.strip()
df_logic.columns = df_logic.columns.str.strip()

# Find all DIO columns and sort them from D6 down to D0
dio_cols = [col for col in df_logic.columns if 'DIO' in col]
dio_cols = sorted(dio_cols, key=lambda x: int(x.split()[-1]), reverse=True)

# Calculate decimal code
df_logic['Code'] = df_logic[dio_cols].sum(axis=1)

t_scope = df_scope['Time (s)'].values
v_in = df_scope['Channel 1 (V)'].values

t_logic = df_logic['Time (s)'].values

# ==========================================
# 2. Time Alignment
# ==========================================
time_shift_guess = 0.0004
time_shift = t_scope[0] - t_logic[0] + time_shift_guess
t_logic_shifted = t_logic + time_shift

# Interpolate ALL digital pins (D6-D0) and the Code onto the scope's time axis
aligned_data = {}
for col in dio_cols + ['Code']:
    interp_func = interp1d(t_logic_shifted, df_logic[col].values, kind='previous', bounds_error=False, fill_value=0)
    aligned_data[col] = interp_func(t_scope)

# ==========================================
# 3. Smart Slicing to the Rising Edge
# ==========================================
dt = t_scope[1] - t_scope[0]
aligned_code = aligned_data['Code']
rising_edges = np.where(np.diff(aligned_code) > 0)[0]

if len(rising_edges) == 0:
    print("ERROR: No digital transitions found.")
else:
    first_trans_idx = rising_edges[0]
    
    # Look backward to the trough of the sine wave
    lookback_points = int(0.00025 / dt)
    search_start = max(0, first_trans_idx - lookback_points)
    true_start_idx = search_start + np.argmin(v_in[search_start:first_trans_idx])
    
    # 0.5ms rising window
    true_end_idx = true_start_idx + int(0.0005 / dt)
    v_in_edge = v_in[true_start_idx:true_end_idx]

    # ==========================================
    # 4. Generate the Voltage Table
    # ==========================================
    target_vins = np.arange(0.1, 0.9, 0.1)  # 0.1V to 0.8V
    table_data = []

    for tv in target_vins:
        # Find the index where actual voltage is closest to the target voltage (0.1, 0.2, etc.)
        idx_in_edge = np.argmin(np.abs(v_in_edge - tv))
        global_idx = true_start_idx + idx_in_edge
        
        actual_v = v_in[global_idx]
        
        # Grab the individual thermometer bits
        bits = [int(aligned_data[col][global_idx]) for col in dio_cols]
        thermometer_str = "".join(map(str, bits))
        
        # Grab the decimal code
        dec_code = int(aligned_data['Code'][global_idx])
        
        table_data.append({
            "Target Vin (V)": f"{tv:.1f}",
            "Actual Vin (V)": f"{actual_v:.3f}",
            "Thermometer (D6...D0)": thermometer_str,
            "Decimal Code": dec_code
        })

    # ==========================================
    # 5. Print the formatted Table
    # ==========================================
    df_results = pd.DataFrame(table_data)
    print("\n=== ADC Output Table ===")
    print(df_results.to_string(index=False))