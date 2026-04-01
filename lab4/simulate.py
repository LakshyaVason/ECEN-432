import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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

dio_cols = [col for col in df_logic.columns if 'DIO' in col]
df_logic['Code'] = df_logic[dio_cols].sum(axis=1)

t_scope = df_scope['Time (s)'].values
v_in = df_scope['Channel 1 (V)'].values

t_logic = df_logic['Time (s)'].values
code_out = df_logic['Code'].values

# ==========================================
# 2. Time Alignment
# ==========================================
time_shift_guess = 0.0004  # Your perfectly calibrated shift!
time_shift = t_scope[0] - t_logic[0] + time_shift_guess
t_logic_shifted = t_logic + time_shift

interp_func = interp1d(t_logic_shifted, code_out, kind='previous', bounds_error=False, fill_value=0)
aligned_code = interp_func(t_scope)

# ==========================================
# 3. SMART SLICING (The Fix for NaN)
# ==========================================
dt = t_scope[1] - t_scope[0]

# Find the first index where the digital code actually transitions upwards
rising_edges = np.where(np.diff(aligned_code) > 0)[0]

if len(rising_edges) == 0:
    print("ERROR: No digital transitions found. Your time_shift_guess might push the data out of bounds.")
else:
    first_trans_idx = rising_edges[0]
    
    # Look backward 1/4 of a period (0.25ms) to find the trough of the sine wave
    lookback_points = int(0.00025 / dt)
    search_start = max(0, first_trans_idx - lookback_points)
    
    # The true start is the local minimum immediately preceding the digital transition
    true_start_idx = search_start + np.argmin(v_in[search_start:first_trans_idx])
    
    # Define a 0.5ms window (half period of a 1kHz wave) going up the rising slope
    true_end_idx = true_start_idx + int(0.0005 / dt)

    v_in_edge = v_in[true_start_idx:true_end_idx]
    code_edge = aligned_code[true_start_idx:true_end_idx]

    # ==========================================
    # 4. Calculate Transition Voltages (V_T)
    # ==========================================
    transitions = []
    for k in range(1, 8):
        voltages_at_k = v_in_edge[code_edge == k]
        if len(voltages_at_k) > 0:
            transitions.append(np.min(voltages_at_k))
        else:
            transitions.append(np.nan)

    V_T = np.array(transitions)

    # ==========================================
    # 5. Calculate ADC Metrics
    # ==========================================
    LSB_ideal = 0.100  # 100 mV LSB
    V_T_ideal = np.array([1, 2, 3, 4, 5, 6, 7]) * LSB_ideal

    offset_error_V = V_T[0] - V_T_ideal[0]
    offset_error_LSB = offset_error_V / LSB_ideal

    fs_error_V = V_T[-1] - V_T_ideal[-1]
    fs_error_LSB = fs_error_V / LSB_ideal

    LSB_actual = (V_T[-1] - V_T[0]) / 6
    gain_error = (LSB_actual / LSB_ideal) - 1

    step_widths = np.diff(V_T)
    dnl = (step_widths / LSB_actual) - 1
    dnl = np.insert(dnl, 0, 0)

    inl = (V_T - V_T[0]) / LSB_actual - np.arange(7)

    print("=== 3-Bit Flash ADC Metrics ===")
    print(f"Ideal LSB: {LSB_ideal*1000:.1f} mV")
    print(f"Actual LSB: {LSB_actual*1000:.1f} mV")
    print(f"Offset Error: {offset_error_LSB:.2f} LSB ({offset_error_V*1000:.1f} mV)")
    print(f"Full-Scale Error: {fs_error_LSB:.2f} LSB ({fs_error_V*1000:.1f} mV)")
    print(f"Linear Gain Error: {gain_error*100:.2f} %")
    print("\nDNL (LSB):", np.round(dnl, 2))
    print("INL (LSB):", np.round(inl, 2))

    # ==========================================
    # 6. Plot the Transfer Function
    # ==========================================
    plt.figure(figsize=(10, 6))

    plt.step(v_in_edge, code_edge, where='post', label='Actual ADC Output', color='blue', linewidth=2)

    ideal_vin = np.linspace(0, 0.8, 1000)
    ideal_code = np.floor(ideal_vin / LSB_ideal)
    ideal_code = np.clip(ideal_code, 0, 7)
    plt.step(ideal_vin, ideal_code, where='post', label='Ideal ADC', color='red', linestyle='dashed', alpha=0.7)

    plt.title('3-Bit Flash ADC Transfer Function')
    plt.xlabel('Analog Input Voltage (V)')
    plt.ylabel('Digital Output Code (Decimal)')
    plt.yticks(np.arange(0, 8))
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()