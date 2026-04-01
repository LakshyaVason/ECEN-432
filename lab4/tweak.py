import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

# ==========================================
# 1. Load and Clean the Data
# ==========================================
# Note: Adjust skiprows if your exact files have slightly different header lengths
import os
script_dir = os.path.dirname(os.path.abspath(__file__))
scope_path = os.path.join(script_dir, 'q2_input_scope_analyzer_CSV.csv')
logic_path = os.path.join(script_dir, 'q2_logic analyzer_CSV.csv')

df_scope = pd.read_csv(scope_path, skiprows=27, encoding='latin1')
df_logic = pd.read_csv(logic_path, skiprows=7, encoding='latin1')

# Clean column names (remove leading/trailing spaces)
df_scope.columns = df_scope.columns.str.strip()
df_logic.columns = df_logic.columns.str.strip()

# Convert the 7 DIO thermometer columns into a single decimal code (0 to 7)
# Assuming DIO 0 to DIO 6 are binary 0 or 1
dio_cols = [col for col in df_logic.columns if 'DIO' in col]
df_logic['Code'] = df_logic[dio_cols].sum(axis=1)

# Extract time and values
t_scope = df_scope['Time (s)'].values
v_in = df_scope['Channel 1 (V)'].values

t_logic = df_logic['Time (s)'].values
code_out = df_logic['Code'].values
# ==========================================
# 🛑 TWEAK THIS NUMBER 🛑
# ==========================================
time_shift_guess = 0.0004  
# ==========================================

time_shift = t_scope[0] - t_logic[0] + time_shift_guess
t_logic_shifted = t_logic + time_shift

interp_func = interp1d(t_logic_shifted, code_out, kind='previous', bounds_error=False, fill_value=0)
aligned_code = interp_func(t_scope)

# Plotting against Time
plt.figure(figsize=(10, 4))
plt.plot(t_scope, v_in, label='Analog Sine Wave')
plt.plot(t_scope, aligned_code * 0.1, label='Digital Code (Scaled)') # Scaled to fit on the same graph
plt.xlim(t_scope[0], t_scope[0] + 0.002) # Look at just 2 periods (2ms)
plt.title(f"Calibration Plot (Shift Guess: {time_shift_guess})")
plt.xlabel("Time (s)")
plt.ylabel("Voltage (V)")
plt.legend()
plt.grid(True)
plt.show()