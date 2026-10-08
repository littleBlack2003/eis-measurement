"""从已有 CSV 重新生成 EIS 图 (无需重新测量)."""
import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

out_dir = 'data'
results = []
for fname in sorted(os.listdir(out_dir)):
    if not fname.startswith('eis_ch') or not fname.endswith('.csv'):
        continue
    df = pd.read_csv(os.path.join(out_dir, fname))
    ch = int(fname.replace('eis_ch', '').replace('.csv', ''))
    z = df['Z_real [ohm]'].values + 1j * df['Z_imag [ohm]'].values
    results.append({
        'channel': ch,
        'freq': df['freq [Hz]'].values,
        'z': z,
        'z_raw': z,
        'z_mag': np.abs(z),
        'theta_deg': df['theta [deg]'].values,
    })

from utils.data_io import plot_eis_all
plot_eis_all(results, out_dir)
print("Plots regenerated:", sorted(f for f in os.listdir(out_dir) if f.endswith('.png')))
