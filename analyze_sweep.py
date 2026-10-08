"""深入对比各测量条件: 等效电路拟合质量 + 幅值线性度."""
import os
import sys
import glob
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analysis.fitting import fit_equivalent_circuit

print("=" * 80)
print("各条件等效电路拟合对比 (全频段 Rs(RpC) 与 Rs(RpCPE))")
print("=" * 80)

rows = []
for fname in sorted(glob.glob('data/sweep_*.csv')):
    df = pd.read_csv(fname)
    label = os.path.basename(fname).replace('sweep_', '').replace('.csv', '')
    freq = df['freq [Hz]'].values
    z = df['Z_real [ohm]'].values + 1j * df['Z_imag [ohm]'].values

    try:
        r1 = fit_equivalent_circuit(freq, z, use_cpe=False, freq_min=0)
        r2 = fit_equivalent_circuit(freq, z, use_cpe=True, freq_min=0)
        rows.append({
            'label': label,
            'err_RC': r1['norm_error'],
            'err_CPE': r2['norm_error'],
            'Rs_RC': r1['Rs'],
            'Rp_RC': r1['Rp'],
            'C_RC': r1['C'],
            'tau_RC': r1['tau'],
            'Q_CPE': r2['Q'],
            'n_CPE': r2['n'],
        })
    except Exception as e:
        print(f"{label}: 拟合失败 {e}")

tbl = pd.DataFrame(rows).sort_values('err_RC')
pd.set_option('display.width', 120)
pd.set_option('display.float_format', lambda x: f'{x:.4g}')
print(tbl.to_string(index=False))

print("\n" + "=" * 80)
print("幅值线性度检查: C(10kHz) 随幅值变化 (理想器件应无变化)")
print("=" * 80)
c_at_10k = {}
for fname in sorted(glob.glob('data/sweep_*mV_avg3.csv')):
    df = pd.read_csv(fname)
    label = os.path.basename(fname).replace('sweep_', '').replace('.csv', '')
    idx = np.argmin(np.abs(df['freq [Hz]'].values - 1e4))
    f = df['freq [Hz]'].values[idx]
    z = df['Z_real [ohm]'].values[idx] + 1j * df['Z_imag [ohm]'].values[idx]
    c = -1.0 / (2 * np.pi * f * z.imag)
    c_at_10k[label] = c

for k, v in sorted(c_at_10k.items()):
    print(f"  {k}: C(10kHz) = {v:.4e} F")
