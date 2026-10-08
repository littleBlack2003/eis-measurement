"""对可靠区段 (f>=10kHz) 的 EIS 数据做等效电路拟合."""
import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analysis.fitting import fit_equivalent_circuit

df = pd.read_csv('data/eis_ch0.csv')
freq = df['freq [Hz]'].values
z = df['Z_real [ohm]'].values + 1j * df['Z_imag [ohm]'].values

# 仅最可靠区段 f>=10kHz, 且实部>0
mask = (freq >= 1e4) & (z.real > 0)
print(f'可用点: {mask.sum()} / {len(freq)}')
f2, z2 = freq[mask], z[mask]

for use_cpe, label in [(False, 'Rs(RpC)'), (True, 'Rs(RpCPE)')]:
    try:
        r = fit_equivalent_circuit(f2, z2, use_cpe=use_cpe, freq_min=0)
        print(f'\n【{label} 模型, f>=10kHz】')
        for name, val, err in zip(r['names'], r['params'], r['params_err']):
            print(f'  {name:>20s} = {val:.6g} ± {err:.3g}')
        print(f'  归一化拟合误差: {r["norm_error"]:.2%}')
        if 'tau' in r:
            print(f'  τ = {r["tau"]:.3g} s, f_apex = {r["f_apex"]:.1f} Hz')
    except Exception as e:
        print(f'【{label}】拟合失败: {e}')
