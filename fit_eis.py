"""对 a-Si 电池 EIS 数据做等效电路拟合并绘图 (全频段)."""
import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analysis.fitting import fit_equivalent_circuit

df = pd.read_csv('data/eis_ch0.csv')
freq = df['freq [Hz]'].values
z = df['Z_real [ohm]'].values + 1j * df['Z_imag [ohm]'].values

print('=== 全频段等效电路拟合 ===')
for use_cpe, label in [(False, 'Rs(RpC)'), (True, 'Rs(RpCPE)')]:
    try:
        r = fit_equivalent_circuit(freq, z, use_cpe=use_cpe, freq_min=0)
        print(f'\n【{label}】')
        for name, val, err in zip(r['names'], r['params'], r['params_err']):
            print(f'  {name:>18s} = {val:.6g} ± {err:.3g}')
        print(f'  归一化误差: {r["norm_error"]:.2%}')
        if 'tau' in r:
            print(f'  τ = {r["tau"]:.3g} s, f_apex = {r["f_apex"]:.1f} Hz')
    except Exception as e:
        print(f'【{label}】拟合失败: {e}')

# 绘图: 数据 vs 拟合 (全频段 Nyquist + Bode)
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))

    # Nyquist
    ax = axes[0]
    ax.plot(z.real, -z.imag, 'o', ms=4, label='data')
    for use_cpe, label, color in [(False, 'fit Rs(RpC)', 'tab:red'),
                                  (True, 'fit Rs(RpCPE)', 'tab:green')]:
        try:
            r = fit_equivalent_circuit(freq, z, use_cpe=use_cpe, freq_min=0)
            ax.plot(r['z_fit'].real, -r['z_fit'].imag, '-', lw=2, color=color, label=label)
        except Exception:
            pass
    ax.set_xlabel("Z' [Ω]")
    ax.set_ylabel("-Z'' [Ω]")
    ax.set_title("Nyquist (full range)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Bode |Z|
    ax = axes[1]
    ax.loglog(freq, np.abs(z), 'o', ms=3, label='data')
    for use_cpe, label, color in [(False, 'fit Rs(RpC)', 'tab:red'),
                                  (True, 'fit Rs(RpCPE)', 'tab:green')]:
        try:
            r = fit_equivalent_circuit(freq, z, use_cpe=use_cpe, freq_min=0)
            ax.loglog(r['freq_fit'], np.abs(r['z_fit']), '-', lw=2, color=color, label=label)
        except Exception:
            pass
    ax.set_xlabel("f [Hz]")
    ax.set_ylabel("|Z| [Ω]")
    ax.set_title("Bode |Z|")
    ax.legend()
    ax.grid(True, which='both', alpha=0.3)

    fig.tight_layout()
    fig.savefig('data/eis_fit.png', dpi=150)
    print('\n拟合图已保存: data/eis_fit.png')
except ImportError:
    print('\nmatplotlib 未安装, 跳过绘图')
