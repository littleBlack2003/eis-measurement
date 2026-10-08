"""处理 2026-10-06 晚间测量数据: 单次扫描(+0.6V) + 偏压序列(-1.0V, -0.8V).

拟合 Rs(RpC) 与 Rs(RpCPE) 等效电路, 输出对比图与参数表.
"""
import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analysis.fitting import fit_equivalent_circuit

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

DATA_DIR = os.path.join('dist', 'EISScanGUI-20260923-104343', 'data')
SEQ_DIR = os.path.join(DATA_DIR, 'bias_sequence_20261006_220835')
OUT_DIR = os.path.join('data', '20261006_结果')
os.makedirs(OUT_DIR, exist_ok=True)

DATASETS = [
    ('+0.6 V (单次扫描)', os.path.join(DATA_DIR, 'eis_ch0.csv')),
    ('-1.0 V', os.path.join(SEQ_DIR, 'bias_n1p0V', 'eis_ch0.csv')),
    ('-0.8 V', os.path.join(SEQ_DIR, 'bias_n0p8V', 'eis_ch0.csv')),
]


def load(path):
    df = pd.read_csv(path)
    freq = df['freq [Hz]'].values
    z = df['Z_real [ohm]'].values + 1j * df['Z_imag [ohm]'].values
    return freq, z


rows = []
fits = {}
for label, path in DATASETS:
    freq, z = load(path)
    fits[label] = {'freq': freq, 'z': z}
    print(f'\n===== {label}  ({len(freq)} 点, {freq[0]:.0f}–{freq[-1]:.0f} Hz) =====')

    # 短路/失效检测: 阻抗呈纯电阻且量级极小 (接线短路或样品击穿)
    med_z = float(np.median(np.abs(z)))
    med_ph = float(np.median(np.degrees(np.angle(z))))
    if med_z < 20 and abs(med_ph) < 20:
        print(f'  ⚠ 检测到短路特征: |Z| 中位 = {med_z:.3g} Ω, '
              f'相位中位 = {med_ph:.1f}° (全频段近似纯电阻)。'
              '样品疑似击穿或接触短路, 跳过等效电路拟合。')
        rows.append({'数据集': label, '模型': '短路/失效',
                     'Rs [ohm]': med_z, '归一化误差': np.nan})
        fits[label]['shorted'] = True
        continue
    fits[label]['shorted'] = False

    for use_cpe, model in [(False, 'Rs(RpC)'), (True, 'Rs(RpCPE)')]:
        try:
            r = fit_equivalent_circuit(freq, z, use_cpe=use_cpe, freq_min=0)
            fits[label][model] = r
            print(f'【{model}】')
            for name, val, err in zip(r['names'], r['params'], r['params_err']):
                print(f'  {name:>18s} = {val:.6g} ± {err:.3g}')
            print(f'  归一化误差 = {r["norm_error"]:.2%}')
            if 'tau' in r:
                print(f'  τ = {r["tau"]:.4g} s, f_apex = {r["f_apex"]:.4g} Hz')
            row = {'数据集': label, '模型': model}
            row.update({n: v for n, v in zip(r['names'], r['params'])})
            row['归一化误差'] = r['norm_error']
            if 'tau' in r:
                row['tau [s]'] = r['tau']
            rows.append(row)
        except Exception as e:
            print(f'【{model}】拟合失败: {e}')

pd.DataFrame(rows).to_csv(os.path.join(OUT_DIR, '拟合参数.csv'),
                          index=False, encoding='utf-8-sig')

# ---- 图 1: 三组数据 Nyquist + Bode 对比 (含拟合) ----
colors = {'+0.6 V (单次扫描)': 'tab:red', '-1.0 V': 'tab:blue', '-0.8 V': 'tab:green'}
plot_sets = [(l, p) for l, p in DATASETS if not fits[l].get('shorted')]
fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.2))
ax_ny, ax_mag, ax_ph = axes

for label, _ in plot_sets:
    d = fits[label]
    freq, z = d['freq'], d['z']
    c = colors[label]
    ax_ny.plot(z.real, -z.imag, 'o', ms=4, mfc='none', color=c, label=label)
    ax_mag.loglog(freq, np.abs(z), 'o', ms=3, mfc='none', color=c)
    ax_ph.semilogx(freq, np.degrees(np.angle(z)), 'o', ms=3, mfc='none', color=c)
    r = d.get('Rs(RpCPE)')
    if r is not None:
        ax_ny.plot(r['z_fit'].real, -r['z_fit'].imag, '-', lw=1.6, color=c, alpha=0.8)
        ax_mag.loglog(r['freq_fit'], np.abs(r['z_fit']), '-', lw=1.6, color=c, alpha=0.8)
        ax_ph.semilogx(r['freq_fit'], np.degrees(np.angle(r['z_fit'])),
                       '-', lw=1.6, color=c, alpha=0.8)

ax_ny.set_xlabel("Z' [Ω]"); ax_ny.set_ylabel("-Z'' [Ω]")
ax_ny.set_title('Nyquist (Rs(RpCPE) 拟合)')
ax_ny.legend(); ax_ny.grid(alpha=0.3)
ax_mag.set_xlabel('f [Hz]'); ax_mag.set_ylabel('|Z| [Ω]')
ax_mag.set_title('Bode 幅值'); ax_mag.grid(alpha=0.3, which='both')
ax_ph.set_xlabel('f [Hz]'); ax_ph.set_ylabel('Phase [°]')
ax_ph.set_title('Bode 相位'); ax_ph.grid(alpha=0.3, which='both')
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, '对比_Nyquist_Bode.png'), dpi=150)
plt.close(fig)

# ---- 图 2: 参数随偏压变化 (仅 Rs(RpC) 模型) ----
bias_pts = []
for label, _ in DATASETS:
    r = fits[label].get('Rs(RpC)')
    if r is not None:
        bias_pts.append((float(label.split('V')[0].replace('（单次扫描)', '')
                                 .replace('(单次扫描)', '').strip()),
                         r['Rs'], r['Rp'], r['C'], r['tau']))
bias_pts.sort()
if len(bias_pts) >= 2:
    vb = [p[0] for p in bias_pts]
    fig, axs = plt.subplots(1, 4, figsize=(16, 4))
    for ax, idx, name, unit in [(axs[0], 1, 'Rs', 'Ω'), (axs[1], 2, 'Rp', 'Ω'),
                                (axs[2], 3, 'C', 'F'), (axs[3], 4, 'τ = RpC', 's')]:
        vals = [p[idx] for p in bias_pts]
        ax.plot(vb, vals, 'o-', color='tab:blue')
        if name in ('C', 'τ') or max(vals) / max(min(vals), 1e-30) > 50:
            ax.set_yscale('log')
        ax.set_xlabel('$V_{dc}$ [V]'); ax.set_ylabel(f'{name} [{unit}]')
        ax.set_title(f'{name} vs 偏压'); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, '参数随偏压变化.png'), dpi=150)
    plt.close(fig)

print(f'\n输出目录: {OUT_DIR}')
print('完成: 拟合参数.csv, 对比_Nyquist_Bode.png'
      + (', 参数随偏压变化.png' if len(bias_pts) >= 2 else ''))
