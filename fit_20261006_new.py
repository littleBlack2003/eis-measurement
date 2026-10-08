"""用改进后的拟合策略处理 2026-10-06 晚间测量数据 (三组: +0.6V, -1.0V, -0.8V).

同时给出"旧策略(绝对残差 + Rs 自由)"与"新策略(sqrt(|Z|) 加权 + Rs 锚定高频截距)"的对比,
便于判断改进对真实数据的实际影响。
"""
import os
import sys
import numpy as np
import pandas as pd
from scipy.optimize import least_squares

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analysis.fitting import (model_rs_rpc, model_rs_rpcpe,
                              fit_equivalent_circuit)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

DATA_DIR = os.path.join('dist', 'EISScanGUI-20260923-104343', 'data')
SEQ_DIR = os.path.join(DATA_DIR, 'bias_sequence_20261006_220835')
OUT_DIR = os.path.join('data', '20261006_新拟合')
os.makedirs(OUT_DIR, exist_ok=True)

DATASETS = [
    ('+0.6 V (单次扫描)', os.path.join(DATA_DIR, 'eis_ch0.csv')),
    ('-1.0 V', os.path.join(SEQ_DIR, 'bias_n1p0V', 'eis_ch0.csv')),
    ('-0.8 V', os.path.join(SEQ_DIR, 'bias_n0p8V', 'eis_ch0.csv')),
]

# 短路/失效判定阈值 (与 analysis/auto_fit.py 一致)
SHORT_Z_LIMIT = 20.0
SHORT_PHASE_LIMIT = 20.0


def load(path):
    df = pd.read_csv(path)
    freq = df['freq [Hz]'].values.astype(float)
    z = df['Z_real [ohm]'].values + 1j * df['Z_imag [ohm]'].values
    return freq, z


def old_fit(freq, z, use_cpe=False):
    """旧策略: 绝对残差 + Rs 上界 1e6 + 原弧顶频率 C 初值."""
    rs0 = max(float(np.median(z.real[-5:])), 1.0)
    rp0 = max(float(np.median(z.real[:5])) - rs0, rs0 * 10)
    f_apex = np.sqrt(freq[0] * freq[-1])
    if use_cpe:
        q0 = 1.0 / (rp0 * 2 * np.pi * f_apex)
        p0, lo, up, model = [rs0, rp0, q0, 0.95], [0, 0, 1e-15, 0.5], \
                            [1e6, 1e12, 1e-3, 1.0], model_rs_rpcpe
    else:
        c0 = 1.0 / (2 * np.pi * rp0 * f_apex)
        p0, lo, up, model = [rs0, rp0, c0], [0, 0, 1e-15], \
                            [1e6, 1e12, 1e-3], model_rs_rpc

    def residual(p):
        zm = model(freq, *p)
        return np.concatenate([(zm - z).real, (zm - z).imag])

    o = least_squares(residual, p0, bounds=(lo, up), max_nfev=20000)
    zm = model(freq, *o.x)
    err = np.sqrt(np.mean(np.abs(zm - z) ** 2)) / np.sqrt(np.mean(np.abs(z) ** 2))
    return o.x, err, zm


rows = []
fits = {}
for label, path in DATASETS:
    if not os.path.exists(path):
        print(f'[跳过] 文件不存在: {path}')
        continue
    freq, z = load(path)
    fits[label] = {'freq': freq, 'z': z}
    print(f'\n===== {label}  ({len(freq)} 点, {freq[0]:.0f}–{freq[-1]:.0f} Hz) =====')
    print(f'  |Z| 范围: {np.abs(z).min():.4g} ~ {np.abs(z).max():.4g} Ω')

    # 短路/失效检测
    med_z = float(np.median(np.abs(z)))
    med_ph = float(np.median(np.degrees(np.angle(z))))
    if med_z < SHORT_Z_LIMIT and abs(med_ph) < SHORT_PHASE_LIMIT:
        print(f'  ⚠ 短路/失效: |Z|中位={med_z:.3g} Ω, 相位中位={med_ph:.1f}° → 跳过拟合')
        fits[label]['shorted'] = True
        rows.append({'数据集': label, '策略': '—', '模型': '短路/失效',
                     'Rs [ohm]': med_z, '归一化误差': np.nan})
        continue
    fits[label]['shorted'] = False

    for use_cpe, model_name in [(False, 'Rs(RpC)'), (True, 'Rs(RpCPE)')]:
        # --- 旧策略 ---
        try:
            po, eo, _ = old_fit(freq, z, use_cpe)
        except Exception as exc:
            print(f'  【旧 {model_name}】失败: {exc}')
            po, eo = None, np.nan

        # --- 新策略 ---
        try:
            a = fit_equivalent_circuit(freq, z, use_cpe=use_cpe, freq_min=0)
        except Exception as exc:
            print(f'  【新 {model_name}】失败: {exc}')
            a = None

        if po is not None:
            extra_o = f'Q={po[2]:.4g} n={po[3]:.3f}' if use_cpe else f'C={po[2]:.4g}'
            print(f'  旧 {model_name:9s} Rs={po[0]:10.4g} Rp={po[1]:10.4g} '
                  f'{extra_o}  err={eo*100:6.2f}%')
            rows.append({'数据集': label, '策略': '旧', '模型': model_name,
                         'Rs [ohm]': po[0], 'Rp [ohm]': po[1],
                         '归一化误差': eo})
        if a is not None:
            extra_n = (f'Q={a["Q"]:.4g} n={a["n"]:.3f}' if use_cpe
                       else f'C={a["C"]:.4g} tau={a["tau"]:.4g}s')
            print(f'  新 {model_name:9s} Rs={a["Rs"]:10.4g} Rp={a["Rp"]:10.4g} '
                  f'{extra_n}  err={a["norm_error"]*100:6.2f}%')
            rows.append({'数据集': label, '策略': '新', '模型': model_name,
                         'Rs [ohm]': a['Rs'], 'Rp [ohm]': a['Rp'],
                         '归一化误差': a['norm_error']})
            if not use_cpe:
                fits[label]['new_fit'] = a
                fits[label]['old_fit'] = (po, eo)

df = pd.DataFrame(rows)
csv_path = os.path.join(OUT_DIR, '拟合对比_新旧.csv')
df.to_csv(csv_path, index=False, encoding='utf-8-sig')
print(f'\n参数表: {csv_path}')

# ---- 图: Nyquist + Bode, 新策略拟合曲线 ----
plot_sets = [(l, d) for l, d in fits.items() if not d.get('shorted') and 'new_fit' in d]
colors = ['tab:red', 'tab:blue', 'tab:green']
if plot_sets:
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.2))
    ax_ny, ax_mag, ax_ph = axes
    for i, (label, d) in enumerate(plot_sets):
        freq, z = d['freq'], d['z']
        a = d['new_fit']
        c = colors[i % len(colors)]
        ax_ny.plot(z.real, -z.imag, 'o', ms=4, mfc='none', color=c, label=label)
        ax_mag.loglog(freq, np.abs(z), 'o', ms=3, mfc='none', color=c)
        ax_ph.semilogx(freq, np.degrees(np.angle(z)), 'o', ms=3, mfc='none', color=c)
        zf = model_rs_rpc(freq, a['Rs'], a['Rp'], a['C'])
        ax_ny.plot(zf.real, -zf.imag, '-', lw=1.6, color=c, alpha=0.85)
        ax_mag.loglog(freq, np.abs(zf), '-', lw=1.6, color=c, alpha=0.85)
        ax_ph.semilogx(freq, np.degrees(np.angle(zf)), '-', lw=1.6, color=c, alpha=0.85)
    ax_ny.set_xlabel("Z' [Ω]"); ax_ny.set_ylabel("-Z'' [Ω]")
    ax_ny.set_title('Nyquist (新策略 Rs(RpC) 拟合)')
    ax_ny.legend(); ax_ny.grid(alpha=0.3)
    ax_mag.set_xlabel('f [Hz]'); ax_mag.set_ylabel('|Z| [Ω]')
    ax_mag.set_title('Bode 幅值'); ax_mag.grid(alpha=0.3, which='both')
    ax_ph.set_xlabel('f [Hz]'); ax_ph.set_ylabel('Phase [°]')
    ax_ph.set_title('Bode 相位'); ax_ph.grid(alpha=0.3, which='both')
    fig.tight_layout()
    p = os.path.join(OUT_DIR, '对比_Nyquist_Bode_新拟合.png')
    fig.savefig(p, dpi=150); plt.close(fig)
    print(f'拟合图: {p}')

# ---- 图: Rs / Rp / C 随偏压变化 (新策略) ----
pts = []
for label, d in fits.items():
    if d.get('shorted') or 'new_fit' not in d:
        continue
    try:
        v = float(label.split('V')[0].replace('（单次扫描)', '').replace('(单次扫描)', '').strip())
    except ValueError:
        continue
    a = d['new_fit']
    pts.append((v, a['Rs'], a['Rp'], a['C'], a['tau']))
pts.sort()
if len(pts) >= 2:
    vb = [p[0] for p in pts]
    fig, axs = plt.subplots(1, 4, figsize=(16, 4))
    for ax, idx, name, unit in [(axs[0], 1, 'Rs', 'Ω'), (axs[1], 2, 'Rp', 'Ω'),
                                (axs[2], 3, 'C', 'F'), (axs[3], 4, 'τ=RpC', 's')]:
        vals = [p[idx] for p in pts]
        ax.plot(vb, vals, 'o-', color='tab:blue')
        if name in ('C', 'τ') or max(vals) / max(min(vals), 1e-30) > 50:
            ax.set_yscale('log')
        ax.set_xlabel('$V_{dc}$ [V]'); ax.set_ylabel(f'{name} [{unit}]')
        ax.set_title(f'{name} vs 偏压 (新策略)'); ax.grid(alpha=0.3)
    fig.tight_layout()
    p = os.path.join(OUT_DIR, '参数随偏压变化_新拟合.png')
    fig.savefig(p, dpi=150); plt.close(fig)
    print(f'参数趋势图: {p}')

print(f'\n输出目录: {OUT_DIR}')
