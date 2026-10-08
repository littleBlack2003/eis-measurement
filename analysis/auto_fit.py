"""自动数据处理入口: 定位最新测量 CSV, 短路检测, 等效电路拟合.

供 GUI 调用:
  - find_latest_csv(output_dir) -> list[str]   最新一次扫描的全部通道 CSV
  - fit_csv(path) -> dict                      单文件拟合结果 (含短路检测)
"""
import os
import re
import numpy as np
import pandas as pd

from analysis.fitting import fit_equivalent_circuit

# 短路/失效判定阈值: 中位 |Z| 极小且近似纯电阻
SHORT_Z_LIMIT = 20.0      # ohm
SHORT_PHASE_LIMIT = 20.0  # deg


def _bias_from_folder(name: str) -> float:
    """bias_n1p0V -> -1.0; bias_p0p5V -> +0.5; bias_0p0V -> 0.0."""
    name = name.rstrip('V')
    if name.startswith('bias_'):
        name = name[len('bias_'):]
    sign = -1.0 if name.startswith('n') else 1.0
    num = name.lstrip('np').replace('p', '.')
    return sign * float(num)


def _label_for(path: str) -> str:
    parent = os.path.basename(os.path.dirname(path))
    fname = os.path.basename(path)
    m = re.match(r'eis_ch(\d+)\.csv', fname)
    ch = m.group(1) if m else '?'
    if parent.startswith('bias_'):
        return f'ch{ch} @ {_bias_from_folder(parent):+g} V'
    return f'ch{ch} 最新扫描'


def find_latest_csv(output_dir: str) -> list:
    """返回最新一次扫描对应的 CSV 列表 (按偏压排序).

    - 若最新文件位于 bias_sequence_* 目录: 返回该序列下全部 bias_*/eis_ch*.csv
    - 否则: 仅返回该最新文件本身
    """
    hits = []
    for root, _dirs, files in os.walk(output_dir):
        for f in files:
            if re.match(r'eis_ch\d+\.csv$', f):
                hits.append(os.path.join(root, f))
    if not hits:
        raise FileNotFoundError(f'输出目录下没有找到测量数据: {output_dir}')
    newest = max(hits, key=os.path.getmtime)
    seq_dir = os.path.dirname(newest)
    seq_name = os.path.basename(seq_dir)
    if os.path.basename(os.path.dirname(seq_dir)).startswith('bias_sequence_') \
            or seq_name.startswith('bias_'):
        seq_dir = os.path.dirname(seq_dir) if not seq_name.startswith('bias_') else seq_dir
    parent_name = os.path.basename(os.path.dirname(seq_dir))
    if parent_name.startswith('bias_sequence_') or seq_name.startswith('bias_'):
        base = os.path.dirname(seq_dir) if parent_name.startswith('bias_sequence_') \
            else seq_dir
        out = []
        for entry in os.listdir(base):
            sub = os.path.join(base, entry)
            if entry.startswith('bias_') and os.path.isdir(sub):
                for f in os.listdir(sub):
                    if re.match(r'eis_ch\d+\.csv$', f):
                        out.append(os.path.join(sub, f))
        out.sort(key=lambda p: (_bias_from_folder(
            os.path.basename(os.path.dirname(p))), os.path.getmtime(p)))
        return out
    return [newest]


def fit_csv(path: str) -> dict:
    """对单条 EIS 频谱做短路检测与等效电路拟合."""
    df = pd.read_csv(path)
    freq = df['freq [Hz]'].values.astype(float)
    z = df['Z_real [ohm]'].values + 1j * df['Z_imag [ohm]'].values

    res = {
        'path': path,
        'label': _label_for(path),
        'freq': freq,
        'z': z,
        'n_points': len(freq),
        'freq_range': (float(freq[0]), float(freq[-1])),
    }

    med_z = float(np.median(np.abs(z)))
    med_ph = float(np.median(np.degrees(np.angle(z))))
    if med_z < SHORT_Z_LIMIT and abs(med_ph) < SHORT_PHASE_LIMIT:
        res['shorted'] = True
        res['short_z'] = med_z
        res['short_ph'] = med_ph
        return res
    res['shorted'] = False

    fits = {}
    for use_cpe, name in [(False, 'Rs(RpC)'), (True, 'Rs(RpCPE)')]:
        try:
            fits[name] = fit_equivalent_circuit(freq, z, use_cpe=use_cpe, freq_min=0)
        except Exception:
            pass
    if not fits:
        raise RuntimeError(f'所有等效电路模型拟合失败: {path}')
    best_name = min(fits, key=lambda k: fits[k]['norm_error'])
    res['best_name'] = best_name
    res['best_fit'] = fits[best_name]
    res['all_fits'] = fits
    return res
