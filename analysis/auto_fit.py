"""自动数据处理入口: 定位最新测量 CSV, 短路检测, 等效电路拟合.

供 GUI 调用:
  - find_latest_csv(output_dir) -> list[str]   最新一次扫描的全部通道 CSV
  - fit_csv(path) -> dict                      单文件拟合结果 (含短路检测)
"""
import os
import re
import numpy as np
import pandas as pd

from analysis.fitting import DEFAULT_WEIGHTING, MODELS, fit_equivalent_circuit

# 短路/失效判定阈值: 中位 |Z| 极小且近似纯电阻
SHORT_Z_LIMIT = 20.0      # ohm
SHORT_PHASE_LIMIT = 20.0  # deg

# 模型选择准则。只比误差会永远选参数多的模型 (CPE 在 n=1 时就是 RC, 带 L 的
# 模型在 L→0 时就是不带 L 的模型): 2026-10 对 80 条实测谱的检验中, 按误差
# 选有 47 条选了带 L 的模型, 而拟合出的 L 全都贴在 1e-12 H 下界 —— 纯粹是多
# 一个参数占的便宜。BIC 对每个参数罚 ln(N), 比 AIC 更保守。
CRITERIA = ('bic', 'aic')
DEFAULT_CRITERION = 'bic'
# 只有 >100 kHz 出现感性点 (Im Z > 0) 时才让带串联电感的模型参与比较:
# 上述 80 条谱都没有感性点, BIC 仍有 7 条选了带 L 的模型 (都是假阳性),
# 而且 Rs 随偏压的一致性反而变差。
INDUCTIVE_FREQ = 1e5


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
    - 否则: 返回最新文件所在目录下的全部通道 CSV (多通道单次扫描不能只拟合一个通道)
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
    folder = os.path.dirname(newest)
    return sorted(os.path.join(folder, f) for f in os.listdir(folder)
                  if re.match(r'eis_ch\d+\.csv$', f))


def fit_csv(path: str, weighting: str = DEFAULT_WEIGHTING,
            criterion: str = DEFAULT_CRITERION) -> dict:
    """对单条 EIS 频谱做短路检测与等效电路拟合, 按信息准则选模型."""
    df = pd.read_csv(path)
    freq = df['freq [Hz]'].values.astype(float)
    z = df['Z_real [ohm]'].values + 1j * df['Z_imag [ohm]'].values
    # 仪器状态 -1/1/2 的点存的是 NaN, 留着会让 least_squares 直接报错
    keep = np.isfinite(freq) & np.isfinite(z)
    freq, z = freq[keep], z[keep]
    if len(freq) == 0:
        raise ValueError(f'没有有效数据点: {path}')

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

    if criterion not in CRITERIA:
        raise ValueError(f'未知准则: {criterion} (可选 {CRITERIA})')
    inductive = bool(np.any((freq > INDUCTIVE_FREQ) & (z.imag > 0)))
    candidates = ['RC', 'CPE'] + (['L-RC', 'L-CPE'] if inductive else [])
    fits = {}
    for key in candidates:
        try:
            r = fit_equivalent_circuit(freq, z, model=key, freq_min=0, weighting=weighting)
            fits[r['model']] = r
        except Exception:
            pass
    if not fits:
        raise RuntimeError(f'所有等效电路模型拟合失败: {path}')
    best_name = min(fits, key=lambda k: fits[k][criterion])
    res['best_name'] = best_name
    res['best_fit'] = fits[best_name]
    res['all_fits'] = fits
    res['inductive'] = inductive
    res['criterion'] = criterion
    res['weighting'] = weighting
    return res
