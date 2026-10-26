"""
EIS 等效电路拟合模块.

支持模型 (model 参数):
  'RC'     Rs(RpC):     Z = Rs + 1/(1/Rp + jωC)              -- 太阳能电池标准模型
  'CPE'    Rs(RpCPE):   Z = Rs + 1/(1/Rp + Q(jω)^n)          -- 含常相位角元件扩展
  'L-RC'   RsL(RpC):    Z = Rs + jωL + 1/(1/Rp + jωC)        -- 加引线/夹具串联电感
  'L-CPE'  RsL(RpCPE):  Z = Rs + jωL + 1/(1/Rp + Q(jω)^n)

使用复数非线性最小二乘拟合 (scipy.optimize.least_squares),
以复阻抗实部+虚部联合残差为目标.
"""
import logging
import numpy as np
from typing import Dict

logger = logging.getLogger(__name__)

# Rs(RpCPE) 的多起点网格: CPE 指数 n 的误差面几乎平坦 (见 fit_equivalent_circuit
# 里的实测注释), 单一初值会让 Rs 和 n 变成"初值的函数"而不是"数据的函数"。
# 覆盖 0.6–0.99 的典型区间, 跑完取目标函数最小者。
CPE_N_STARTS = (0.60, 0.70, 0.80, 0.90, 0.95, 0.99)

MODELS = {
    'RC': {'label': 'Rs(RpC)', 'keys': ('Rs', 'Rp', 'C')},
    'CPE': {'label': 'Rs(RpCPE)', 'keys': ('Rs', 'Rp', 'Q', 'n')},
    'L-RC': {'label': 'RsL(RpC)', 'keys': ('Rs', 'L', 'Rp', 'C')},
    'L-CPE': {'label': 'RsL(RpCPE)', 'keys': ('Rs', 'L', 'Rp', 'Q', 'n')},
}
PARAM_NAMES = {'Rs': 'Rs [ohm]', 'Rp': 'Rp [ohm]', 'C': 'C [F]', 'Q': 'Q [F·s^(n-1)]',
               'n': 'n', 'L': 'L [H]'}

# 残差权重 w: 残差 = (Z_model - Z) / w
#   unit     w = 1        绝对残差, 低频 MΩ 点主导
#   sqrt     w = √|Z|     折中: 等效于噪声方差 ∝ |Z|
#   modulus  w = |Z|      相对残差: 等效于噪声方差 ∝ |Z|², 与 4284A 的相对误差规格匹配
WEIGHTINGS = ('unit', 'sqrt', 'modulus')
DEFAULT_WEIGHTING = 'sqrt'


def model_rs_rpc(freq: np.ndarray, rs: float, rp: float, c: float) -> np.ndarray:
    """Rs + (Rp || C) 模型复数阻抗."""
    omega = 2.0 * np.pi * np.asarray(freq)
    y_c = 1j * omega * c
    return rs + 1.0 / (1.0 / rp + y_c)


def model_rs_rpcpe(freq: np.ndarray, rs: float, rp: float, q: float, n: float) -> np.ndarray:
    """Rs + (Rp || CPE) 模型复数阻抗, Z_CPE = 1/(Q(jω)^n)."""
    omega = 2.0 * np.pi * np.asarray(freq)
    z_cpe = 1.0 / (q * (1j * omega) ** n)
    return rs + 1.0 / (1.0 / rp + 1.0 / z_cpe)


def model_impedance(model: str, freq: np.ndarray, p: Dict[str, float]) -> np.ndarray:
    """按模型名和参数字典计算复数阻抗."""
    omega = 2.0 * np.pi * np.asarray(freq)
    if 'C' in p:
        y_branch = 1.0 / p['Rp'] + 1j * omega * p['C']
    else:
        y_branch = 1.0 / p['Rp'] + p['Q'] * (1j * omega) ** p['n']
    z = p['Rs'] + 1.0 / y_branch
    if 'L' in p:
        z = z + 1j * omega * p['L']
    return z


def residual_weights(z: np.ndarray, weighting: str) -> np.ndarray:
    if weighting not in WEIGHTINGS:
        raise ValueError(f'未知加权方式: {weighting} (可选 {WEIGHTINGS})')
    mag = np.abs(z)
    if weighting == 'unit':
        return np.ones_like(mag)
    if weighting == 'sqrt':
        return np.sqrt(mag)
    return mag


def fit_equivalent_circuit(freq: np.ndarray, z: np.ndarray,
                           use_cpe: bool = False,
                           freq_min: float = 0.0,
                           max_nfev: int = 20000,
                           model: str = None,
                           weighting: str = DEFAULT_WEIGHTING) -> Dict:
    """
    对复数阻抗谱进行等效电路拟合.

    Parameters
    ----------
    freq : np.ndarray
        频率 (Hz)
    z : np.ndarray
        复数阻抗 (Ω), 与 freq 等长
    use_cpe : bool
        旧接口: 未给 model 时, True 等于 model='CPE', False 等于 model='RC'
    freq_min : float
        仅拟合 freq >= freq_min 的可靠区段
    max_nfev : int
        最大迭代次数
    model : str
        'RC' | 'CPE' | 'L-RC' | 'L-CPE'
    weighting : str
        'unit' | 'sqrt' | 'modulus', 见模块顶部 WEIGHTINGS 说明

    Returns
    -------
    dict: 拟合参数、协方差、拟合质量指标 (含 AIC/BIC)
    """
    try:
        from scipy.optimize import least_squares
    except ImportError:
        raise RuntimeError("scipy not installed. Install via: pip install scipy")

    model = model or ('CPE' if use_cpe else 'RC')
    if model not in MODELS:
        raise ValueError(f'未知模型: {model} (可选 {tuple(MODELS)})')
    keys = MODELS[model]['keys']

    freq = np.asarray(freq, dtype=float)
    z = np.asarray(z, dtype=complex)
    mask = (freq >= freq_min) & np.isfinite(freq) & np.isfinite(z)
    f, zf = freq[mask], z[mask]
    order = np.argsort(f)
    f, zf = f[order], zf[order]

    if len(f) < len(keys) + 2:
        raise ValueError(f"Not enough points for fitting (need >={len(keys) + 2}, got {len(f)})")

    # 初值估计
    # Rs: 最高频实部 (高频截距, 物理上 Rs <= min(Re Z), 故可用作上界)
    # Rp: 最低频实部 - Rs; C: 由弧顶频率估算
    rs0 = max(float(np.median(zf.real[-5:])), 1.0)
    rp0 = max(float(np.median(zf.real[:5])) - rs0, rs0 * 10)
    # Rs 对残差弱约束: 自由拟合时可在不改变拟合质量的前提下漂移数倍
    # (实测 err 仅差 0.05% 而 Rs 差 3.4 倍), 因此必须锚定上界。
    # 锚定依据: 因 Re Z = Rs + Re(Z_rc) 且 Re(Z_rc) >= 0, 故 Rs <= min(Re Z)。
    # (串联电感 jωL 是纯虚数, 不改变这条约束。)
    # 取高频段最小实部作上界 (物理严格); 实测比"中位数×1.2"更自洽——
    # 后者在真实数据上会让 Rs 撞界并超出 min(Re Z), 高频段误差也更高。
    n_hi = min(5, len(f))
    hi_re = zf.real[-n_hi:]
    rs_hi = float(np.min(hi_re))
    if rs_hi <= 0.0:
        # 高频实部出现非正值 —— 实测遇到过: 引线/夹具的串联电感把高频相位推过
        # -90° (628 kHz 处 Re Z = -0.93 Ω), 或噪声所致。此时 "Rs <= min(Re Z)"
        # 这条物理约束已经失效, 再拿它当上界会把 Rs 死死压在 1e-3 的下界上
        # (实测拟合出 Rs=0.001 Ω 这种毫无意义的值), 必须换一个正的上界。
        rs_hi = float(np.median(hi_re))
        if rs_hi <= 0.0:
            rs_hi = float(np.median(np.abs(zf[-n_hi:])))
    rs_hi = max(rs_hi, 1e-3)
    # 中位数 >= 最小值, 直接用 rs0 会超出上界而触发 least_squares 报错
    rs_start = min(rs0, rs_hi * 0.99)

    # 注意: 曾用高频虚部 C ≈ -1/(ω·Im Z) 估初值 (物理上更接近真值),
    # 但实测拟合反而恶化 (err 6.5% → 70.6%) —— 该问题的优化 landscape
    # 对初值敏感, "更准"的初值会落进更差的局部最优。故保留原弧顶频率法。
    f_apex = np.sqrt(f[0] * f[-1])
    cq0 = 1.0 / (2 * np.pi * rp0 * f_apex)
    # L 初值: 高频虚部为正 (感性) 时按 ωL ≈ Im Z 估, 否则取一个在 1 MHz 下
    # 只有毫欧级的小值, 相当于"先当没有电感", 由数据决定要不要它。
    omega_hi = 2 * np.pi * f[-n_hi:]
    pos_im = zf.imag[-n_hi:] > 0
    l0 = float(np.max(zf.imag[-n_hi:][pos_im] / omega_hi[pos_im])) if pos_im.any() else 1e-9

    start = {'Rs': rs_start, 'Rp': rp0, 'C': cq0, 'Q': cq0, 'n': 0.95, 'L': l0}
    bounds = {'Rs': (1e-6, rs_hi), 'Rp': (1e-3, 1e12), 'C': (1e-15, 1e-3),
              'Q': (1e-15, 1e-3), 'n': (0.5, 1.0), 'L': (1e-12, 1e-2)}

    # 参数变换: 除 n 外都取对数再拟合。
    # 各参数相差十几个数量级 (Rs~10 Ω, Rp~1e5 Ω, C~1e-9 F, L~1e-7 H), 线性空间里
    # 优化器的步长对 C、L 几乎无感; 对数空间里每个参数都按"相对变化"移动。
    is_log = [k != 'n' for k in keys]

    def encode(p):
        return np.array([np.log(p[k]) if lg else p[k] for k, lg in zip(keys, is_log)])

    def decode(t):
        return {k: (float(np.exp(v)) if lg else float(v)) for k, v, lg in zip(keys, t, is_log)}

    lower = np.array([np.log(bounds[k][0]) if lg else bounds[k][0] for k, lg in zip(keys, is_log)])
    upper = np.array([np.log(bounds[k][1]) if lg else bounds[k][1] for k, lg in zip(keys, is_log)])

    def clip_start(p):
        t = encode(p)
        span = upper - lower
        return np.clip(t, lower + 1e-6 * span, upper - 1e-6 * span)

    weights = residual_weights(zf, weighting)

    def residual(t):
        diff = (model_impedance(model, f, decode(t)) - zf) / weights
        return np.concatenate([diff.real, diff.imag])

    # CPE 的 landscape 对 n 的初值极其敏感, 单一初值会锁死结果。
    # 实测 (2026-10-14, 五份 0.7 V 扫描): 只改 n 的初值, 同一份数据的
    #   Rs  在 0.12 – 7.90 Ω 之间漂移 (**67 倍**)
    #   n   在 0.83 – 0.98 之间漂移 (0.15)
    #   而归一化误差只从 1.18% 变到 4.04% (3.4 倍)
    # —— 即误差面在 (Rs, n) 方向上几乎平坦, 拟合器停在哪纯看从哪出发。
    # 故用多起点, 全部跑一遍。
    # 挑选依据是优化器实际最小化的加权目标函数 (res.cost)。以前用不加权的
    # norm_error 挑: 它几乎只由低频大阻抗点决定, 会和优化目标不一致 ——
    # 加权残差更小 (高频、Rs 拟合更好) 的那次可能因低频差一点被淘汰。
    n_starts = CPE_N_STARTS if 'n' in keys else (None,)
    best = None
    for n0 in n_starts:
        p_start = dict(start)
        if n0 is not None:
            p_start['n'] = n0
        trial = least_squares(residual, clip_start(p_start), bounds=(lower, upper),
                              max_nfev=max_nfev, x_scale='jac')
        if best is None or trial.cost < best.cost:
            best = trial
    res = best

    p = decode(res.x)
    params = np.array([p[k] for k in keys])
    n_res = 2 * len(f)                    # 实部 + 虚部
    k = len(keys)
    sse = float(np.sum(res.fun ** 2))
    # 参数不确定性: 变换空间里的协方差, 再按 d p = p · d(ln p) 换回原参数
    try:
        sigma2 = sse / max(n_res - k, 1)
        cov_t = sigma2 * np.linalg.inv(res.jac.T @ res.jac)
        sd_t = np.sqrt(np.abs(np.diag(cov_t)))
        perr = np.array([p[kk] * s if lg else s for kk, s, lg in zip(keys, sd_t, is_log)])
    except Exception:
        perr = np.full_like(params, np.nan)

    z_model = model_impedance(model, f, p)
    delta = z_model - zf
    rmse = float(np.sqrt(np.mean(np.abs(delta) ** 2)))
    norm_err = rmse / float(np.sqrt(np.mean(np.abs(zf) ** 2)))
    # 相对误差: 每个点的 |ΔZ|/|Z| 取均方根, 与加权方式无关, 各频段一视同仁
    rel_err = float(np.sqrt(np.mean(np.abs(delta) ** 2 / np.abs(zf) ** 2)))
    # 信息准则 (高斯残差、方差未知): 参数多的模型拟合误差必然不更大,
    # 只比误差会永远选复杂模型; AIC/BIC 对每个参数加罚。
    ll_term = n_res * np.log(max(sse, 1e-300) / n_res)
    aic = float(ll_term + 2 * k)
    bic = float(ll_term + k * np.log(n_res))

    result = {
        'model': MODELS[model]['label'],
        'model_key': model,
        'names': [PARAM_NAMES[kk] for kk in keys],
        'params': params,
        'params_err': perr,
        'rmse': rmse,
        'norm_error': norm_err,
        'rel_error': rel_err,
        'cost': float(res.cost),
        'aic': aic,
        'bic': bic,
        'k': k,
        'weighting': weighting,
        # Rs 贴在上界 min(Re Z) 上: 说明 Rs 由约束而不是数据决定, 报告时要注明
        'rs_at_bound': bool(p['Rs'] >= 0.99 * rs_hi),
        # Rp 撞上 1e12 上限: 弧在测量频段内没有闭合 (反偏常见), Rp 只能说"远大于
        # 实测最大 |Z|", 1e12 这个数没有意义
        'rp_at_bound': bool(p['Rp'] >= 0.99 * bounds['Rp'][1]),
        'z_max': float(np.max(np.abs(zf))),
        'n_points': len(f),
        'freq_range': (f.min(), f.max()),
        'z_fit': z_model,
        'freq_fit': f,
    }
    result.update(p)
    if 'C' in p:
        result['tau'] = p['Rp'] * p['C']  # 载流子寿命
        result['f_apex'] = 1.0 / (2 * np.pi * p['Rp'] * p['C'])

    logger.info(f"Fit [{result['model']}, {weighting}]: Rs={result['Rs']:.3g} Ω, "
                f"Rp={result['Rp']:.3g} Ω, rel_err={rel_err:.2%}, BIC={bic:.1f}")
    return result
