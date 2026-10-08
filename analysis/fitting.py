"""
EIS 等效电路拟合模块.

支持模型:
  1. Rs(RpC):  Z = Rs + 1/(1/Rp + jωC)          -- 太阳能电池标准模型
  2. Rs(RpCPE): Z = Rs + 1/(1/Rp + Q(jω)^n)      -- 含常相位角元件扩展

使用复数非线性最小二乘拟合 (scipy.optimize.least_squares),
以复阻抗实部+虚部联合残差为目标.
"""
import logging
import numpy as np
from typing import Dict, Optional, Tuple

logger = logging.getLogger(__name__)


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


def fit_equivalent_circuit(freq: np.ndarray, z: np.ndarray,
                           use_cpe: bool = False,
                           freq_min: float = 0.0,
                           max_nfev: int = 20000) -> Dict:
    """
    对复数阻抗谱进行等效电路拟合.

    Parameters
    ----------
    freq : np.ndarray
        频率 (Hz)
    z : np.ndarray
        复数阻抗 (Ω), 与 freq 等长
    use_cpe : bool
        是否使用 CPE 模型 (默认 False, 用理想电容)
    freq_min : float
        仅拟合 freq >= freq_min 的可靠区段
    max_nfev : int
        最大迭代次数

    Returns
    -------
    dict: 拟合参数、协方差、拟合质量指标
    """
    try:
        from scipy.optimize import least_squares
    except ImportError:
        raise RuntimeError("scipy not installed. Install via: pip install scipy")

    freq = np.asarray(freq, dtype=float)
    z = np.asarray(z, dtype=complex)
    mask = freq >= freq_min
    f, zf = freq[mask], z[mask]

    if len(f) < 5:
        raise ValueError(f"Not enough points for fitting (need >=5, got {len(f)})")

    # 初值估计
    # Rs: 最高频实部 (高频截距, 物理上 Rs <= min(Re Z), 故可用作上界)
    # Rp: 最低频实部 - Rs; C: 由弧顶频率估算
    rs0 = max(float(np.median(zf.real[-5:])), 1.0)
    rp0 = max(float(np.median(zf.real[:5])) - rs0, rs0 * 10)
    # Rs 对残差弱约束: 自由拟合时可在不改变拟合质量的前提下漂移数倍
    # (实测 err 仅差 0.05% 而 Rs 差 3.4 倍), 因此必须锚定上界。
    # 锚定依据: 因 Re Z = Rs + Re(Z_rc) 且 Re(Z_rc) >= 0, 故 Rs <= min(Re Z)。
    # 取高频段最小实部作上界 (物理严格); 实测比"中位数×1.2"更自洽——
    # 后者在真实数据上会让 Rs 撞界并超出 min(Re Z), 高频段误差也更高。
    n_hi = min(5, len(f))
    rs_hi = max(float(np.min(zf.real[-n_hi:])), 1e-3)
    # 中位数 >= 最小值, 直接用 rs0 会超出上界而触发 least_squares 报错
    rs_start = min(rs0, rs_hi * 0.99)

    # 注意: 曾用高频虚部 C ≈ -1/(ω·Im Z) 估初值 (物理上更接近真值),
    # 但实测拟合反而恶化 (err 6.5% → 70.6%) —— 该问题的优化 landscape
    # 对初值敏感, "更准"的初值会落进更差的局部最优。故保留原弧顶频率法。
    f_apex = np.sqrt(f[0] * f[-1])
    if use_cpe:
        q0 = 1.0 / (rp0 * 2 * np.pi * f_apex)
        n0 = 0.95
        p0 = [rs_start, rp0, q0, n0]
        lower = [0, 0, 1e-15, 0.5]
        upper = [rs_hi, 1e12, 1e-3, 1.0]
        model = model_rs_rpcpe
    else:
        # 由弧顶频率 f_c = 1/(2πRpC) 估算 C
        c0 = 1.0 / (2 * np.pi * rp0 * f_apex)
        p0 = [rs_start, rp0, c0]
        lower = [0, 0, 1e-15]
        upper = [rs_hi, 1e12, 1e-3]
        model = model_rs_rpc

    # 残差加权: 除以 sqrt(|Z|)。
    # 全频段绝对残差下, 低频 MΩ 级点会主导优化, 牺牲 Rs 等小量级参数
    # (实测 Rs 拟合出 9.9e4 而真值 25)。sqrt 加权平衡各频段贡献。
    weights = np.sqrt(np.abs(zf))

    def residual(p):
        z_model = model(f, *p)
        diff = (z_model - zf) / weights
        return np.concatenate([diff.real, diff.imag])

    res = least_squares(residual, p0, bounds=(lower, upper), max_nfev=max_nfev)

    params = res.x
    # 参数不确定性 (从雅可比近似协方差)
    try:
        jac = res.jac
        dof = max(len(zf) * 2 - len(params), 1)
        sse = np.sum(res.fun ** 2)
        sigma2 = sse / dof
        cov = sigma2 * np.linalg.inv(jac.T @ jac)
        perr = np.sqrt(np.abs(np.diag(cov)))
    except Exception:
        perr = np.full_like(params, np.nan)

    # 拟合质量: 归一化残差
    z_model = model(f, *params)
    rmse = np.sqrt(np.mean(np.abs(z_model - zf) ** 2))
    norm_err = rmse / np.sqrt(np.mean(np.abs(zf) ** 2))

    if use_cpe:
        names = ['Rs [ohm]', 'Rp [ohm]', 'Q [F·s^(n-1)]', 'n']
    else:
        names = ['Rs [ohm]', 'Rp [ohm]', 'C [F]']

    result = {
        'model': 'Rs(RpCPE)' if use_cpe else 'Rs(RpC)',
        'names': names,
        'params': params,
        'params_err': perr,
        'rmse': rmse,
        'norm_error': norm_err,
        'n_points': len(f),
        'freq_range': (f.min(), f.max()),
        'z_fit': z_model,
        'freq_fit': f,
    }
    if not use_cpe:
        rs, rp, c = params
        result['Rs'] = rs
        result['Rp'] = rp
        result['C'] = c
        result['tau'] = rp * c  # 载流子寿命
        result['f_apex'] = 1.0 / (2 * np.pi * rp * c)
    else:
        rs, rp, q, n = params
        result['Rs'] = rs
        result['Rp'] = rp
        result['Q'] = q
        result['n'] = n

    logger.info(f"Fit [{result['model']}]: Rs={result['Rs']:.1f} Ω, "
                f"Rp={result['Rp']:.1f} Ω, norm_err={norm_err:.2%}")
    return result
