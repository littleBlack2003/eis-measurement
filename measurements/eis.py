"""
EIS 扫描测量流程: 通道切换 → 稳定 → 变频测复数阻抗 → 校准修正.
"""
import logging
import numpy as np
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


def build_freq_axis(freq_start: float, freq_stop: float, points_per_decade: int) -> np.ndarray:
    """对数频率轴."""
    f1 = float(freq_start)
    f2 = float(freq_stop)
    decades = np.log10(f2 / f1)
    num_points = int(decades * int(points_per_decade)) + 1
    return np.logspace(np.log10(f1), np.log10(f2), num_points)


def run_eis_channel(instrument, channel: int, config: Dict,
                    freqs: np.ndarray, calibration=None, on_point=None) -> Dict:
    """
    对单个通道执行 EIS 变频扫描.

    Parameters
    ----------
    instrument : 阻抗分析仪 (HP4284A 或模拟器)
    channel : int
        通道号
    config : dict
        配置
    freqs : np.ndarray
        频率轴 (Hz)
    calibration : ChannelCalibration, optional
        该通道的校准数据; 若给出则对测量结果做开路/短路修正

    Returns
    -------
    dict: freq, z_raw, z_corrected(若有校准), |Z|, theta, C_p, G_p
    """
    eis = config['measurement']['eis']
    vdc = float(eis['v_dc'])
    vac = float(eis['v_ac'])
    avg = int(eis['averaging'])
    settle = float(eis.get('settle_time', 0.08))

    z_raw = []
    for index, f in enumerate(freqs, start=1):
        z = instrument.measure_complex_z(f, vac, vdc, averaging=avg,
                                         settle_time=settle)
        z_raw.append(z)
        if on_point is not None:
            z_display = calibration.correct(z, float(f)) if calibration is not None \
                and calibration.calibrated else z
            on_point(channel, float(f), complex(z_display), index, len(freqs))

    z_raw = np.asarray(z_raw, dtype=complex)

    result = {
        'channel': channel,
        'freq': freqs,
        'z_raw': z_raw,
    }

    if calibration is not None and calibration.calibrated:
        result['z'] = calibration.correct_array(z_raw, freqs)
    else:
        result['z'] = z_raw

    # 派生量: |Z|, 相位, 并联电容/电导
    result['z_mag'] = np.abs(result['z'])
    result['theta_deg'] = np.angle(result['z'], deg=True)
    result['C_p'], result['G_p'] = z_to_cpg(result['z'], freqs)

    logger.info(f"Channel {channel}: EIS scan done, {len(freqs)} points "
                f"(calibrated={calibration is not None and calibration.calibrated})")
    return result


def z_to_cpg(z: np.ndarray, freqs: np.ndarray):
    """
    由复数阻抗转换为并联等效 Cp-Gp.

    Y = 1/Z = Gp + jωCp
    """
    y = 1.0 / z
    gp = y.real
    omega = 2.0 * np.pi * freqs
    cp = y.imag / omega
    return cp, gp


def run_eis_all(instrument, relay, config: Dict,
                calibration_store=None, on_point=None) -> List[Dict]:
    """
    对所有配置通道执行 EIS 扫描.

    Parameters
    ----------
    instrument : 阻抗分析仪
    relay : RelayBoard 或 None (无继电器板时单通道直连)
    config : dict
    calibration_store : CalibrationStore, optional

    Returns
    -------
    list of channel result dicts
    """
    channels = config['measurement']['channels']
    freqs = build_freq_axis(
        config['measurement']['eis']['freq_start'],
        config['measurement']['eis']['freq_stop'],
        config['measurement']['eis']['points_per_decade'],
    )

    results = []
    for ch in channels:
        if relay is not None:
            relay.select(ch)
        elif hasattr(instrument, 'set_channel'):
            instrument.set_channel(ch)
        cal = None
        if calibration_store is not None:
            try:
                cal = calibration_store.get(ch)
            except KeyError:
                logger.warning(f"Channel {ch} has no calibration, using raw Z")

        results.append(run_eis_channel(instrument, ch, config, freqs, cal,
                                       on_point=on_point))

    if relay is not None:
        relay.all_off()
    return results
