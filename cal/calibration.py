"""
每通道开路/短路校准与复数阻抗修正.

2 线法经继电器板引入串联阻抗 Z_ser(ch,f) 与并联导纳 Y_par(ch,f).
对每个通道在探针尖端做开路/短路校准, 逐频率点存储复数校准值,
测量后按标准 OSM 修正公式还原器件真实阻抗.

修正模型 (2 线等效网络):
    Z_meas = Z_ser + (Z_dut ∥ 1/Y_par)
即:
    Z_dut = (Z_meas - Z_short) / (1 - (Z_meas - Z_short) * Y_open)
其中 Z_short 为短路残余(串联阻抗), Y_open = 1/Z_open 为开路残余导纳.
"""
import json
import logging
import numpy as np
from typing import Dict, List

logger = logging.getLogger(__name__)


class ChannelCalibration:
    """单个通道的开路/短路校准数据."""

    def __init__(self, channel: int, freqs: List[float]):
        self.channel = channel
        self.freqs = np.asarray(freqs, dtype=float)
        self.z_open = np.full_like(self.freqs, np.inf, dtype=complex)
        self.z_short = np.zeros_like(self.freqs, dtype=complex)
        self._done = False

    @property
    def calibrated(self) -> bool:
        return self._done

    def set_open(self, z_open: np.ndarray):
        if len(z_open) != len(self.freqs):
            raise ValueError("open array length mismatch")
        self.z_open = np.asarray(z_open, dtype=complex)
        # 开路阻抗极大时视为无穷大(导纳≈0)
        self.z_open[np.isinf(self.z_open)] = 1e30

    def set_short(self, z_short: np.ndarray):
        if len(z_short) != len(self.freqs):
            raise ValueError("short array length mismatch")
        self.z_short = np.asarray(z_short, dtype=complex)
        self._done = True

    def correct(self, z_meas: complex, freq: float) -> complex:
        """对单点测量阻抗做开路/短路修正."""
        if not self._done:
            raise RuntimeError(f"Channel {self.channel} not calibrated")
        # 在频率轴上插值校准值
        z_open = _interp(self.freqs, self.z_open, freq)
        z_short = _interp(self.freqs, self.z_short, freq)

        z_meas_c = z_meas - z_short
        y_open = 1.0 / z_open
        z_dut = z_meas_c / (1.0 - z_meas_c * y_open)
        return z_dut

    def correct_array(self, z_meas: np.ndarray, freqs: np.ndarray) -> np.ndarray:
        """对整条频谱做修正."""
        if not self._done:
            raise RuntimeError(f"Channel {self.channel} not calibrated")
        z_open = np.interp(freqs, self.freqs, self.z_open.real) + \
                 1j * np.interp(freqs, self.freqs, self.z_open.imag)
        z_short = np.interp(freqs, self.freqs, self.z_short.real) + \
                  1j * np.interp(freqs, self.freqs, self.z_short.imag)

        z_meas_c = z_meas - z_short
        y_open = 1.0 / z_open
        return z_meas_c / (1.0 - z_meas_c * y_open)

    def to_dict(self) -> dict:
        return {
            'channel': self.channel,
            'freqs': self.freqs.tolist(),
            'z_open_real': self.z_open.real.tolist(),
            'z_open_imag': self.z_open.imag.tolist(),
            'z_short_real': self.z_short.real.tolist(),
            'z_short_imag': self.z_short.imag.tolist(),
            'done': self._done,
        }

    @classmethod
    def from_dict(cls, d: dict) -> 'ChannelCalibration':
        cal = cls(d['channel'], d['freqs'])
        cal.z_open = np.array(d['z_open_real']) + 1j * np.array(d['z_open_imag'])
        cal.z_short = np.array(d['z_short_real']) + 1j * np.array(d['z_short_imag'])
        cal._done = d.get('done', True)
        return cal


def _interp(freqs: np.ndarray, vals: np.ndarray, freq: float) -> complex:
    re = np.interp(freq, freqs, vals.real)
    im = np.interp(freq, freqs, vals.imag)
    return complex(re, im)


class CalibrationStore:
    """多通道校准表的集合, 支持保存/加载 JSON."""

    def __init__(self):
        self.channels: Dict[int, ChannelCalibration] = {}

    def add(self, cal: ChannelCalibration):
        self.channels[cal.channel] = cal

    def get(self, channel: int) -> ChannelCalibration:
        if channel not in self.channels:
            raise KeyError(f"Channel {channel} has no calibration")
        return self.channels[channel]

    def calibrated_channels(self) -> List[int]:
        return [ch for ch, cal in self.channels.items() if cal.calibrated]

    def save(self, filename: str):
        data = [cal.to_dict() for cal in self.channels.values()]
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
        logger.info(f"Calibration saved to {filename}")

    @classmethod
    def load(cls, filename: str) -> 'CalibrationStore':
        store = cls()
        with open(filename, 'r', encoding='utf-8') as f:
            data = json.load(f)
        for d in data:
            store.add(ChannelCalibration.from_dict(d))
        logger.info(f"Calibration loaded from {filename}")
        return store
