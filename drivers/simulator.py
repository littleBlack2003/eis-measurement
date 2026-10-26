"""
EIS 仿真器: 每个通道模拟一个 RC 网络 (Rs + Rsh ∥ C), 返回复数阻抗.
用于无硬件时验证校准与多通道切换逻辑.
"""
import logging
import numpy as np
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)


class EISSimulator:
    """RC 网络阻抗仿真器, 接口与 HP4284A 驱动一致."""

    def __init__(self, config: Dict):
        sim = config.get('simulator', {})
        self.channels = sim.get('channels', {})
        self.noise = float(sim.get('noise_level', 0.005))
        self._function = 'ztd'
        self._channel = 0
        self.temperature = 300.0     # K, 仅用于陷阱模型; TAS 流程会按输入的实际温度更新

    def set_sample_temperature(self, kelvin: float):
        """仿真专用: 真实仪器不知道样品温度, 由操作员手动控温。"""
        self.temperature = float(kelvin)

    def close(self):
        """仿真器无需资源释放."""
        pass

    def set_channel(self, channel: int):
        """记录当前选中通道 (仿真继电器切换)."""
        self._channel = channel

    def _z(self, channel: int, freq: float, vdc: float, vac: float) -> complex:
        p = self.channels.get(str(channel), self.channels.get(channel, {}))
        rs = float(p.get('series_resistance', 25.0))
        rsh = float(p.get('shunt_resistance', 1e7))
        c = float(p.get('capacitance', 2e-9))

        omega = 2.0 * np.pi * freq
        # 并联 RC 支路 (含结电容对偏压的一阶依赖)
        y = 1.0 / rsh + 1j * omega * c * (1.0 - 0.05 * vdc)
        trap = p.get('trap')
        if trap:
            # 陷阱支路 C_t 串 R_t, 时间常数 1/ω0, ω0 = 2 ξ0 T² exp(-Ea/kT):
            # 低频 C = C + C_t, 高频冻结回 C, -dC/dlnω 在 ω0 处出峰
            t = self.temperature
            w0 = 2.0 * float(trap['xi0']) * t ** 2 * np.exp(
                -float(trap['energy']) / (8.617333262e-5 * t))
            ct = float(trap['capacitance'])
            y += 1j * omega * ct / (1.0 + 1j * omega / w0)
        return rs + 1.0 / y

    def measure_complex_z(self, freq: float, vac: float, vdc: float = 0.0,
                          averaging: int = 1, settle_time: float = 0.05) -> complex:
        z = self._z(self._channel, freq, vdc, vac)
        if self.noise > 0:
            z *= 1.0 + self.noise * np.random.randn()
        return z

    def measure_channel_z(self, channel: int, freq: float, vac: float,
                          vdc: float = 0.0, averaging: int = 1,
                          settle_time: float = 0.05) -> complex:
        """按指定通道返回阻抗 (供多通道仿真)."""
        z = self._z(channel, freq, vdc, vac)
        if self.noise > 0:
            z *= 1.0 + self.noise * np.random.randn()
        return z
