"""
HP/Agilent 4284A 阻抗分析仪 EIS 驱动.

以 Z-θ 模式测量复数阻抗, 供阻抗谱 (EIS) 使用.
返回复数 Z = |Z| * exp(jθ), 并支持转换为 Cp-Gp.
"""
import time
import logging
from typing import Tuple, Optional
import numpy as np

try:
    import pyvisa
    VISA_AVAILABLE = True
except ImportError:
    VISA_AVAILABLE = False
    pyvisa = None

logger = logging.getLogger(__name__)

# FETCH? 第三个字段的含义 (操作手册 7-8, Data Transfer / ASCII Format)
FETCH_STATUS = {
    -1: '数据缓冲区无数据',
    0: '正常',
    1: '模拟电桥失衡',
    2: 'A/D 转换器不工作',
    3: '信号源过载',
    4: 'ALC 无法稳压',
}
# 这几种状态下仪器返回的是占位值 9.9E37 而不是读数 (手册 7-8 Note);
# 3、4 仍是实际测量值, 只是不保证准确, 保留数值并靠 status 列标记
INVALID_STATUS = (-1, 1, 2)


class HP4284A:
    """HP/Agilent 4284A LCR Meter via GPIB/USB/LAN (Z-θ 模式)."""

    def __init__(self, address: str, timeout: int = 30, function: str = 'ztd'):
        if not VISA_AVAILABLE:
            raise RuntimeError("PyVISA not installed. Install via: pip install pyvisa")

        self.rm = pyvisa.ResourceManager()
        self.inst = self.rm.open_resource(address)
        self.inst.timeout = timeout * 1000  # ms
        self.inst.write_termination = '\n'
        self.inst.read_termination = '\n'

        self.reset()
        idn = self.query("*IDN?")
        logger.info(f"Connected to: {idn.strip()}")
        # 最近一次 measure_complex_z 的 FETCH? 状态码, 调用方据此标记坏点
        self.last_status = 0

        self.function = function.lower()
        if self.function == 'ztd':
            # Z-θ 模式
            self.write("FUNC:IMP ZTD")
        elif self.function == 'cpg':
            self.write("FUNC:IMP CPG")
        else:
            raise ValueError(f"Unknown function: {function}")
        self.write("APER MED,2")
        # *RST 后触发源是 INT (内部连续触发, 附录 C-2): 仪器一直在自己测,
        # FETCH? 可能拿到改频率/偏压之前那一次的结果。改为 BUS, 只在
        # TRIG:IMM 时测一次, 与手册的程控示例 (7-14 Figure 7-11) 一致。
        self.write("TRIG:SOUR BUS")

    def reset(self):
        self.write("*RST")
        self.write("*CLS")
        time.sleep(0.5)

    def write(self, cmd: str):
        self.inst.write(cmd)

    def query(self, cmd: str) -> str:
        return self.inst.query(cmd)

    def close(self):
        self.inst.close()
        self.rm.close()

    def set_frequency(self, freq: float):
        if not (20 <= freq <= 1e6):
            raise ValueError(f"Frequency {freq} Hz out of range [20, 1e6]")
        self.write(f"FREQ {freq}")

    def set_ac_level(self, vrms: float):
        if not (0.0 <= vrms <= 2.0):
            raise ValueError(f"AC level {vrms} Vrms out of range [0, 2.0]")
        self.write(f"VOLT:LEV {vrms}")

    def set_dc_bias(self, vdc: float):
        if not (-40 <= vdc <= 40):
            raise ValueError(f"DC bias {vdc} V out of range [-40, 40]")
        self.write(f"BIAS:VOLT {vdc}")
        self.write("BIAS:STAT ON")

    def disable_dc_bias(self):
        self.write("BIAS:STAT OFF")

    def set_averaging(self, n: int):
        # 操作手册 8-39 写 1~128; 数据手册写 1~256 (可能是后期固件), 取保守值
        if not (1 <= n <= 128):
            raise ValueError(f"Averaging {n} out of range [1, 128]")
        self.write(f"APER MED,{n}")

    def trigger_and_fetch(self) -> Tuple[float, float, int]:
        """
        触发单次测量并获取主/副参数.

        4284A 的 FETCH? 直接返回 3 个字段: (主参数, 副参数, 状态),
        无需再查询 STAT? (该命令在此机型会超时).

        返回 (primary, secondary, status):
          - ZTD 模式: (|Z| [Ω], θ [deg])
          - CPG 模式: (Cp [F], Gp [S])
        """
        self.write("TRIG:IMM")
        result = self.query("FETCH?")
        parts = result.strip().split(',')
        prim = float(parts[0])
        sec = float(parts[1])
        status = int(parts[2]) if len(parts) > 2 else 0
        return prim, sec, status

    def measure_complex_z(self, freq: float, vac: float, vdc: float = 0.0,
                          averaging: int = 1, settle_time: float = 0.05) -> complex:
        """
        在单频点测量复数阻抗 Z = Z' + j Z''.

        Parameters
        ----------
        freq : float
            频率 (Hz)
        vac : float
            交流幅值 (Vrms)
        vdc : float
            直流偏压 (V)
        averaging : int
            平均次数
        settle_time : float
            设置频率/偏压后的稳定等待时间 (秒), 取自配置 measurement.eis.settle_time

        Returns
        -------
        complex
            复数阻抗 (Ω)。状态码为 -1/1/2 时返回 nan+nanj;
            状态码本身存在 self.last_status。
        """
        if self.function != 'ztd':
            self.write("FUNC:IMP ZTD")
            self.function = 'ztd'
        self.set_frequency(freq)
        self.set_ac_level(vac)
        if abs(vdc) > 1e-6:
            self.set_dc_bias(vdc)
        else:
            self.disable_dc_bias()
        self.set_averaging(averaging)

        time.sleep(settle_time)
        z_mag, theta_deg, status = self.trigger_and_fetch()
        self.last_status = status
        if status != 0:
            logger.warning(f"Z measure status={status} "
                           f"({FETCH_STATUS.get(status, '未知')}) at f={freq:.1f} Hz")
        if status in INVALID_STATUS:
            # 9.9E37 当成 |Z| 存下去, 在 √|Z| 加权的拟合里一个点的残差就有 ~3e18
            return complex(np.nan, np.nan)

        theta = np.deg2rad(theta_deg)
        return z_mag * np.exp(1j * theta)

    def sweep_frequency_z(self, vdc: float, vac: float, freq_start: float,
                          freq_stop: float, points_per_decade: int = 10,
                          averaging: int = 2, settle_time: float = 0.08):
        """
        对数变频扫描, 逐点返回复数阻抗.

        Yields
        ------
        freq : float
        z : complex
        """
        decades = np.log10(freq_stop / freq_start)
        num_points = int(decades * points_per_decade) + 1
        freqs = np.logspace(np.log10(freq_start), np.log10(freq_stop), num_points)

        self.set_ac_level(vac)
        self.set_averaging(averaging)
        if abs(vdc) > 1e-6:
            self.set_dc_bias(vdc)
        else:
            self.disable_dc_bias()

        for f in freqs:
            z = self.measure_complex_z(f, vac, vdc, averaging)
            time.sleep(settle_time)
            yield float(f), z

    def sweep_frequency_cpg(self, vdc: float, vac: float, freq_start: float,
                            freq_stop: float, points_per_decade: int = 10,
                            averaging: int = 2):
        """对数变频扫描 Cp-Gp 模式, 逐点返回 (freq, Cp, Gp)."""
        if self.function != 'cpg':
            self.write("FUNC:IMP CPG")
            self.function = 'cpg'
        decades = np.log10(freq_stop / freq_start)
        num_points = int(decades * points_per_decade) + 1
        freqs = np.logspace(np.log10(freq_start), np.log10(freq_stop), num_points)

        self.set_ac_level(vac)
        self.set_averaging(averaging)
        if abs(vdc) > 1e-6:
            self.set_dc_bias(vdc)
        else:
            self.disable_dc_bias()

        for f in freqs:
            self.set_frequency(float(f))
            time.sleep(0.08)
            cp, gp, status = self.trigger_and_fetch()
            if status != 0:
                logger.warning(f"CPG sweep f={f:.1f} status={status}")
            yield float(f), cp, gp
