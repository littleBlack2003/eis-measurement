"""
USB 串口继电器板驱动 (自定义 ASCII 协议).

通过 COM 口发送文本命令切换通道. 命令模板在配置中定义,
如 "CH{ch} ON\\r" 或 "REL,{ch:02d},1\\r", 其中 {ch} 为通道号占位符.
"""
import time
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class RelayBoard:
    """多通道继电器板, 每次仅导通一个通道 (1-of-N 复用)."""

    def __init__(self, port: str, baudrate: int = 115200,
                 cmd_on: str = "CH{ch} ON\r",
                 cmd_off: str = "CH{ch} OFF\r",
                 cmd_all_off: str = "ALLOFF\r",
                 settle_time: float = 0.2,
                 n_channels: int = 8,
                 timeout: float = 1.0):
        try:
            import serial
        except ImportError:
            raise RuntimeError("pyserial not installed. Install via: pip install pyserial")

        self.port = port
        self.ser = serial.Serial(port, baudrate, timeout=timeout)
        self.cmd_on = cmd_on
        self.cmd_off = cmd_off
        self.cmd_all_off = cmd_all_off
        self.settle_time = settle_time
        self.n_channels = n_channels
        self._active: Optional[int] = None

        # 上电复位: 全部关断
        self.all_off()
        logger.info(f"Relay board on {port} ready, {n_channels} channels")

    def close(self):
        """断开串口."""
        try:
            self.all_off()
        finally:
            self.ser.close()

    def send(self, cmd: str):
        """发送命令并清空可能的应答缓存."""
        self.ser.reset_input_buffer()
        self.ser.write(cmd.encode('ascii'))
        self.ser.flush()
        time.sleep(self.settle_time)

    def all_off(self):
        """全部通道关断."""
        if self.cmd_all_off:
            self.ser.reset_input_buffer()
            self.ser.write(self.cmd_all_off.encode('ascii'))
            self.ser.flush()
            time.sleep(self.settle_time)
        self._active = None

    def select(self, channel: int):
        """导通指定通道, 其余关断 (1-of-N)."""
        if not (0 <= channel < self.n_channels):
            raise ValueError(f"Channel {channel} out of range [0, {self.n_channels-1}]")

        if self._active == channel:
            return  # 已导通, 无需切换

        # 先关断旧通道再导通新通道, 避免瞬间并联
        if self._active is not None:
            self.send(self.cmd_off.format(ch=channel_format(self._active)))
        self.send(self.cmd_on.format(ch=channel_format(channel)))
        self._active = channel
        logger.info(f"Relay switched to channel {channel}")

    @property
    def active(self) -> Optional[int]:
        return self._active


def channel_format(ch: int) -> str:
    """默认通道号格式: 不带前导零 (可按协议调整)."""
    return str(ch)
