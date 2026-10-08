"""阻抗谱自动测量桌面界面。"""
import cmath
import copy
import csv
from datetime import datetime
from decimal import Decimal, InvalidOperation
import logging
import math
import os
import queue
import sys
import threading
import time
import traceback
from pathlib import Path

import dearpygui.dearpygui as dpg
import numpy as np
import yaml

from eis_main import create_instrument, create_relay, run_calibration, run_eis
from cal.calibration import CalibrationStore
from analysis.auto_fit import fit_csv, find_latest_csv


APP_DIR = Path(sys.executable).resolve().parent if getattr(sys, 'frozen', False) \
    else Path(__file__).resolve().parent
CONFIG_PATH = APP_DIR / 'eis_config.yaml'
# 打包后附加数据落在 onedir 的 _internal 目录（sys._MEIPASS），源码运行时就在项目根
BUNDLE_DIR = Path(getattr(sys, '_MEIPASS', APP_DIR))
# 应用图标（抽象 Nyquist 图）：ICO 供 PyInstaller 打包 exe，也用于设置窗口图标
ICON_ICO = BUNDLE_DIR / 'assets' / 'app_icon.ico'
# DearPyGui 的 load_image 不支持 ICO，另存 PNG 备用
ICON_PNG = BUNDLE_DIR / 'assets' / 'app_icon_ui.png'


class QueueLogHandler(logging.Handler):
    def __init__(self, event_queue):
        super().__init__()
        self.event_queue = event_queue
        self.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(message)s',
                                            datefmt='%H:%M:%S'))

    def emit(self, record):
        self.event_queue.put(('log', self.format(record)))


class CalibrationCancelled(Exception):
    pass


class ScanCancelled(Exception):
    pass


def parse_bias_sequence(text):
    """解析 0:1:0.1 或 -0.5,0,0.5 形式的偏压序列。"""
    spec = text.strip().replace('，', ',')
    if not spec:
        raise ValueError('请填写偏压序列。')
    try:
        if ':' in spec:
            parts = [Decimal(part.strip()) for part in spec.split(':')]
            if len(parts) != 3:
                raise ValueError('范围格式应为 起点:终点:步长，例如 0:1:0.1。')
            start, stop, step = parts
            if step <= 0:
                raise ValueError('偏压步长必须大于 0。')
            direction = 1 if stop >= start else -1
            values = []
            value = start
            while (value <= stop if direction == 1 else value >= stop):
                values.append(value)
                if len(values) > 200:
                    raise ValueError('偏压点不能超过 200 个。')
                value += direction * step
            if values[-1] != stop:
                values.append(stop)
        else:
            values = [Decimal(part.strip()) for part in spec.split(',') if part.strip()]
    except InvalidOperation as exc:
        raise ValueError('偏压序列中有无效数字。') from exc
    if not values:
        raise ValueError('请填写至少一个偏压值。')
    if len(values) > 200:
        raise ValueError('偏压点不能超过 200 个。')
    if any(not value.is_finite() or value < -40 or value > 40 for value in values):
        raise ValueError('偏压值必须在 -40 V 到 +40 V 之间。')
    if len(set(values)) != len(values):
        raise ValueError('偏压序列中不能有重复值。')
    return [float(value) for value in values]


def parse_int(value, name, minimum=None, maximum=None):
    """把界面文本解析为整数, 失败时给出中文提示而非抛出原始异常。"""
    text = str(value).strip()
    try:
        result = int(text)
    except (TypeError, ValueError):
        raise ValueError(f'「{name}」必须是整数，当前输入：{text or "(空)"}')
    if minimum is not None and result < minimum:
        raise ValueError(f'「{name}」不能小于 {minimum}。')
    if maximum is not None and result > maximum:
        raise ValueError(f'「{name}」不能大于 {maximum}。')
    return result


def parse_float(value, name, minimum=None, maximum=None, allow_zero=True):
    """把界面文本解析为浮点数, 失败时给出中文提示。"""
    text = str(value).strip()
    try:
        result = float(text)
    except (TypeError, ValueError):
        raise ValueError(f'「{name}」必须是数字，当前输入：{text or "(空)"}')
    if not math.isfinite(result):
        raise ValueError(f'「{name}」必须是有限数值。')
    if not allow_zero and result == 0:
        raise ValueError(f'「{name}」不能为 0。')
    if minimum is not None and result < minimum:
        raise ValueError(f'「{name}」不能小于 {minimum}。')
    if maximum is not None and result > maximum:
        raise ValueError(f'「{name}」不能大于 {maximum}。')
    return result


def bias_folder_name(bias):
    value = Decimal(str(bias)).normalize()
    text = format(value, 'f')
    if '.' not in text:
        text += '.0'
    return 'bias_' + text.replace('-', 'n').replace('.', 'p') + 'V'


class EISGui:
    def __init__(self):
        self.events = queue.Queue()
        self.worker = None
        self.conn_worker = None          # 连接测试／资源枚举线程
        self.cancel_event = threading.Event()
        self.prompt_response = None
        self.prompt_result = None
        self.live_data = {}
        self.live_series = {}
        self.fit_series = []
        self.fit_results = []
        self._last_axis_update = 0.0
        try:
            with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
                self.config = yaml.safe_load(f) or {}
        except Exception as exc:
            self.config = {}
            self._load_error = str(exc)
        else:
            self._load_error = ''

        logging.getLogger().setLevel(logging.INFO)
        self.log_handler = QueueLogHandler(self.events)
        logging.getLogger().addHandler(self.log_handler)
        self._create_ui()
        if self._load_error:
            self._show_message('配置读取失败', f'无法读取 {CONFIG_PATH}\n{self._load_error}')

    # ---------- 视觉常量（明亮主题：深色文字 + 浅色底） ----------
    ACCENT = (24, 92, 168, 255)           # 强调蓝：标题、勾选标记
    HINT = (104, 116, 132, 255)           # 说明文字：中灰（太浅在白底上看不清）
    OK_GREEN = (26, 138, 76, 255)         # 成功／就绪
    WARN_AMBER = (186, 126, 20, 255)      # 警告
    ERR_RED = (198, 60, 48, 255)          # 失败
    # 左右分栏：左栏放连接/测量参数，右栏放数据窗口
    LEFT_WIDTH = 470                      # 左栏总宽
    LEFT_CONTENT = 440                    # 左栏可用内容宽（说明文字换行用）
    RIGHT_WIDTH = 900                     # 右栏总宽（与 1420 视口匹配，可被缩放回调改写）
    CONTENT_WIDTH = 874                   # 右栏内容统一宽度
    PANEL_HEIGHT = 850                    # 两栏高度（超出各自滚动）

    # 明亮主题配色：控件走 mvThemeCat_Core，图表走 mvThemeCat_Plots
    LIGHT_CORE = {
        'WindowBg': (245, 247, 250), 'ChildBg': (255, 255, 255),
        'PopupBg': (255, 255, 255), 'MenuBarBg': (240, 244, 249),
        'Text': (28, 34, 44), 'TextDisabled': (130, 140, 154),
        'TextSelectedBg': (176, 209, 245),
        'Border': (196, 205, 216), 'BorderShadow': (234, 238, 244),
        'Separator': (203, 211, 221),
        'FrameBg': (255, 255, 255), 'FrameBgHovered': (236, 243, 251),
        'FrameBgActive': (219, 233, 247),
        'Button': (223, 231, 241), 'ButtonHovered': (203, 216, 232),
        'ButtonActive': (178, 199, 223),
        'Header': (223, 231, 241), 'HeaderHovered': (203, 216, 232),
        'HeaderActive': (178, 199, 223),
        'Tab': (226, 233, 242), 'TabHovered': (204, 217, 234),
        'TabActive': (255, 255, 255), 'TabUnfocused': (233, 238, 245),
        'TabUnfocusedActive': (245, 249, 252),
        'TitleBg': (233, 239, 246), 'TitleBgActive': (214, 227, 242),
        'TitleBgCollapsed': (234, 240, 247),
        'ScrollbarBg': (242, 245, 250), 'ScrollbarGrab': (188, 199, 213),
        'ScrollbarGrabHovered': (168, 181, 197), 'ScrollbarGrabActive': (148, 162, 180),
        'CheckMark': (24, 92, 168), 'SliderGrab': (24, 92, 168),
        'SliderGrabActive': (16, 72, 136),
        'TableHeaderBg': (230, 238, 247), 'TableRowBg': (255, 255, 255),
        'TableRowBgAlt': (247, 250, 253), 'TableBorderStrong': (176, 189, 204),
        'TableBorderLight': (214, 222, 232),
        'ResizeGrip': (200, 210, 222), 'ResizeGripHovered': (170, 184, 200),
        'ResizeGripActive': (140, 158, 178),
        'NavHighlight': (210, 226, 244),
        'ModalWindowDimBg': (70, 82, 98, 90),   # 需要半透明，否则弹窗背景全黑
    }
    LIGHT_PLOTS = {
        'PlotBg': (255, 255, 255), 'FrameBg': (250, 251, 253),
        'PlotBorder': (198, 207, 218),
        'LegendBg': (255, 255, 255), 'LegendBorder': (206, 214, 224),
        'LegendText': (40, 48, 60),
        'AxisBg': (250, 251, 253), 'AxisBgHovered': (240, 246, 252),
        'AxisBgActive': (232, 240, 250), 'AxisGrid': (223, 230, 239),
        'AxisText': (55, 65, 78), 'AxisTick': (95, 106, 120),
        'TitleText': (32, 40, 52), 'InlayText': (40, 48, 60),
        'Selection': (198, 220, 245), 'Crosshairs': (110, 125, 145),
    }

    def _build_theme(self):
        """明亮主题：统一圆角与间距，并把控件、表格、图表整套改成浅底深字。

        图表有独立的配色类别（mvThemeCat_Plots），只改 Core 会让曲线区域仍然
        是深色，所以两套一起设。
        """
        with dpg.theme(tag='app_theme'):
            with dpg.theme_component(dpg.mvAll):
                dpg.add_theme_style(dpg.mvStyleVar_WindowPadding, 14, 10,
                                    category=dpg.mvThemeCat_Core)
                dpg.add_theme_style(dpg.mvStyleVar_FramePadding, 8, 5,
                                    category=dpg.mvThemeCat_Core)
                dpg.add_theme_style(dpg.mvStyleVar_ItemSpacing, 10, 7,
                                    category=dpg.mvThemeCat_Core)
                dpg.add_theme_style(dpg.mvStyleVar_FrameRounding, 4,
                                    category=dpg.mvThemeCat_Core)
                dpg.add_theme_style(dpg.mvStyleVar_ChildRounding, 4,
                                    category=dpg.mvThemeCat_Core)
                dpg.add_theme_style(dpg.mvStyleVar_WindowRounding, 6,
                                    category=dpg.mvThemeCat_Core)
                dpg.add_theme_style(dpg.mvStyleVar_ScrollbarSize, 14,
                                    category=dpg.mvThemeCat_Core)
                for name, rgb in self.LIGHT_CORE.items():
                    rgba = tuple(rgb) if len(rgb) == 4 else (*rgb, 255)
                    dpg.add_theme_color(getattr(dpg, 'mvThemeCol_' + name), rgba,
                                        category=dpg.mvThemeCat_Core)
                for name, rgb in self.LIGHT_PLOTS.items():
                    dpg.add_theme_color(getattr(dpg, 'mvPlotCol_' + name), (*rgb, 255),
                                        category=dpg.mvThemeCat_Plots)
        dpg.bind_theme('app_theme')

    def _button_themes(self):
        """给三个主按钮上色，避免四个按钮一模一样而误点"停止扫描"。

        明亮主题下正文是深色，所以实底按钮必须同时把按钮文字刷成白色，
        否则深底 + 深字根本看不见。
        """
        schemes = (
            ('start_button', (45, 125, 78), (56, 150, 94), (68, 172, 108)),
            ('bias_start_button', (38, 108, 152), (48, 132, 182), (60, 152, 206)),
            ('stop_button', (176, 62, 62), (204, 74, 74), (228, 90, 90)),
        )
        for tag, base, hover, active in schemes:
            if not dpg.does_item_exist(tag):
                continue
            with dpg.theme(tag=f'{tag}_theme'):
                with dpg.theme_component(dpg.mvButton):
                    dpg.add_theme_color(dpg.mvThemeCol_Button, (*base, 255),
                                        category=dpg.mvThemeCat_Core)
                    dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, (*hover, 255),
                                        category=dpg.mvThemeCat_Core)
                    dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, (*active, 255),
                                        category=dpg.mvThemeCat_Core)
                    dpg.add_theme_color(dpg.mvThemeCol_Text, (255, 255, 255, 255),
                                        category=dpg.mvThemeCat_Core)
            dpg.bind_item_theme(tag, f'{tag}_theme')

    def _hint(self, text, width=None):
        """淡色说明文字，给参数区补充格式或环境要求。"""
        dpg.add_text(text, color=self.HINT, wrap=width or self.CONTENT_WIDTH)

    def _set_window_icon(self):
        """用 Win32 API 给窗口设图标（标题栏 + 任务栏）。

        为什么不用 DearPyGui 自带的接口：2.3.1 下 create_viewport(small_icon=...)
        在 Windows 上实测不生效（事后 configure_viewport 更不行——GLFW 只在建窗时
        读该参数），截图确认标题栏一直是默认图标。这里拿到 HWND 后直接
        LoadImage + WM_SETICON，可靠且不需要重启窗口。

        拿不到图标不影响功能（exe 自身已由 PyInstaller 写入图标，资源管理器正常），
        所以失败只记日志。
        """
        if not ICON_ICO.exists():
            return
        try:
            self._apply_win32_icon()
        except Exception:
            logging.getLogger(__name__).exception('设置窗口图标失败')

    def _apply_win32_icon(self):
        import ctypes
        from ctypes import wintypes

        user32 = ctypes.WinDLL('user32', use_last_error=True)
        user32.FindWindowW.restype = wintypes.HWND
        user32.LoadImageW.restype = wintypes.HANDLE

        # DearPyGui 的视口标题，用它在桌面上定位窗口
        hwnd = user32.FindWindowW(None, 'EIS Impedance Scanner')
        if not hwnd:
            return
        IMAGE_ICON, LR_LOADFROMFILE, LR_DEFAULTSIZE = 1, 0x0010, 0x0040
        # 大图标给 Alt+Tab / 任务栏，小图标给标题栏
        for kind, size in ((1, 32), (0, 16)):        # ICON_BIG / ICON_SMALL
            handle = user32.LoadImageW(None, str(ICON_ICO), IMAGE_ICON,
                                       size, size, LR_LOADFROMFILE)
            if handle:
                user32.SendMessageW(hwnd, 0x0080, kind, handle)   # WM_SETICON
        logging.getLogger(__name__).info('已设置窗口图标：%s', ICON_ICO.name)

    def _create_ui(self):
        instrument = self.config.get('instrument', {})
        relay = self.config.get('relay_board', {})
        measurement = self.config.get('measurement', {})
        eis = measurement.get('eis', {})
        calibration = measurement.get('calibration', {})
        output = self.config.get('output', {})

        dpg.create_context()
        # 注意：DearPyGui 2.3.1 的 create_viewport(title=...) 传入中文会在首帧渲染时
        # 触发访问违规崩溃（窗口 label 用中文没问题，只有视口标题不行），故保持 ASCII。
        dpg.create_viewport(title='EIS Impedance Scanner', width=1420, height=950,
                            min_width=1000, min_height=600)
        self._build_theme()
        with dpg.window(tag='main_window', label='阻抗谱自动测量系统'):
            with dpg.group(horizontal=True):
                dpg.add_text('EIS 阻抗谱自动测量系统', color=self.ACCENT)
                dpg.add_text(f'配置文件：{CONFIG_PATH.name}', color=self.HINT)
            dpg.add_separator()
            # 两栏必须同属一个 horizontal group：child_window 会结束外层的横向布局，
            # 后继同级控件会退回纵向堆叠。
            with dpg.group(horizontal=True):
                # ================= 左栏：连接选项 + 测量参数（默认折叠） =================
                with dpg.child_window(tag='left_panel', width=self.LEFT_WIDTH,
                                      height=self.PANEL_HEIGHT, border=True):
                    with dpg.collapsing_header(label='连接选项（仪器与通道）',
                                               default_open=False):
                        with dpg.group(horizontal=True):
                            current_mode = instrument.get('type', 'hp4284a')
                            mode_value = '仿真器' if current_mode == 'simulator' else 'HP 4284A'
                            dpg.add_combo(['HP 4284A', '仿真器'], label='测量设备',
                                          default_value=mode_value, tag='mode', width=180)
                            dpg.add_input_text(label='超时（秒）', default_value=str(
                                instrument.get('timeout', 30)), tag='timeout', width=90)
                        dpg.add_input_text(label='GPIB 地址', default_value=instrument.get(
                            'address', 'GPIB0::17::INSTR'), tag='address', width=300)
                        with dpg.group(horizontal=True):
                            dpg.add_input_text(label='串口', default_value=relay.get('port', 'COM3'),
                                               tag='port', width=100)
                            channels = ','.join(map(str, measurement.get('channels', [0])))
                            dpg.add_input_text(label='测量通道', default_value=channels,
                                               tag='channels', width=240)
                        dpg.add_checkbox(label='启用继电器板', default_value=bool(
                            relay.get('enabled', False)), tag='relay_enabled')
                        with dpg.group(horizontal=True):
                            dpg.add_button(label='列出可用仪器', tag='list_resources_button',
                                           callback=self._list_resources,
                                           width=150, height=30)
                            self._hint('不确定地址就先点它扫一遍 VISA。', width=270)
                        self._hint('通道可填多个，用逗号分隔（例如 0,1,2）。GPIB 需要本机安装 '
                                   'NI-VISA 或 Keysight IO Libraries；串口／TCP 才可用内置的 '
                                   'pyvisa-py 后端。', width=self.LEFT_CONTENT)
                    with dpg.collapsing_header(label='测量参数', default_open=False):
                        with dpg.group(horizontal=True):
                            dpg.add_input_text(label='起始频率 Hz', default_value=str(
                                eis.get('freq_start', 20)), tag='freq_start', width=110)
                            dpg.add_input_text(label='终止频率 Hz', default_value=str(
                                eis.get('freq_stop', 1.0e6)), tag='freq_stop', width=110)
                        with dpg.group(horizontal=True):
                            dpg.add_input_text(label='直流偏压 V', default_value=str(
                                eis.get('v_dc', 0.0)), tag='v_dc', width=100)
                            dpg.add_input_text(label='交流幅值 Vrms', default_value=str(
                                eis.get('v_ac', 0.03)), tag='v_ac', width=100)
                        with dpg.group(horizontal=True):
                            dpg.add_input_text(label='每十倍频点数', default_value=str(
                                eis.get('points_per_decade', 20)), tag='points', width=90)
                            dpg.add_input_text(label='平均次数', default_value=str(
                                eis.get('averaging', 3)), tag='averaging', width=90)
                        dpg.add_input_text(label='频点稳定时间 s', default_value=str(
                            eis.get('settle_time', 0.08)), tag='settle_time', width=90)
                        dpg.add_checkbox(label='扫描前执行开路/短路校准', default_value=bool(
                            calibration.get('enabled', True)), tag='calibrate')
                        dpg.add_checkbox(label='校准后确认再测量', default_value=bool(
                            calibration.get('confirm_after', False)), tag='cal_confirm')
                        self._hint('HP 4284A 频率范围 20 Hz – 1 MHz；交流幅值按小信号条件取 '
                                   '10–50 mVrms。总点数 = 每十倍频点数 × 十倍频数 × 通道数。',
                                   width=self.LEFT_CONTENT)
                    with dpg.collapsing_header(label='偏压序列扫描', default_open=False):
                        dpg.add_input_text(label='偏压序列 V', default_value=str(
                            measurement.get('bias_sequence', '0:1:0.1')),
                            tag='bias_sequence', width=300)
                        dpg.add_input_text(label='每个偏压稳定时间 s', default_value=str(
                            measurement.get('bias_settle_s', 0.5)), tag='bias_settle',
                            width=110)
                        self._hint('格式：起点:终点:步长（例如 1:-1:0.2）；或逗号列表（0,0.2,0.5）。'
                                   '切换偏压后先等待上述时间，再开始该偏压的整段扫描。',
                                   width=self.LEFT_CONTENT)
                    dpg.add_separator()
                    dpg.add_input_text(label='输出目录', default_value=str(
                        output.get('directory', './data')), tag='output_dir', width=330)
                    with dpg.group(horizontal=True):
                        dpg.add_button(label='使用默认目录', callback=self._default_output,
                                       width=130, height=30)
                        dpg.add_button(label='打开目录', callback=self._open_output_dir,
                                       width=120, height=30)
                    dpg.add_separator()
                    with dpg.group(horizontal=True):
                        dpg.add_button(label='开始完整扫描', tag='start_button',
                                       callback=self._start, user_data='single',
                                       width=200, height=38)
                        dpg.add_button(label='扫描偏压序列', tag='bias_start_button',
                                       callback=self._start, user_data='bias',
                                       width=200, height=38)
                    with dpg.group(horizontal=True):
                        dpg.add_button(label='停止扫描', tag='stop_button',
                                       callback=self._cancel_scan, enabled=False,
                                       width=140, height=38)
                        dpg.add_button(label='保存配置', callback=self._save_config,
                                       width=140, height=38)
                    # 连接测试放在始终可见的位置：参数区默认是折叠的，藏进去会找不到
                    with dpg.group(horizontal=True):
                        dpg.add_button(label='测试连接', tag='connect_button',
                                       callback=self._test_connection,
                                       width=140, height=30)
                        dpg.add_text('未测试', tag='conn_status', color=self.HINT)
                    dpg.add_spacer(height=4)
                    with dpg.group(horizontal=True):
                        dpg.add_text('进度', color=self.ACCENT)
                        dpg.add_progress_bar(tag='progress', default_value=0.0,
                                             width=self.LEFT_CONTENT - 50, height=18,
                                             overlay='0 / 0 点')
                    with dpg.group(horizontal=True):
                        dpg.add_text('状态', color=self.ACCENT)
                        dpg.add_text('就绪', tag='status', color=self.OK_GREEN)

                # ================= 右栏：数据窗口 =================
                with dpg.child_window(tag='right_panel', width=self.RIGHT_WIDTH,
                                      height=self.PANEL_HEIGHT, border=True):
                    dpg.add_text('数据窗口', color=self.ACCENT)
                    dpg.add_text('实时曲线（已测频点）', color=self.HINT)
                    with dpg.tab_bar():
                        with dpg.tab(label='Nyquist'):
                            with dpg.plot(label='Nyquist', tag='plot_nyquist',
                                          width=self.CONTENT_WIDTH, height=300):
                                dpg.add_plot_legend()
                                dpg.add_plot_axis(dpg.mvXAxis, label='Re(Z) [Ohm]',
                                                  tag='nyquist_x', auto_fit=True)
                                dpg.add_plot_axis(dpg.mvYAxis, label='-Im(Z) [Ohm]',
                                                  tag='nyquist_y', auto_fit=True)
                        with dpg.tab(label='Bode |Z|'):
                            with dpg.plot(label='Bode magnitude', tag='plot_magnitude',
                                          width=self.CONTENT_WIDTH, height=300):
                                dpg.add_plot_legend()
                                dpg.add_plot_axis(dpg.mvXAxis, label='Frequency [Hz]',
                                                  tag='magnitude_x',
                                                  scale=dpg.mvPlotScale_Log10,
                                                  auto_fit=True)
                                dpg.add_plot_axis(dpg.mvYAxis, label='|Z| [Ohm]',
                                                  tag='magnitude_y', auto_fit=True)
                        with dpg.tab(label='Bode 相位'):
                            with dpg.plot(label='Bode phase', tag='plot_phase',
                                          width=self.CONTENT_WIDTH, height=300):
                                dpg.add_plot_legend()
                                dpg.add_plot_axis(dpg.mvXAxis, label='Frequency [Hz]',
                                                  tag='phase_x',
                                                  scale=dpg.mvPlotScale_Log10,
                                                  auto_fit=True)
                                dpg.add_plot_axis(dpg.mvYAxis, label='Phase [degree]',
                                                  tag='phase_y', auto_fit=True)
                    dpg.add_separator()
                    with dpg.collapsing_header(label='数据处理（等效电路拟合）',
                                               default_open=True):
                        with dpg.group(horizontal=True):
                            dpg.add_button(label='拟合最新数据', callback=self._fit_latest_clicked,
                                           width=140, height=30)
                            dpg.add_checkbox(label='扫描完成后自动拟合', default_value=bool(
                                (measurement.get('fitting') or {}).get('auto_fit', True)),
                                tag='auto_fit')
                        self._hint('自动比较 Rs(RpC) 与 Rs(RpCPE) 两个模型并取误差较小者；'
                                   '残差按 √|Z| 加权，Rs 受限于 min(Re Z)。'
                                   '|Z| 中位 < 20 Ω 的数据判定为短路，不参与拟合。')
                        with dpg.group(horizontal=True):
                            with dpg.plot(label='Nyquist 拟合', tag='fit_plot_nyq',
                                          width=425, height=290):
                                dpg.add_plot_legend()
                                dpg.add_plot_axis(dpg.mvXAxis, label='Re(Z) [Ohm]',
                                                  tag='fit_nyq_x', auto_fit=True)
                                dpg.add_plot_axis(dpg.mvYAxis, label='-Im(Z) [Ohm]',
                                                  tag='fit_nyq_y', auto_fit=True)
                            with dpg.plot(label='Bode |Z| 拟合', tag='fit_plot_bode',
                                          width=425, height=290):
                                dpg.add_plot_legend()
                                dpg.add_plot_axis(dpg.mvXAxis, label='Frequency [Hz]',
                                                  tag='fit_bode_x',
                                                  scale=dpg.mvPlotScale_Log10,
                                                  auto_fit=True)
                                dpg.add_plot_axis(dpg.mvYAxis, label='|Z| [Ohm]',
                                                  tag='fit_bode_y', auto_fit=True)
                        with dpg.group(horizontal=True):
                            dpg.add_button(label='导出拟合报告', callback=self._export_fit_report,
                                           width=140, height=30)
                            dpg.add_text('', tag='fit_export_hint')
                        with dpg.child_window(tag='fit_table_win', height=150,
                                              width=self.CONTENT_WIDTH, border=False):
                            with dpg.table(tag='fit_table', header_row=True, resizable=True,
                                           policy=dpg.mvTable_SizingStretchProp,
                                           borders_innerH=True, borders_outerH=True,
                                           borders_innerV=True, borders_outerV=True,
                                           scrollY=True):
                                for col in ('数据集', '模型', 'Rs [Ω]', 'Rp [Ω]',
                                            'C [F] / Q', 'n', 'τ [s]', '误差'):
                                    dpg.add_table_column(label=col)
                        dpg.add_input_text(tag='fit_summary', multiline=True, readonly=True,
                                           width=self.CONTENT_WIDTH, height=110,
                                           default_value='暂无拟合结果。点击“拟合最新数据”，'
                                                         '或勾选“扫描完成后自动拟合”。')
                    dpg.add_separator()
                    with dpg.group(horizontal=True):
                        dpg.add_text('运行日志')
                        dpg.add_button(label='清空日志', callback=self._clear_log, width=100)
                    dpg.add_input_text(tag='log_text', multiline=True, readonly=True,
                                       width=self.CONTENT_WIDTH, height=170, default_value='')

        with dpg.window(label='开路/短路校准', tag='calibration_dialog', modal=True,
                        show=False, no_title_bar=True, width=470, height=180):
            dpg.add_text('', tag='calibration_prompt', wrap=440)
            dpg.add_spacer(height=15)
            with dpg.group(horizontal=True):
                dpg.add_button(label='继续', callback=self._answer_prompt, user_data=True, width=120)
                dpg.add_button(label='取消校准', callback=self._answer_prompt,
                               user_data=False, width=120)

        with dpg.window(label='提示', tag='message_dialog', modal=True, show=False,
                        no_title_bar=False, width=520, height=220):
            dpg.add_text('', tag='message_title')
            dpg.add_separator()
            dpg.add_text('', tag='message_text', wrap=490)
            dpg.add_spacer(height=10)
            dpg.add_button(label='确定', callback=lambda: dpg.configure_item(
                'message_dialog', show=False), width=100)

        font_dir = Path(os.environ.get('WINDIR', r'C:\Windows')) / 'Fonts'
        for font_name in ('msyh.ttc', 'Deng.ttf', 'simhei.ttf', 'simsun.ttc'):
            font_path = font_dir / font_name
            if font_path.exists():
                with dpg.font_registry():
                    cjk_font = dpg.add_font(str(font_path), 19)
                dpg.bind_font(cjk_font)
                break

        self._button_themes()
        dpg.set_primary_window('main_window', True)
        # 视口尺寸变化（最大化／拖动边框）后同步重算两栏，避免面板固定尺寸不跟随
        dpg.set_viewport_resize_callback(self._on_viewport_resize)
        dpg.setup_dearpygui()
        dpg.show_viewport()
        # show_viewport() 之后原生窗口才存在，此刻才能设置标题栏/任务栏图标
        self._set_window_icon()
        # 启动后先按实际视口（含 DPI 缩放后的真实尺寸）校正一次
        self._apply_responsive_layout()

    def _on_viewport_resize(self, sender=None, app_data=None, user_data=None):
        self._apply_responsive_layout()

    def _apply_responsive_layout(self):
        """按当前视口重算两栏与右栏控件的尺寸。

        面板和图表原本都是写死的宽高：窗口一旦最大化，它们不会跟着放大，
        内容就挤在左上角、其余区域留下一片空白（看起来像"全屏异常"）。
        这里在每次视口变化后同时重算宽度和高度。
        """
        try:
            vw = dpg.get_viewport_client_width()
            vh = dpg.get_viewport_client_height()
        except Exception:
            return
        if not vw or not vh:
            return
        # 窗口内边距（左右各 ~14）+ 标题行与分隔行占用的高度
        height = max(320, vh - 100)
        right_w = max(420, vw - 40 - self.LEFT_WIDTH - 10)
        content = max(400, right_w - 26)
        dpg.configure_item('left_panel', height=height)
        dpg.configure_item('right_panel', width=right_w, height=height)
        for item in ('plot_nyquist', 'plot_magnitude', 'plot_phase'):
            dpg.configure_item(item, width=content)
        half = max(200, (content - 10) // 2)
        for item in ('fit_plot_nyq', 'fit_plot_bode'):
            dpg.configure_item(item, width=half)
        for item in ('fit_table_win', 'fit_table', 'fit_summary', 'log_text'):
            dpg.configure_item(item, width=content)
        self._apply_vertical_layout(height)

    def _apply_vertical_layout(self, height):
        """把多出来的纵向空间分配给图表／表格／日志。

        基准视口下右栏内容本来就超出面板高度（靠滚动查看），所以只在视口
        确实更高时才放大各控件，窗口变小时一律回落到设计尺寸，避免图表被
        压得没法看。
        """
        surplus = height - self.PANEL_HEIGHT
        if surplus <= 0:
            plan = {
                'plot_nyquist': 300, 'plot_magnitude': 300, 'plot_phase': 300,
                'fit_plot_nyq': 290, 'fit_plot_bode': 290,
                'fit_table_win': 150, 'fit_table': 150,
                'fit_summary': 110, 'log_text': 170,
            }
        else:
            plan = {
                'plot_nyquist': 300 + surplus * 0.30,
                'plot_magnitude': 300 + surplus * 0.30,
                'plot_phase': 300 + surplus * 0.30,
                'fit_plot_nyq': 290 + surplus * 0.25,
                'fit_plot_bode': 290 + surplus * 0.25,
                'fit_table_win': 150 + surplus * 0.20,
                'fit_table': 150 + surplus * 0.20,
                'fit_summary': 110 + surplus * 0.10,
                'log_text': 170 + surplus * 0.15,
            }
        limits = {
            'plot_nyquist': (300, 620), 'plot_magnitude': (300, 620),
            'plot_phase': (300, 620),
            'fit_plot_nyq': (290, 520), 'fit_plot_bode': (290, 520),
            'fit_table_win': (150, 320), 'fit_table': (150, 320),
            'fit_summary': (110, 220), 'log_text': (170, 340),
        }
        for item, value in plan.items():
            low, high = limits[item]
            dpg.configure_item(item, height=int(min(high, max(low, value))))

    def _default_output(self, sender=None, app_data=None, user_data=None):
        dpg.set_value('output_dir', './data')

    def _open_output_dir(self, sender=None, app_data=None, user_data=None):
        """在文件管理器中打开当前输出目录，省去手动复制路径。"""
        try:
            path = Path(self._resolved_output_dir())
            path.mkdir(parents=True, exist_ok=True)
            if hasattr(os, 'startfile'):
                os.startfile(str(path))
            else:
                self._show_message('输出目录', str(path))
        except Exception as exc:
            self._show_message('无法打开目录', str(exc))

    def _clear_log(self, sender=None, app_data=None, user_data=None):
        dpg.set_value('log_text', '')

    def _set_conn_status(self, text, color=None):
        dpg.set_value('conn_status', text)
        dpg.configure_item('conn_status', color=color or self.HINT)

    def _connection_settings(self):
        """只取连接相关设置。

        刻意不走 `_settings()`：那里会连频率、偏压一起校验，填错一个参数就没法
        单独试仪器连通性了。
        """
        config = copy.deepcopy(self.config)
        config.setdefault('instrument', {})
        config['instrument']['type'] = 'simulator' if dpg.get_value('mode') == '仿真器' \
            else 'hp4284a'
        config['instrument']['address'] = dpg.get_value('address').strip()
        config['instrument']['timeout'] = parse_int(dpg.get_value('timeout'), '超时（秒）',
                                                    minimum=1, maximum=3600)
        config['instrument'].setdefault('function', 'ztd')
        config.setdefault('relay_board', {})
        config['relay_board']['enabled'] = dpg.get_value('relay_enabled')
        config['relay_board']['port'] = dpg.get_value('port').strip()
        return config

    def _busy_with_connection(self):
        return bool(self.conn_worker and self.conn_worker.is_alive())

    def _test_connection(self, sender=None, app_data=None, user_data=None):
        """打开仪器读一次 *IDN?，确认地址/后端/继电器板是否可用（不开始测量）。"""
        if self._busy_with_connection():
            return
        # 扫描期间不能再开一次会话：HP4284A 驱动初始化会发 *RST，会把正在测的
        # 仪器复位掉，测量数据也就废了。
        if self.worker and self.worker.is_alive():
            self._show_message('仪器占用中',
                               '扫描正在进行，仪器已被占用。请等扫描结束（或点“停止扫描”）'
                               '后再测试连接。')
            return
        try:
            config = self._connection_settings()
        except Exception as exc:
            self._show_message('设置无效', str(exc))
            return
        dpg.configure_item('connect_button', enabled=False)
        dpg.configure_item('list_resources_button', enabled=False)
        self._set_conn_status('正在连接…')
        self._append_log(f"测试连接：{config['instrument']['address']}")
        self.conn_worker = threading.Thread(target=self._connect_worker,
                                            args=(config,), daemon=True)
        self.conn_worker.start()

    def _connect_worker(self, config):
        instrument = None
        relay = None
        try:
            instrument = create_instrument(config)
            idn = ''
            if hasattr(instrument, 'query'):
                try:
                    idn = (instrument.query('*IDN?') or '').strip()
                except Exception:
                    idn = ''
            lines = ['阻抗分析仪：已连接',
                     f"地址：{config['instrument']['address']}"]
            if idn:
                lines.append(f'识别：{idn}')
            if config['relay_board']['enabled']:
                relay = create_relay(config)
                lines.append(f"继电器板：{config['relay_board']['port']} 就绪"
                             f"（{config['relay_board'].get('n_channels', 8)} 通道）")
            else:
                lines.append('继电器板：未启用（单器件直连）')
            self.events.put(('conn_done', True, '\n'.join(lines)))
        except Exception as exc:
            logging.getLogger(__name__).exception('连接测试失败')
            detail = str(exc)
            if 'VI_ERROR' in detail or 'Insufficient location' in detail:
                detail += ('\n\n可能原因：GPIB 地址填错、仪器未开机，或本机没装 NI-VISA / '
                           'Keysight IO Libraries（内置的 pyvisa-py 不支持 GPIB）。'
                           '\n可展开「连接选项」点「列出可用仪器」确认地址。')
            self.events.put(('conn_done', False, detail))
        finally:
            # 测试连接只是一次握手，用完立刻释放，避免占着 GPIB/串口影响正式扫描
            if relay is not None:
                try:
                    relay.close()
                except Exception:
                    logging.getLogger(__name__).exception('关闭继电器板失败')
            if instrument is not None:
                try:
                    instrument.close()
                except Exception:
                    logging.getLogger(__name__).exception('关闭仪器失败')

    def _list_resources(self, sender=None, app_data=None, user_data=None):
        """枚举 VISA 可见资源，用来确认 GPIB 地址。"""
        if self._busy_with_connection():
            return
        # 枚举本身不动仪器，但会和正在进行的测量抢 VISA 后端，扫描中一律不触发
        if self.worker and self.worker.is_alive():
            self._show_message('仪器占用中', '扫描正在进行，请等扫描结束后再枚举资源。')
            return
        dpg.configure_item('connect_button', enabled=False)
        dpg.configure_item('list_resources_button', enabled=False)
        self._set_conn_status('正在扫描 VISA 资源…')
        self.conn_worker = threading.Thread(target=self._resource_worker, daemon=True)
        self.conn_worker.start()

    def _resource_worker(self):
        try:
            import pyvisa
            rm = pyvisa.ResourceManager()
            try:
                resources = [str(item) for item in rm.list_resources()]
            finally:
                try:
                    rm.close()
                except Exception:
                    pass
            self.events.put(('res_done', True, resources))
        except Exception as exc:
            self.events.put(('res_done', False, str(exc)))

    def _settings(self):
        config = self.config
        config.setdefault('instrument', {})
        config['instrument']['type'] = 'simulator' if dpg.get_value('mode') == '仿真器' else 'hp4284a'
        config['instrument']['address'] = dpg.get_value('address').strip()
        config['instrument']['timeout'] = parse_int(dpg.get_value('timeout'), '超时（秒）',
                                                    minimum=1, maximum=3600)
        config['instrument'].setdefault('function', 'ztd')
        config.setdefault('relay_board', {})
        config['relay_board']['enabled'] = dpg.get_value('relay_enabled')
        config['relay_board']['port'] = dpg.get_value('port').strip()
        channels = []
        for item in dpg.get_value('channels').split(','):
            item = item.strip()
            if not item:
                continue
            try:
                channels.append(int(item))
            except ValueError:
                raise ValueError(f'测量通道「{item}」不是有效整数。')
        if not channels:
            raise ValueError('至少填写一个测量通道。')
        config.setdefault('measurement', {})
        config['measurement']['channels'] = channels
        config['measurement']['bias_sequence'] = dpg.get_value('bias_sequence').strip()
        config['measurement']['bias_settle_s'] = parse_float(
            dpg.get_value('bias_settle'), '偏压稳定时间', minimum=0, maximum=3600)
        config['measurement'].setdefault('eis', {})
        eis = config['measurement']['eis']
        eis['v_dc'] = parse_float(dpg.get_value('v_dc'), '直流偏压 V',
                                  minimum=-40, maximum=40)
        eis['v_ac'] = parse_float(dpg.get_value('v_ac'), '交流幅值 Vrms',
                                  minimum=0, maximum=2, allow_zero=False)
        eis['freq_start'] = parse_float(dpg.get_value('freq_start'), '起始频率 Hz',
                                        minimum=20, maximum=1e6)
        eis['freq_stop'] = parse_float(dpg.get_value('freq_stop'), '终止频率 Hz',
                                       minimum=20, maximum=1e6)
        if eis['freq_stop'] <= eis['freq_start']:
            raise ValueError('终止频率必须大于起始频率。')
        eis['points_per_decade'] = parse_int(dpg.get_value('points'), '每十倍频点数',
                                             minimum=1, maximum=100)
        eis['averaging'] = parse_int(dpg.get_value('averaging'), '平均次数',
                                     minimum=1, maximum=256)
        # 该值此前在界面上没有入口，配置里的 eis.settle_time 无法被修改
        eis['settle_time'] = parse_float(dpg.get_value('settle_time'), '频点稳定时间 s',
                                         minimum=0, maximum=60)
        config['measurement'].setdefault('calibration', {})
        config['measurement']['calibration']['enabled'] = dpg.get_value('calibrate')
        config['measurement']['calibration']['confirm_after'] = dpg.get_value('cal_confirm')
        config['measurement'].setdefault('fitting', {})
        config['measurement']['fitting']['auto_fit'] = dpg.get_value('auto_fit')
        config.setdefault('output', {})
        config['output']['directory'] = dpg.get_value('output_dir').strip() or './data'
        output_path = Path(config['output']['directory']).expanduser()
        if not output_path.is_absolute():
            output_path = CONFIG_PATH.parent / output_path
        config['output']['directory'] = str(output_path.resolve())
        return config

    def _write_config(self, config):
        with open(CONFIG_PATH, 'w', encoding='utf-8') as f:
            yaml.safe_dump(config, f, allow_unicode=True, sort_keys=False)

    def _save_config(self, sender=None, app_data=None, user_data=None):
        try:
            config = self._settings()
            self._write_config(config)
            self._show_message('配置已保存', f'设置已保存到：\n{CONFIG_PATH}')
        except Exception as exc:
            self._show_message('设置无效', str(exc))

    def _start(self, sender=None, app_data=None, user_data=None):
        if self.worker and self.worker.is_alive():
            return
        try:
            config = self._settings()
            biases = parse_bias_sequence(config['measurement']['bias_sequence']) \
                if user_data == 'bias' else None
            self._write_config(config)
        except Exception as exc:
            self._show_message('设置无效', str(exc))
            return
        dpg.configure_item('start_button', enabled=False)
        dpg.configure_item('bias_start_button', enabled=False)
        dpg.configure_item('stop_button', enabled=True)
        dpg.configure_viewport(0, disable_close=True)
        self.cancel_event.clear()
        dpg.set_value('status', '偏压序列运行中…' if biases is not None else '测量运行中…')
        self._reset_live_curves()
        self._append_log('开始测量。')
        self.worker = threading.Thread(target=self._run_worker,
                                       args=(config, biases), daemon=True)
        self.worker.start()

    def _cancel_scan(self, sender=None, app_data=None, user_data=None):
        self.cancel_event.set()
        dpg.configure_item('stop_button', enabled=False)
        dpg.set_value('status', '正在停止扫描…')

    def _check_cancelled(self):
        if self.cancel_event.is_set():
            raise ScanCancelled('扫描已停止。')

    def _request_calibration_prompt(self, text):
        response = threading.Event()
        result = {}
        self.events.put(('prompt', text, response, result))
        # 原先是无限期 wait(): 若界面事件循环卡住或用户已点停止, 工作线程会
        # 永久阻塞导致"停止扫描"失效。改成轮询等待, 期间可响应取消请求。
        while not response.wait(timeout=0.2):
            if self.cancel_event.is_set():
                # 通知界面关闭弹窗, 避免残留的模态框挡住后续操作
                self.events.put(('close_prompt',))
                raise ScanCancelled('已在校准提示处停止扫描。')
        if not result.get('continue', False):
            raise CalibrationCancelled('用户取消了校准。')

    def _point_observer(self, channel, frequency, z, index, total, bias=None):
        self._check_cancelled()
        self.events.put(('point', bias, channel, frequency, z.real, z.imag,
                         abs(z), math.degrees(cmath.phase(z)),
                         index, total))

    def _run_bias_sequence(self, instrument, relay, config, biases,
                           calibration_store, run_dir):
        settle_s = config['measurement']['bias_settle_s']
        channels = ','.join(map(str, config['measurement']['channels']))
        manifest = run_dir / 'bias_sequence.csv'
        rows = []
        with open(run_dir / 'eis_config.yaml', 'w', encoding='utf-8') as f:
            yaml.safe_dump(config, f, allow_unicode=True, sort_keys=False)
        for index, bias in enumerate(biases, start=1):
            self._check_cancelled()
            self.events.put(('bias_start', bias, index, len(biases)))
            logging.getLogger(__name__).info(
                'Bias %d/%d: %+g V', index, len(biases), bias)
            if abs(bias) > 1e-6 and hasattr(instrument, 'set_dc_bias'):
                instrument.set_dc_bias(bias)
            elif abs(bias) <= 1e-6 and hasattr(instrument, 'disable_dc_bias'):
                instrument.disable_dc_bias()
            if settle_s:
                time.sleep(settle_s)
            self._check_cancelled()
            bias_config = copy.deepcopy(config)
            bias_config['measurement']['eis']['v_dc'] = bias
            subdir = bias_folder_name(bias)
            bias_config['output']['directory'] = str(run_dir / subdir)
            run_eis(instrument, relay, bias_config, str(run_dir / subdir),
                    calibration_store,
                    on_point=lambda ch, freq, z, point, total, v=bias:
                    self._point_observer(ch, freq, z, point, total, bias=v))
            rows.append((bias, subdir, channels))
            with open(manifest, 'w', newline='', encoding='utf-8-sig') as f:
                writer = csv.writer(f)
                writer.writerow(('bias_V', 'folder', 'channels'))
                writer.writerows(rows)

    def _run_worker(self, config, biases):
        instrument = None
        relay = None
        success = False
        details = ''
        try:
            instrument = create_instrument(config)
            relay = create_relay(config)
            output_dir = Path(config['output']['directory'])
            output_dir.mkdir(parents=True, exist_ok=True)
            run_dir = output_dir
            if biases is not None:
                stem = 'bias_sequence_' + datetime.now().strftime('%Y%m%d_%H%M%S')
                run_dir = output_dir / stem
                suffix = 2
                while run_dir.exists():
                    run_dir = output_dir / f'{stem}_{suffix}'
                    suffix += 1
                run_dir.mkdir()
                logging.getLogger(__name__).info('Bias sweep output: %s', run_dir)
            calibration_store = None
            if config['measurement']['calibration']['enabled']:
                cal_config = copy.deepcopy(config)
                if biases is not None:
                    cal_config['measurement']['eis']['v_dc'] = 0.0
                calibration_store = run_calibration(
                    instrument, relay, cal_config, str(run_dir),
                    prompt_fn=self._request_calibration_prompt,
                    check_cancel=self._check_cancelled,
                )
                if config['measurement']['calibration'].get('confirm_after'):
                    self._request_calibration_prompt(
                        '校准已完成。请确认样品连接与偏压设置无误，再开始测量。')
            elif (output_dir / 'calibration.json').exists():
                calibration_store = CalibrationStore.load(
                    str(output_dir / 'calibration.json'))
            if biases is None:
                run_eis(instrument, relay, config, str(output_dir),
                        calibration_store, on_point=self._point_observer)
            else:
                self._run_bias_sequence(instrument, relay, config, biases,
                                        calibration_store, run_dir)
            success = True
            details = f'偏压序列扫描完成，共 {len(biases)} 个偏压。\n结果：{run_dir}' \
                if biases is not None else '扫描完成。'
        except CalibrationCancelled as exc:
            details = str(exc)
        except ScanCancelled as exc:
            details = str(exc)
        except Exception as exc:
            logging.getLogger(__name__).exception('测量失败')
            details = f'{exc}\n\n{traceback.format_exc()}'
        finally:
            if instrument is not None and hasattr(instrument, 'disable_dc_bias'):
                try:
                    instrument.disable_dc_bias()
                except Exception:
                    logging.getLogger(__name__).exception('关闭直流偏压失败')
            if relay is not None:
                try:
                    relay.close()
                except Exception:
                    logging.getLogger(__name__).exception('关闭继电器板失败')
            if instrument is not None:
                try:
                    instrument.close()
                except Exception:
                    logging.getLogger(__name__).exception('关闭仪器失败')
        self.events.put(('finished', success, details))

    def _poll_events(self):
        while True:
            try:
                event = self.events.get_nowait()
            except queue.Empty:
                break
            if event[0] == 'log':
                self._append_log(event[1])
            elif event[0] == 'point':
                self._record_point(*event[1:])
            elif event[0] == 'prompt':
                _, text, self.prompt_response, self.prompt_result = event
                dpg.set_value('calibration_prompt', text + '\n\n确认后请选择“继续”。')
                dpg.configure_item('calibration_dialog', show=True)
            elif event[0] == 'close_prompt':
                # 工作线程因取消而放弃等待, 收掉可能残留的模态弹窗
                if self.prompt_response is not None:
                    self.prompt_response.set()
                self.prompt_response = None
                self.prompt_result = None
                dpg.configure_item('calibration_dialog', show=False)
            elif event[0] == 'bias_start':
                _, bias, index, total = event
                dpg.set_value('status', f'偏压 {index}/{total}：{bias:+g} V')
            elif event[0] == 'conn_done':
                _, ok, text = event
                dpg.configure_item('connect_button', enabled=True)
                dpg.configure_item('list_resources_button', enabled=True)
                self._set_conn_status('已连接' if ok else '连接失败',
                                      self.OK_GREEN if ok else self.ERR_RED)
                dpg.set_value('status', '仪器就绪' if ok else '连接失败')
                self._append_log(text)
                self._show_message('连接成功' if ok else '连接失败', text)
            elif event[0] == 'res_done':
                _, ok, resources = event
                dpg.configure_item('connect_button', enabled=True)
                dpg.configure_item('list_resources_button', enabled=True)
                if not ok:
                    self._set_conn_status('枚举失败', self.ERR_RED)
                    self._show_message(
                        '扫描失败', f'无法枚举 VISA 资源：{resources}\n\n'
                        'GPIB 需要本机安装 NI-VISA 或 Keysight IO Libraries。')
                elif resources:
                    self._set_conn_status(f'发现 {len(resources)} 个资源',
                                          self.OK_GREEN)
                    self._append_log('VISA 资源：' + ', '.join(resources))
                    self._show_message(
                        f'发现 {len(resources)} 个 VISA 资源',
                        '\n'.join(resources) +
                        '\n\n把需要的地址填到上面的「GPIB 地址」里，再点「测试连接」。')
                else:
                    self._set_conn_status('未发现资源', self.WARN_AMBER)
                    self._show_message(
                        '未发现 VISA 资源',
                        'VISA 后端没有列出任何仪器。请检查：\n'
                        '· GPIB 是否安装了 NI-VISA / Keysight IO Libraries\n'
                        '· 仪器是否开机、GPIB 线缆是否接好\n'
                        '· 仪器自身的 GPIB 地址（4284A 菜单里查看）')
            elif event[0] == 'fit_done':
                self._show_fit_results(event[1])
            elif event[0] == 'finished':
                _, success, details = event
                dpg.configure_item('start_button', enabled=True)
                dpg.configure_item('bias_start_button', enabled=True)
                dpg.configure_item('stop_button', enabled=False)
                dpg.configure_viewport(0, disable_close=False)
                dpg.set_value('status', '扫描完成' if success else (
                    '扫描已停止' if '停止' in details else
                    '校准已取消' if '取消' in details else '测量失败'))
                self._show_message('完成' if success else '扫描已停止' if '停止' in details
                                   else '测量失败', details)
                if success and dpg.get_value('auto_fit'):
                    threading.Thread(target=self._fit_worker, daemon=True).start()

    def _delete_series(self, tag):
        """删除曲线时一并回收它的着色主题，避免长时间扫描累积泄漏。"""
        if dpg.does_item_exist(tag):
            dpg.delete_item(tag)
        theme = f'{tag}_theme'
        if dpg.does_item_exist(theme):
            dpg.delete_item(theme)

    def _reset_live_curves(self):
        for tags in self.live_series.values():
            for tag in tags.values():
                self._delete_series(tag)
        self.live_data.clear()
        self.live_series.clear()
        self._last_axis_update = 0.0
        dpg.set_value('progress', 0.0)
        dpg.configure_item('progress', overlay='0 / 0 点')
        for axis in ('nyquist_x', 'nyquist_y', 'magnitude_x', 'magnitude_y',
                     'phase_x', 'phase_y'):
            dpg.set_axis_limits_auto(axis)

    def _record_point(self, bias, channel, frequency, z_real, z_imag, magnitude,
                      phase_deg, index, total):
        key = (bias, channel)
        data = self.live_data.setdefault(key, {
            'freq': [], 'z_real': [], 'neg_z_imag': [], 'magnitude': [], 'phase': [],
        })
        data['freq'].append(frequency)
        data['z_real'].append(z_real)
        data['neg_z_imag'].append(-z_imag)
        data['magnitude'].append(magnitude)
        data['phase'].append(phase_deg)
        series_label = f'CH {channel}, {bias:+g} V' if bias is not None else f'CH {channel}'
        if key not in self.live_series:
            color = self.PALETTE[len(self.live_series) % len(self.PALETTE)]
            self.live_series[key] = {
                'nyquist': dpg.add_line_series(data['z_real'], data['neg_z_imag'],
                                                label=series_label, parent='nyquist_y'),
                'magnitude': dpg.add_line_series(data['freq'], data['magnitude'],
                                                 label=series_label, parent='magnitude_y'),
                'phase': dpg.add_line_series(data['freq'], data['phase'],
                                             label=series_label, parent='phase_y'),
            }
            for tag in self.live_series[key].values():
                self._style_series(tag, color)
        else:
            tags = self.live_series[key]
            dpg.set_value(tags['nyquist'], [data['z_real'], data['neg_z_imag']])
            dpg.set_value(tags['magnitude'], [data['freq'], data['magnitude']])
            dpg.set_value(tags['phase'], [data['freq'], data['phase']])
        prefix = f'{bias:+g} V · ' if bias is not None else ''
        dpg.set_value('status', f'{prefix}CH {channel}: {index}/{total} 点，{frequency:.1f} Hz')
        # 进度条 (偏压序列时叠加显示第几个偏压的进度)
        dpg.set_value('progress', index / total if total else 0.0)
        dpg.configure_item('progress', overlay=f'{index} / {total} 点')
        # 坐标轴自适应开销较大: 原先每个频点都对 6 个轴重算一次 (94 点约 564 次),
        # 扫描过程中会明显掉帧。改为按时间节流, 最多每 0.25 s 更新一次。
        now = time.monotonic()
        if now - self._last_axis_update >= 0.25 or index >= total:
            self._last_axis_update = now
            for axis in ('nyquist_x', 'nyquist_y', 'magnitude_x', 'magnitude_y',
                         'phase_x', 'phase_y'):
                dpg.set_axis_limits_auto(axis)

    # 曲线配色：白底上要压暗一档，否则黄／浅绿几乎看不见
    PALETTE = [(214, 63, 48), (37, 122, 190), (32, 160, 88), (140, 74, 172),
               (196, 148, 12), (18, 152, 138), (206, 108, 22), (110, 122, 136)]

    def _style_series(self, series_tag, color, scatter=False):
        """给曲线着色。

        DearPyGui 2.x 的 add_line_series / add_scatter_series 不再接受 color= 参数
        （那是 1.x 写法，传入会直接抛 SystemError），只能通过绑定主题设置颜色。
        """
        rgba = tuple(color[:3]) + (255,)
        theme_tag = f'{series_tag}_theme'
        with dpg.theme(tag=theme_tag):
            with dpg.theme_component(dpg.mvLineSeries):
                dpg.add_theme_color(dpg.mvPlotCol_Line, rgba,
                                    category=dpg.mvThemeCat_Plots)
            if scatter:
                with dpg.theme_component(dpg.mvScatterSeries):
                    dpg.add_theme_color(dpg.mvPlotCol_MarkerOutline, rgba,
                                        category=dpg.mvThemeCat_Plots)
                    dpg.add_theme_color(dpg.mvPlotCol_MarkerFill, rgba,
                                        category=dpg.mvThemeCat_Plots)
        dpg.bind_item_theme(series_tag, theme_tag)
        return theme_tag

    def _fit_latest_clicked(self, sender=None, app_data=None, user_data=None):
        dpg.set_value('status', '正在拟合最新数据…')
        threading.Thread(target=self._fit_worker, daemon=True).start()

    def _resolved_output_dir(self):
        out = dpg.get_value('output_dir').strip() or './data'
        path = Path(out).expanduser()
        if not path.is_absolute():
            path = CONFIG_PATH.parent / path
        return str(path.resolve())

    def _fit_worker(self):
        try:
            csvs = find_latest_csv(self._resolved_output_dir())
            results = []
            for path in csvs[:20]:
                try:
                    results.append(fit_csv(path))
                except Exception as exc:
                    logging.getLogger(__name__).warning('拟合失败 %s: %s', path, exc)
            self.events.put(('fit_done', results))
        except Exception as exc:
            logging.getLogger(__name__).exception('自动拟合失败')
            self.events.put(('fit_done', [('error', str(exc))]))

    def _clear_fit_table(self):
        """清空拟合参数表格的所有数据行 (保留表头列定义)。"""
        for row in list(dpg.get_item_children('fit_table', slot=1) or []):
            if dpg.does_item_exist(row):
                dpg.delete_item(row)

    def _add_fit_row(self, cells):
        with dpg.table_row(parent='fit_table'):
            for text in cells:
                dpg.add_text(str(text))

    def _refresh_fit_table(self, results):
        """把拟合结果填充到参数表格, 便于横向比较多组数据。"""
        self._clear_fit_table()
        for r in results:
            if r.get('shorted'):
                self._add_fit_row([r['label'], '短路/失效',
                                   f"{r['short_z']:.4g}", '—', '—', '—', '—',
                                   f"相位 {r['short_ph']:.1f}°"])
                continue
            fit = r['best_fit']
            if 'C' in fit:
                cq, n = f"{fit['C']:.4g}", '—'
                tau = f"{fit['tau']:.4g}"
            else:
                cq, n = f"{fit['Q']:.4g}", f"{fit['n']:.3f}"
                tau = '—'
            self._add_fit_row([r['label'], fit['model'],
                               f"{fit['Rs']:.4g}", f"{fit['Rp']:.4g}",
                               cq, n, tau, f"{fit['norm_error']:.2%}"])

    def _export_fit_report(self, sender=None, app_data=None, user_data=None):
        """把当前拟合结果导出为 CSV 报告到输出目录。"""
        if not self.fit_results:
            self._show_message('无法导出', '当前没有拟合结果，请先执行拟合。')
            return
        try:
            out_dir = Path(self._resolved_output_dir())
            out_dir.mkdir(parents=True, exist_ok=True)
            stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            path = out_dir / f'拟合报告_{stamp}.csv'
            with open(path, 'w', newline='', encoding='utf-8-sig') as f:
                writer = csv.writer(f)
                writer.writerow(('数据集', '模型', 'Rs [ohm]', 'Rp [ohm]',
                                 'C [F]', 'Q', 'n', 'tau [s]', 'Rs_upper_note',
                                 '归一化误差', '数据点数', '频率范围下限 Hz',
                                 '频率范围上限 Hz', '数据文件'))
                for r in self.fit_results:
                    if r.get('shorted'):
                        writer.writerow((r['label'], '短路/失效', f"{r['short_z']:.6g}",
                                         '', '', '', '', '', '',
                                         f"相位中位 {r['short_ph']:.2f}°",
                                         r['n_points'], r['freq_range'][0],
                                         r['freq_range'][1], r['path']))
                        continue
                    fit = r['best_fit']
                    writer.writerow((r['label'], fit['model'],
                                     f"{fit['Rs']:.6g}", f"{fit['Rp']:.6g}",
                                     f"{fit.get('C', ''):.6g}" if 'C' in fit else '',
                                     f"{fit.get('Q', ''):.6g}" if 'Q' in fit else '',
                                     f"{fit.get('n', ''):.6g}" if 'n' in fit else '',
                                     f"{fit.get('tau', ''):.6g}" if 'tau' in fit else '',
                                     'Rs <= min(Re Z)',
                                     f"{fit['norm_error']:.6g}", r['n_points'],
                                     r['freq_range'][0], r['freq_range'][1], r['path']))
            dpg.set_value('fit_export_hint', f'已导出：{path.name}')
            self._show_message('导出成功', f'拟合报告已保存到：\n{path}')
        except Exception as exc:
            self._show_message('导出失败', str(exc))

    def _show_fit_results(self, results):
        for tag in self.fit_series:
            self._delete_series(tag)
        self.fit_series = []
        if results and isinstance(results[0], tuple) and results[0][0] == 'error':
            dpg.set_value('fit_summary', f'拟合失败: {results[0][1]}')
            dpg.set_value('status', '拟合失败')
            self.fit_results = []
            self._clear_fit_table()
            return
        self.fit_results = results
        self._refresh_fit_table(results)
        dpg.set_value('fit_export_hint', '')
        lines = []
        for i, r in enumerate(results):
            color = self.PALETTE[i % len(self.PALETTE)]
            if r.get('shorted'):
                lines.append(f"[{r['label']}] ⚠ 短路/失效: |Z| 中位 = "
                             f"{r['short_z']:.3g} Ω, 相位中位 = {r['short_ph']:.1f}°"
                             f'（未拟合，疑似样品击穿或接触短路）')
                continue
            fit = r['best_fit']
            if 'C' in fit:
                param_text = (f"Rs={fit['Rs']:.4g} Ω, Rp={fit['Rp']:.4g} Ω, "
                              f"C={fit['C']:.4g} F, τ={fit['tau']:.3g} s")
            else:
                param_text = (f"Rs={fit['Rs']:.4g} Ω, Rp={fit['Rp']:.4g} Ω, "
                              f"Q={fit['Q']:.3g}, n={fit['n']:.3f}")
            lines.append(f"[{r['label']}] {fit['model']}: {param_text}, "
                         f"误差={fit['norm_error']:.2%}")
            for tag, scatter in (
                    (dpg.add_scatter_series(
                        r['z'].real.tolist(), (-r['z'].imag).tolist(),
                        label=f"{r['label']} 数据", parent='fit_nyq_y'), True),
                    (dpg.add_line_series(
                        fit['z_fit'].real.tolist(), (-fit['z_fit'].imag).tolist(),
                        label=f"{r['label']} 拟合", parent='fit_nyq_y'), False),
                    (dpg.add_scatter_series(
                        r['freq'].tolist(), np.abs(r['z']).tolist(),
                        label=f"{r['label']} 数据", parent='fit_bode_y'), True),
                    (dpg.add_line_series(
                        fit['freq_fit'].tolist(), np.abs(fit['z_fit']).tolist(),
                        label=f"{r['label']} 拟合", parent='fit_bode_y'), False),
            ):
                self.fit_series.append(tag)
                self._style_series(tag, color, scatter=scatter)
        for axis in ('fit_nyq_x', 'fit_nyq_y', 'fit_bode_x', 'fit_bode_y'):
            dpg.set_axis_limits_auto(axis)
        dpg.set_value('fit_summary', '\n'.join(lines) if lines else '没有可拟合的数据。')
        dpg.set_value('status', f'拟合完成: {len(results)} 个数据集')

    def _answer_prompt(self, sender, app_data, user_data):
        if self.prompt_result is not None:
            self.prompt_result['continue'] = bool(user_data)
        if self.prompt_response is not None:
            self.prompt_response.set()
        self.prompt_response = None
        self.prompt_result = None
        dpg.configure_item('calibration_dialog', show=False)

    def _show_message(self, title, message):
        dpg.set_value('message_title', title)
        dpg.set_value('message_text', message)
        dpg.configure_item('message_dialog', show=True)

    def _append_log(self, message):
        current = dpg.get_value('log_text')
        dpg.set_value('log_text', (current + message + '\n')[-60000:])

    def run(self):
        try:
            while dpg.is_dearpygui_running():
                # 单个事件处理异常不应终止整个界面循环（否则扫描中一次绘图失败就退出程序）
                try:
                    self._poll_events()
                except Exception:
                    logging.getLogger(__name__).exception('界面事件处理失败')
                dpg.render_dearpygui_frame()
        finally:
            logging.getLogger().removeHandler(self.log_handler)
            dpg.destroy_context()


def main():
    EISGui().run()


if __name__ == '__main__':
    main()
