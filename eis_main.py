"""
阻抗谱 (EIS) 自动化测量系统主入口.

用法:
    # 无硬件仿真测试 (验证多通道 + 校准逻辑)
    python eis_main.py --simulator

    # 真实 HP 4284A + 继电器板, 完整流程 (校准 + 测量)
    python eis_main.py --config eis_config.yaml --run

    # 只做每通道开路/短路校准
    python eis_main.py --config eis_config.yaml --calibrate

    # 只做测量 (使用已保存的校准)
    python eis_main.py --config eis_config.yaml --run --no-calibrate

测量链: 镀金探针 → PCB继电器板 → HP 4284A (2线法, 每通道校准)
"""
import argparse
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import yaml

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger('eis_main')


def load_config(path: str) -> dict:
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def create_instrument(config: dict):
    """按配置创建阻抗分析仪实例."""
    itype = config['instrument']['type'].lower()
    if itype == 'simulator':
        from drivers.simulator import EISSimulator
        logger.info("Using EIS simulator (no hardware)")
        return EISSimulator(config)
    elif itype in ('hp4284a', 'hp4285a', 'agilent4284a'):
        from drivers.hp4284a import HP4284A
        addr = config['instrument']['address']
        timeout = config['instrument'].get('timeout', 30)
        function = config['instrument'].get('function', 'ztd')
        logger.info(f"Connecting to HP 4284A at {addr}")
        return HP4284A(addr, timeout, function)
    else:
        raise ValueError(f"Unknown instrument type: {itype}")


def create_relay(config: dict):
    """按配置创建继电器板; 未启用时返回 None (单器件直连)."""
    rb = config.get('relay_board', {})
    if not rb.get('enabled', False):
        logger.info("Relay board disabled (direct connection)")
        return None
    from drivers.relay_board import RelayBoard
    return RelayBoard(
        port=rb['port'],
        baudrate=rb.get('baudrate', 115200),
        cmd_on=rb.get('cmd_on', 'CH{ch} ON\r'),
        cmd_off=rb.get('cmd_off', 'CH{ch} OFF\r'),
        cmd_all_off=rb.get('cmd_all_off', 'ALLOFF\r'),
        settle_time=rb.get('settle_time', 0.2),
        n_channels=rb.get('n_channels', 8),
    )


def run_calibration(instrument, relay, config: dict, out_dir: str,
                    prompt_fn=input, check_cancel=None):
    """
    对每个配置通道做开路/短路校准.

    开路校准: 探针悬空 (不接触器件)
    短路校准: 探针短接 (或短路夹具)

    校准在探针尖端进行, 需人工配合: 程序在每个通道切换后暂停等待,
    用户将探针置于开路/短路状态后按回车继续.
    """
    from cal.calibration import CalibrationStore, ChannelCalibration
    from measurements.eis import build_freq_axis

    cal_cfg = config['measurement']['calibration']
    eis_cfg = config['measurement']['eis']
    freqs = build_freq_axis(
        eis_cfg['freq_start'], eis_cfg['freq_stop'],
        eis_cfg['points_per_decade'],
    )
    vac = float(eis_cfg['v_ac'])
    vdc = float(eis_cfg['v_dc'])
    avg = int(cal_cfg['averaging'])
    settle = float(cal_cfg['settle_time'])

    channels = config['measurement']['channels']
    store = CalibrationStore()

    def measure_calibration_spectrum():
        values = []
        for f in freqs:
            if check_cancel is not None:
                check_cancel()
            values.append(instrument.measure_complex_z(f, vac, vdc, averaging=avg))
        return np.asarray(values)

    for ch in channels:
        if check_cancel is not None:
            check_cancel()
        if relay is not None:
            relay.select(ch)
        elif hasattr(instrument, 'set_channel'):
            instrument.set_channel(ch)

        cal = ChannelCalibration(ch, freqs)

        # 开路校准
        prompt_fn(f"通道 {ch}: 请将探针置于【开路】状态 (悬空), 准备好后继续。")
        z_open = measure_calibration_spectrum()
        cal.set_open(z_open)
        logger.info(f"Channel {ch}: open calibration done")

        # 短路校准
        prompt_fn(f"通道 {ch}: 请将探针置于【短路】状态 (短接), 准备好后继续。")
        z_short = measure_calibration_spectrum()
        cal.set_short(z_short)
        logger.info(f"Channel {ch}: short calibration done")

        store.add(cal)

    if relay is not None:
        relay.all_off()

    os.makedirs(out_dir, exist_ok=True)
    store.save(os.path.join(out_dir, 'calibration.json'))
    return store


def run_eis(instrument, relay, config: dict, out_dir: str,
            calibration_store=None, on_point=None):
    """执行多通道 EIS 扫描并保存结果."""
    from measurements.eis import run_eis_all
    from utils.data_io import ensure_output_dir, save_eis_channel, plot_eis_all

    out_dir = ensure_output_dir(config)

    results = run_eis_all(instrument, relay, config, calibration_store,
                          on_point=on_point)

    if config['output']['save_raw']:
        for r in results:
            save_eis_channel(r, out_dir)

    if config['output']['save_plots']:
        plot_eis_all(results, out_dir)

    logger.info(f"EIS complete: {len(results)} channels → {out_dir}")
    return results


def main():
    # 在源码模式下以脚本所在目录为基准；打包后以 exe 所在目录为基准。
    app_dir = Path(sys.executable).resolve().parent if getattr(sys, 'frozen', False) \
        else Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="阻抗谱 (EIS) 自动化测量")
    parser.add_argument('--config', default=str(app_dir / 'eis_config.yaml'))
    parser.add_argument('--simulator', action='store_true',
                        help='使用仿真器 (无硬件)')
    parser.add_argument('--run', action='store_true', help='执行 EIS 测量')
    parser.add_argument('--calibrate', action='store_true',
                        help='执行每通道开路/短路校准')
    parser.add_argument('--no-calibrate', action='store_true',
                        help='跳过校准 (使用已保存的校准文件)')
    args = parser.parse_args()

    config_path = Path(args.config).expanduser().resolve()
    config = load_config(str(config_path))
    # 相对输出路径相对于配置文件位置，双击 exe 时结果也会写入程序目录。
    output_dir = Path(config['output']['directory']).expanduser()
    if not output_dir.is_absolute():
        output_dir = config_path.parent / output_dir
    config['output']['directory'] = str(output_dir.resolve())
    if args.simulator:
        config['instrument']['type'] = 'simulator'

    instrument = create_instrument(config)
    relay = create_relay(config)

    out_dir = config['output']['directory']
    os.makedirs(out_dir, exist_ok=True)
    cal_file = os.path.join(out_dir, 'calibration.json')

    try:
        calibration_store = None
        if args.calibrate or (args.run and not args.no_calibrate):
            calibration_store = run_calibration(instrument, relay, config, out_dir)
        elif args.run and os.path.exists(cal_file):
            from cal.calibration import CalibrationStore
            calibration_store = CalibrationStore.load(cal_file)

        if args.run:
            run_eis(instrument, relay, config, out_dir, calibration_store)

    finally:
        if relay is not None:
            relay.close()
        instrument.close()


if __name__ == '__main__':
    main()
